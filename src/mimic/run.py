"""Run a learned skill on a batch of tasks and measure how well the agent did.

    uv run python -m mimic.run --skill enter_invoice --data sandbox/data/invoices.json
    uv run python -m mimic.run --skill enter_invoice --layout v2      # redesigned UI
    uv run python -m mimic.run --skill enter_invoice --show --limit 5 # watch it work

Success is judged against the app's own records (/api/entries), not the agent's opinion.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from playwright.sync_api import sync_playwright
from psycopg.types.json import Jsonb

from mimic.agent import ReplayAgent, StepResult
from mimic.approval import ConsoleApprover, build_aliases
from mimic.config import get_settings
from mimic.healing import Healer, HealMemory
from mimic.llm import OllamaClient
from mimic.mapping import apply_mapping, map_inputs
from mimic.recorder import launch_browser
from mimic.skills import Skill
from mimic.vision import VisionLocator


def _with_layout(url: str, layout: str, **extra) -> str:
    parts = urlsplit(url)
    query = urlencode({"layout": layout, **{k: v for k, v in extra.items() if v is not None}})
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))


def _base(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class StepLogger:
    """Writes step telemetry to Postgres: the training data of the failure risk model."""

    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        self.rows: list[tuple] = []

    def __call__(self, step: StepResult) -> None:
        # How far the UI drifted from the demo for this step: 0 exact, 0.5 fallback, 1 healed.
        layout_change = 1.0 if step.healed else (0.5 if step.fallbacks else 0.0)
        self.rows.append(
            (
                self.run_id,
                step.index,
                step.action,
                step.confidence,
                layout_change,
                step.latency_ms,
                step.success,
                step.strategy,
                Jsonb(step.features) if step.features is not None else None,
                step.risk,
                bool(step.mitigations),
                step.error,
            )
        )

    def flush(self) -> None:
        if not self.enabled or not self.rows:
            return
        try:
            from mimic.db import connect

            with connect() as conn, conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO step_logs (run_id, step_index, action, match_score,
                       layout_change, latency_ms, success, strategy, features, risk,
                       mitigated, error)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    self.rows,
                )
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Telemetry not saved ({type(exc).__name__}).")
        self.rows.clear()


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 0.005
    except (TypeError, ValueError):
        return " ".join(str(a).lower().split()) == " ".join(str(b).lower().split())


def _matches(entry: dict | None, expected: dict) -> bool:
    """Every field the app recorded must match what was expected (case/spacing-insensitive)."""
    if entry is None:
        return False
    shared = [f for f in entry if f in expected]
    return bool(shared) and all(_same(entry[f], expected[f]) for f in shared)


class AppGrader:
    """Grades against what the application really recorded, when it exposes /api/entries
    (the sandbox does). On any other site grading falls back to the agent's own checks."""

    def __init__(self, url: str):
        self.base = _base(url)
        self.available = False

    def reset(self) -> None:
        try:
            r = httpx.delete(f"{self.base}/api/entries", timeout=3)
            self.available = r.status_code < 400
        except httpx.HTTPError:
            self.available = False

    def grade(self, expected_records: list[dict]) -> list[bool] | None:
        if not self.available:
            return None
        entries = httpx.get(f"{self.base}/api/entries", timeout=5).json()
        key = "invoice_number"
        by_key = {str(e.get(key)): e for e in entries if key in e}
        out = []
        for r in expected_records:
            if key in r:
                out.append(_matches(by_key.get(str(r[key])), r))
            else:
                out.append(any(_matches(e, r) for e in entries))
        return out


def run_batch(
    skill: Skill,
    records: list[dict],
    url: str,
    layout: str = "v1",
    show: bool = False,
    log_to_db: bool = True,
    client=None,
    out_dir: Path | None = Path("reports"),
    healer: Healer | None = None,
    approver: ConsoleApprover | None = None,
    chaos: float = 0.0,
    seed_base: int | None = None,
    risk_policy: str = "never",
    risk_model=None,
    truth: list[dict] | None = None,  # what the app should record, if not the records
    titles: list[str] | None = None,  # shown in the plan review (e.g. document names)
    flags: list[dict] | None = None,  # per record: parameter -> reason to double-check
) -> dict:
    settings = get_settings()
    mapping = map_inputs(skill.parameters, list(records[0]), client, settings.planner_model)
    grader = AppGrader(url)
    grader.reset()

    run_id = f"{skill.name}-{layout}-{risk_policy}-{uuid.uuid4().hex[:8]}"
    logger = StepLogger(run_id, enabled=log_to_db)
    results = []
    t0 = time.monotonic()

    with sync_playwright() as p:
        browser = launch_browser(p, headless=not show)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        if show:
            page.set_default_timeout(5000)
        agent = ReplayAgent(
            page,
            on_step=logger,
            healer=healer,
            skill_name=skill.name,
            before_submit=approver.confirm_submit if approver else None,
            risk_policy=risk_policy,
            risk_model=risk_model,
        )

        if approver is not None:
            approver.aliases = build_aliases(skill, mapping)
        expected_records: list[dict] = []  # what each executed task should have recorded
        skipped = 0
        for i, record in enumerate(records, 1):
            inputs = apply_mapping(record, mapping)
            expected = dict(truth[i - 1]) if truth else dict(record)
            record_flags = flags[i - 1] if flags else {}
            if approver is not None:
                title = titles[i - 1] if titles else str(next(iter(record.values()), ""))
                decision, actions = approver.review(
                    i, len(records), inputs, skill.render, title, record_flags
                )
                if decision.action == "quit":
                    print("  Stopped by user.")
                    break
                if decision.action == "skip":
                    skipped += 1
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped by user")
                    continue
                if not truth:
                    for param, value in decision.edits.items():
                        expected[mapping[param]] = value
                if any(a["action"] != "click" and a["value"] in (None, "None") for a in actions):
                    skipped += 1
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped: a value is still missing")
                    continue
            else:
                missing = [p for p, v in inputs.items() if v is None]
                if missing:
                    skipped += 1
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped: missing {missing}")
                    continue
                actions = skill.render(inputs)

            seed = None if seed_base is None else seed_base + i
            page.goto(_with_layout(url, layout, chaos=chaos or None, seed=seed))
            result = agent.run(actions)
            if result.skipped:
                skipped += 1
                print(f"  Task {i:>3}/{len(records)}  ⏭  submit declined by user")
                continue
            results.append(result)
            expected_records.append(expected)
            healed = [s.index for s in result.steps if s.healed]
            mark = "✅" if result.success else f"❌ step {len(result.steps) - 1}: {result.error}"
            if healed:
                strategies = ",".join(s.strategy for s in result.steps if s.healed)
                mark += f"  🩹 healed step {healed} via {strategies}"
            print(f"  Task {i:>3}/{len(records)}  {result.duration_ms:>5} ms  {mark}")
        browser.close()

    logger.flush()
    executed = max(len(expected_records), 1)
    correct = grader.grade(expected_records)
    if correct is None:  # no ground truth on this site: trust the agent's verified steps
        correct = [r.success for r in results]

    steps = [s for r in results for s in r.steps]
    report = {
        "run_id": run_id,
        "skill": skill.name,
        "layout": layout,
        "tasks": len(records),
        "executed": len(expected_records),
        "skipped_by_user": skipped,
        "task_success_rate": round(sum(correct) / executed, 4),
        "graded_by": "application" if grader.available else "agent",
        "agent_reported_success_rate": round(sum(r.success for r in results) / executed, 4),
        "step_success_rate": round(sum(s.success for s in steps) / max(len(steps), 1), 4),
        "avg_task_ms": int(sum(r.duration_ms for r in results) / max(len(results), 1)),
        "total_s": round(time.monotonic() - t0, 1),
        "retries": sum(s.attempts - 1 for s in steps),
        "strategies": dict(Counter(s.strategy or "failed" for s in steps)),
        "failures": dict(Counter(r.error for r in results if not r.success)),
        "healing": healer.stats.as_dict() if healer else None,
        "chaos": chaos,
        "risk_policy": risk_policy,
        "failed_actions": sum(1 for s in steps if s.features is not None and not s.success),
        "mitigated_steps": sum(1 for s in steps if s.mitigations),
        "approval": approver.stats if approver else None,
        "mapping": mapping,
        "steps": [asdict(s) for s in steps],
    }
    if out_dir:
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    return report


def collect_options(skill: Skill, url: str, layout: str = "v1") -> dict[str, list[str]]:
    """Read the choices of every dropdown the skill uses, from the live page."""
    wanted = [s for s in skill.steps if s.action == "select" and s.parameter]
    if not wanted:
        return {}
    found = {}
    with sync_playwright() as p:
        browser = launch_browser(p, headless=True)
        page = browser.new_page()
        page.goto(_with_layout(url, layout))
        for step in wanted:
            el = step.element
            for sel in filter(None, [el.selector, f'[name="{el.name}"]' if el.name else None]):
                loc = page.locator(sel)
                if loc.count() == 1:
                    found[step.parameter] = loc.evaluate(
                        "el => [...el.options].map(o => o.value || o.text)"
                    )
                    break
        browser.close()
    return found


def records_from_documents(
    skill: Skill, folder: Path, extractor, options: dict, limit: int | None = None
):
    """Extract one record per document. Returns records, titles, flags and truth (if any)."""
    from mimic.extract import LOW_CONFIDENCE, field_specs, list_documents

    specs = field_specs(skill, options)
    docs = list_documents(folder)[:limit]
    truth_file = folder / "truth.json"
    truth_map = json.loads(truth_file.read_text()) if truth_file.exists() else {}
    records, titles, flags, truth = [], [], [], []
    print(f"Reading {len(docs)} documents with {extractor.model}...")
    for doc in docs:
        ex = extractor.extract(doc, specs)
        record_flags = {}
        for param, got in ex.fields.items():
            if got.value is None:
                record_flags[param] = got.note or "not found"
            elif got.confidence < LOW_CONFIDENCE:
                record_flags[param] = got.note or f"low confidence {got.confidence:.2f}"
        status = f"{len(record_flags)} field(s) to check" if record_flags else "all fields read"
        print(f"  {doc.name:<24} {ex.seconds:5.1f} s  {status}")
        records.append(ex.inputs())
        titles.append(doc.name)
        flags.append(record_flags)
        if doc.name in truth_map:
            truth.append({k: v for k, v in truth_map[doc.name].items() if k != "layout"})
    return records, titles, flags, (truth if len(truth) == len(records) else None)


def _load_matcher():
    """The fine-tuned element matcher, if it was trained and its extra is installed."""
    from mimic.matcher import MODEL_DIR

    if not MODEL_DIR.exists():
        return None
    try:
        from mimic.matcher import ElementMatcher

        matcher = ElementMatcher.load(MODEL_DIR)
        print("Element matcher: loaded (fine-tuned, Hugging Face)")
        return matcher
    except Exception as exc:  # noqa: BLE001
        print(f"  ! element matcher not loaded ({type(exc).__name__}): uv sync --extra matcher")
        return None


def print_report(report: dict) -> None:
    print(f"\n=== {report['skill']} on layout {report['layout']} ===")
    print(
        f"Task success      : {report['task_success_rate']:.0%}  "
        f"({report['executed']} executed, {report['skipped_by_user']} skipped by you)"
    )
    print(f"Step success      : {report['step_success_rate']:.0%}")
    print(f"Avg time per task : {report['avg_task_ms']} ms   (total {report['total_s']} s)")
    print(f"Retries           : {report['retries']}")
    print(f"Locator strategies: {report['strategies']}")
    if report["healing"]:
        print(f"Self-healing      : {report['healing']}")
    if report.get("chaos"):
        print(
            f"Chaos {report['chaos']:.0%}, policy {report['risk_policy']}: "
            f"{report['failed_actions']} failed actions, "
            f"{report['mitigated_steps']} mitigated steps"
        )
    if report["approval"]:
        print(f"Your decisions    : {report['approval']}")
    if report["failures"]:
        print(f"Failures          : {report['failures']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a skill on a batch of tasks.")
    parser.add_argument("--skill", required=True)
    parser.add_argument("--data", type=Path, default=Path("sandbox/data/invoices.json"))
    parser.add_argument(
        "--docs", type=Path, default=None, help="Folder of PDFs/images to read the inputs from"
    )
    parser.add_argument("--url", default=None, help="Defaults to the skill's start URL")
    parser.add_argument("--layout", default="v1", choices=["v1", "v2"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Show the browser")
    parser.add_argument("--no-heal", action="store_true", help="Disable self-healing (baseline)")
    parser.add_argument("--no-vision", action="store_true", help="Heal without the VLM")
    parser.add_argument("--debug", action="store_true", help="Print healing decisions")
    parser.add_argument("--no-matcher", action="store_true", help="Skip the fine-tuned matcher")
    parser.add_argument("--yes", action="store_true", help="Skip plan review (no human check)")
    parser.add_argument("--chaos", type=float, default=0.0, help="Sandbox flakiness, 0 to 1")
    parser.add_argument("--seed", type=int, default=None, help="Reproducible chaos (base seed)")
    parser.add_argument("--risk-policy", choices=["never", "always", "predicted"], default="never")
    parser.add_argument(
        "--confirm-submit", action="store_true", help="Ask before each submit (use with --show)"
    )
    args = parser.parse_args()

    from mimic.store import SkillStore

    skill = SkillStore().get(args.skill)
    if skill is None:
        raise SystemExit(f"Unknown skill '{args.skill}'. Run: python -m mimic.learn list")
    url = args.url or skill.start_url or get_settings().sandbox_url
    titles = flags = truth = None
    if args.docs:
        from mimic.extract import DocumentExtractor

        options = collect_options(skill, url, args.layout)
        extractor = DocumentExtractor(OllamaClient(), get_settings().vision_model)
        records, titles, flags, truth = records_from_documents(
            skill, args.docs, extractor, options, args.limit
        )
    else:
        records = json.loads(args.data.read_text(encoding="utf-8"))[: args.limit]

    print(f"Running '{skill.name}' on {len(records)} tasks ({args.layout})...")
    settings = get_settings()
    client = OllamaClient()
    healer = None
    if not args.no_heal:
        healer = Healer(
            embedder=client.embed,
            vision=None if args.no_vision else VisionLocator(client, settings.vision_model),
            memory=HealMemory(use_db=True),
            debug=args.debug,
            matcher=None if args.no_matcher else _load_matcher(),
        )
    approver = None if args.yes else ConsoleApprover(confirm_submit=args.confirm_submit)
    risk_model = None
    if args.risk_policy == "predicted":
        from mimic.risk import RiskModel

        risk_model = RiskModel.load()
    report = run_batch(
        skill,
        records,
        url,
        args.layout,
        args.show,
        client=client,
        healer=healer,
        approver=approver,
        chaos=args.chaos,
        seed_base=args.seed,
        risk_policy=args.risk_policy,
        risk_model=risk_model,
        truth=truth,
        titles=titles,
        flags=flags,
    )
    print_report(report)
    print(f"\nReport saved to reports/{report['run_id']}.json")


if __name__ == "__main__":
    main()
