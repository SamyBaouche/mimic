"""Run a learned skill on a batch of tasks and measure how well the agent did.

    uv run python -m mimic.run --skill enter_invoice --data sandbox/data/invoices.json
    uv run python -m mimic.run --skill enter_invoice --layout v2      # redesigned UI
    uv run python -m mimic.run --skill enter_invoice --show --limit 5 # watch it work

Success is judged against the app's own records (/api/entries), not the agent's opinion.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
from playwright.sync_api import sync_playwright

from mimic.agent import ReplayAgent, StepResult
from mimic.approval import ConsoleApprover, build_aliases, plan_label, plan_rows
from mimic.config import get_settings
from mimic.healing import Healer, HealMemory
from mimic.live import Screencast
from mimic.llm import OllamaClient
from mimic.mapping import apply_mapping, map_inputs
from mimic.outcome import check_signal
from mimic.recorder import launch_browser
from mimic.skills import Skill, secret_env
from mimic.vision import VisionLocator


def _with_layout(url: str, layout: str, **extra) -> str:
    parts = urlsplit(url)
    query = urlencode({"layout": layout, **{k: v for k, v in extra.items() if v is not None}})
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", query, ""))


def _base(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class MissingSecret(ValueError):
    pass


def check_secrets(skill: Skill, record: dict) -> None:
    """Secret fields (passwords...) come from the data or MIMIC_SECRET_<NAME>, never the skill."""
    missing = [p for p in skill.secrets if p not in record and not os.environ.get(secret_env(p))]
    if missing:
        names = ", ".join(secret_env(p) for p in missing)
        raise MissingSecret(
            f"Skill '{skill.name}' types a secret ({', '.join(missing)}) that is never stored. "
            f'Set it for this session: {names} (PowerShell: $env:{secret_env(missing[0])}="...")'
        )


class StepLogger:
    """Writes step telemetry to storage: the training data of the failure risk model."""

    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        self.rows: list[dict] = []

    def __call__(self, step: StepResult) -> None:
        # How far the UI drifted from the demo for this step: 0 exact, 0.5 fallback, 1 healed.
        layout_change = 1.0 if step.healed else (0.5 if step.fallbacks else 0.0)
        self.rows.append(
            {
                "run_id": self.run_id,
                "step_index": step.index,
                "action": step.action,
                "match_score": step.confidence,
                "layout_change": layout_change,
                "latency_ms": step.latency_ms,
                "success": step.success,
                "strategy": step.strategy,
                "features": step.features,
                "risk": step.risk,
                "mitigated": bool(step.mitigations),
                "error": step.error,
            }
        )

    def flush(self) -> None:
        if not self.enabled or not self.rows:
            return
        try:
            from mimic.storage import get_storage

            get_storage().log_steps(self.rows)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! Telemetry not saved ({type(exc).__name__}: {exc}).")
        self.rows.clear()


def emit_is_live(emit) -> bool:
    """Only stream frames when someone is watching (the dashboard sets this attribute)."""
    return bool(getattr(emit, "wants_frames", False))


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
    emit=None,  # callable(event dict): live events for the dashboard (see events.py)
) -> dict:
    emit = emit or (lambda event: None)
    settings = get_settings()
    secrets = skill.secrets
    check_secrets(skill, records[0])
    plain = [p for p in skill.parameters if p not in secrets]
    fields = [f for f in records[0] if f not in secrets]
    # Values typed in the demo stand in for columns your data does not have (e.g. a user
    # name on a sign-in page). Uploads and secrets never do.
    demo_values = {
        st.parameter: st.value
        for st in skill.steps
        if st.parameter and st.value is not None and st.action not in ("upload",) and not st.secret
    }
    aliases = {
        st.parameter: [st.element.label or "", st.element.name or "", st.element.placeholder or ""]
        for st in skill.steps
        if st.parameter
    }
    mapping = map_inputs(
        plain, fields, client, settings.planner_model, aliases=aliases, optional=set(demo_values)
    )
    from_demo = {p: demo_values[p] for p in plain if p not in mapping}
    for p, v in from_demo.items():
        note = f"No column for '{p}': every task uses '{v}' from your demo."
        print(f"  ! {note}")
        emit({"type": "status", "message": note})
    grader = AppGrader(url)
    grader.reset()

    run_id = f"{skill.name}-{layout}-{risk_policy}-{uuid.uuid4().hex[:8]}"
    logger = StepLogger(run_id, enabled=log_to_db)
    results = []
    t0 = time.monotonic()
    current: dict = {"index": 0, "actions": []}
    emit(
        {
            "type": "run_started",
            "run_id": run_id,
            "skill": skill.name,
            "total": len(records),
            "layout": layout,
            "url": url,
        }
    )

    def on_step(step: StepResult) -> None:
        logger(step)
        # Step numbers are ids, not positions: an edited skill can skip some.
        action = next((a for a in current["actions"] if a["index"] == step.index), {})
        emit(
            {
                "type": "step",
                "task": current["index"],
                "index": step.index,
                "action": step.action,
                "label": plan_label(action) if action else "",
                "success": step.success,
                "strategy": step.strategy,
                "healed": step.healed,
                "confidence": step.confidence,
                "latency_ms": step.latency_ms,
                "error": step.error,
            }
        )

    with sync_playwright() as p:
        browser = launch_browser(p, headless=not show)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        if show:
            page.set_default_timeout(5000)
        live = Screencast(page, emit).start() if emit_is_live(emit) else None
        agent = ReplayAgent(
            page,
            on_step=on_step,
            healer=healer,
            skill_name=skill.name,
            before_submit=approver.confirm_submit if approver else None,
            risk_policy=risk_policy,
            risk_model=risk_model,
        )

        if approver is not None:
            approver.aliases = build_aliases(skill, mapping)
        expected_records: list[dict] = []  # what each executed task should have recorded
        executed_index: list[int] = []  # record number (1-based) of each executed task
        rows: dict[int, dict] = {}  # what happened to each input row, for the results file
        skipped = 0
        for i, record in enumerate(records, 1):
            if approver is not None and not getattr(approver, "checkpoint", lambda: True)():
                print("  Stopped by user.")
                break
            inputs = from_demo | apply_mapping(record, mapping)
            inputs |= {p: record[p] for p in secrets if p in record}
            expected = dict(truth[i - 1]) if truth else dict(record)
            for p in secrets:  # the app never shows a password back: do not grade it
                expected.pop(p, None)
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
                    rows[i] = {"index": i, "status": "skipped", "error": "skipped by you"}
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped by user")
                    emit(
                        {
                            "type": "task_done",
                            "index": i,
                            "status": "skipped",
                            "reason": "skipped by you",
                        }
                    )
                    continue
                if not truth:
                    for param, value in decision.edits.items():
                        expected[mapping[param]] = value
                if any(a["action"] != "click" and a["value"] in (None, "None") for a in actions):
                    skipped += 1
                    rows[i] = {"index": i, "status": "skipped", "error": "a value is missing"}
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped: a value is still missing")
                    emit(
                        {
                            "type": "task_done",
                            "index": i,
                            "status": "skipped",
                            "reason": "a value is still missing",
                        }
                    )
                    continue
            else:
                missing = [p for p, v in inputs.items() if v is None]
                if missing:
                    skipped += 1
                    rows[i] = {"index": i, "status": "skipped", "error": f"missing {missing}"}
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped: missing {missing}")
                    emit(
                        {
                            "type": "task_done",
                            "index": i,
                            "status": "skipped",
                            "reason": f"missing {', '.join(missing)}",
                        }
                    )
                    continue
                actions = skill.render(inputs)

            if not truth:
                # Your column names ("Invoice number") are not the app's ("invoice_number"):
                # also expect each value under the name of the field it was typed into.
                expected |= {
                    a["element"].name: a["value"]
                    for a in actions
                    if a["action"] in ("fill", "select") and a["element"].name and not a["secret"]
                }
            seed = None if seed_base is None else seed_base + i
            title = titles[i - 1] if titles else str(next(iter(record.values()), ""))
            current.update(index=i, actions=actions)
            emit(
                {
                    "type": "task_started",
                    "index": i,
                    "total": len(records),
                    "title": title,
                    "plan": plan_rows(actions, record_flags),
                }
            )
            page.goto(_with_layout(url, layout, chaos=chaos or None, seed=seed))
            result = agent.run(actions)
            if result.success and skill.success is not None:
                # Every step worked: now check the page looks like the end of the demo.
                problem = check_signal(page, skill.success)
                if problem:
                    result.success, result.error = False, problem
            if result.skipped:
                skipped += 1
                rows[i] = {"index": i, "status": "skipped", "error": "submit declined by you"}
                print(f"  Task {i:>3}/{len(records)}  ⏭  submit declined by user")
                emit(
                    {
                        "type": "task_done",
                        "index": i,
                        "status": "skipped",
                        "reason": "submit declined by you",
                    }
                )
                continue
            results.append(result)
            expected_records.append(expected)
            executed_index.append(i)
            healed = [s.index for s in result.steps if s.healed]
            mark = "✅" if result.success else f"❌ step {len(result.steps) - 1}: {result.error}"
            if healed:
                strategies = ",".join(s.strategy for s in result.steps if s.healed)
                mark += f"  🩹 healed step {healed} via {strategies}"
            print(f"  Task {i:>3}/{len(records)}  {result.duration_ms:>5} ms  {mark}")
            emit(
                {
                    "type": "task_done",
                    "index": i,
                    "status": "ok" if result.success else "failed",
                    "ms": result.duration_ms,
                    "error": result.error,
                    "healed": [s.strategy for s in result.steps if s.healed],
                    "done": len(results),
                    "succeeded": sum(r.success for r in results),
                    "skipped": skipped,
                    "avg_ms": int(sum(r.duration_ms for r in results) / len(results)),
                }
            )
        if live is not None:
            page.wait_for_timeout(500)  # let the last submit reach the screen
            live.stop()
        browser.close()

    logger.flush()
    executed = max(len(expected_records), 1)
    # Ground truth, best first: the app's own records (sandbox invoices, skills without a
    # learned signal), else the confirmation learned from the demo, else the agent's checks.
    correct = grader.grade(expected_records) if skill.success is None else None
    graded_by = (
        "application"
        if correct is not None
        else ("confirmation" if skill.success is not None else "agent")
    )
    if correct is None:
        correct = [r.success for r in results]
    for ok, i, r in zip(correct, executed_index, results, strict=True):
        error = None if ok else (r.error or "the application did not record it")
        rows[i] = {
            "index": i,
            "status": "done" if ok else "failed",
            "error": error,
            "ms": r.duration_ms,
        }

    steps = [s for r in results for s in r.steps]
    report = {
        "run_id": run_id,
        "skill": skill.name,
        "layout": layout,
        "tasks": len(records),
        "executed": len(expected_records),
        "skipped_by_user": skipped,
        "task_success_rate": round(sum(correct) / executed, 4),
        "graded_by": graded_by,
        "rows": [rows[i] for i in sorted(rows)],
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
    emit({"type": "run_done", "report": {k: v for k, v in report.items() if k != "steps"}})
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
    skill: Skill, folder: Path, extractor, options: dict, limit: int | None = None, emit=None
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
        if emit:
            emit(
                {
                    "type": "reading",
                    "index": len(records) + 1,
                    "total": len(docs),
                    "name": doc.name,
                    "seconds": round(ex.seconds, 1),
                    "to_check": len(record_flags),
                }
            )
        # A file upload in the skill (e.g. "Receipt") gets the document that was just read.
        uploads = {st.parameter: str(doc.resolve()) for st in skill.steps if st.action == "upload"}
        records.append({**ex.inputs(), **uploads})
        titles.append(doc.name)
        flags.append(record_flags)
        if doc.name in truth_map:
            truth.append({k: v for k, v in truth_map[doc.name].items() if k != "layout"})
    return records, titles, flags, (truth if len(truth) == len(records) else None)


def _load_matcher():
    """The fine-tuned element matcher, if it was trained and its extra is installed."""
    try:
        import torch  # noqa: F401  (the matcher extra)
    except ImportError:
        return None
    from mimic.matcher import ElementMatcher, find_model

    path = find_model()
    if path is None:
        return None
    try:
        matcher = ElementMatcher.load(path)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay a skill on a batch of tasks.")
    parser.add_argument("--skill", required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("sandbox/data/invoices.json"),
        help="Tasks to do, one per row: .xlsx, .csv or .json",
    )
    parser.add_argument("--sheet", default=None, help="Excel sheet name (default: the first)")
    parser.add_argument(
        "--only-failed", action="store_true", help="With a results file: redo what is not done"
    )
    parser.add_argument(
        "--results", type=Path, default=None, help="Where to write the results (.csv or .xlsx)"
    )
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
    args = parser.parse_args(argv)

    from mimic.store import SkillStore

    skill = SkillStore().get(args.skill)
    if skill is None:
        raise SystemExit(f"Unknown skill '{args.skill}'. See your skills with: mimic skills")
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
        from mimic.tables import load_records, only_failed, strip_results

        try:
            records = load_records(args.data, args.sheet)
            if args.only_failed:
                records = only_failed(records)
        except (OSError, ValueError, KeyError) as exc:
            raise SystemExit(f"Cannot read {args.data}: {exc}") from None
        records = strip_results(records)[: args.limit]
        if not records:
            raise SystemExit(f"Nothing to do in {args.data}.")

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
    try:
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
    except MissingSecret as exc:
        raise SystemExit(f"\n{exc}") from None
    print_report(report)
    print(f"\nReport saved to reports/{report['run_id']}.json")

    from mimic.tables import write_results

    path = write_results(
        args.results or Path("reports") / f"{report['run_id']}-results.csv", records, report["rows"]
    )
    left = sum(r["status"] != "done" for r in report["rows"]) + len(records) - len(report["rows"])
    print(f"Results per row: {path}")
    if left:
        print(
            f"{left} row(s) not done. Retry them:"
            f" mimic run {skill.name} --data {path} --only-failed"
        )


if __name__ == "__main__":
    main()
