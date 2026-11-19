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

FIELDS = ["vendor", "invoice_number", "date", "total", "category"]


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


def _matches(entry: dict | None, expected: dict) -> bool:
    if entry is None:
        return False
    for f in FIELDS:
        if f == "total":
            if abs(float(entry[f]) - float(expected[f])) > 0.005:
                return False
        elif str(entry[f]) != str(expected[f]):
            return False
    return True


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
) -> dict:
    settings = get_settings()
    mapping = map_inputs(skill.parameters, list(records[0]), client, settings.planner_model)
    base = _base(url)
    httpx.delete(f"{base}/api/entries").raise_for_status()

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
            expected = dict(record)
            if approver is not None:
                title = str(next(iter(record.values()), ""))
                decision, actions = approver.review(i, len(records), inputs, skill.render, title)
                if decision.action == "quit":
                    print("  Stopped by user.")
                    break
                if decision.action == "skip":
                    skipped += 1
                    print(f"  Task {i:>3}/{len(records)}  ⏭  skipped by user")
                    continue
                for param, value in decision.edits.items():
                    expected[mapping[param]] = value
            else:
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
    entries = {e["invoice_number"]: e for e in httpx.get(f"{base}/api/entries").json()}
    correct = [_matches(entries.get(r["invoice_number"]), r) for r in expected_records]
    executed = max(len(expected_records), 1)

    steps = [s for r in results for s in r.steps]
    report = {
        "run_id": run_id,
        "skill": skill.name,
        "layout": layout,
        "tasks": len(records),
        "executed": len(expected_records),
        "skipped_by_user": skipped,
        "task_success_rate": round(sum(correct) / executed, 4),
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
    parser.add_argument("--url", default=None, help="Defaults to the skill's start URL")
    parser.add_argument("--layout", default="v1", choices=["v1", "v2"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Show the browser")
    parser.add_argument("--no-heal", action="store_true", help="Disable self-healing (baseline)")
    parser.add_argument("--no-vision", action="store_true", help="Heal without the VLM")
    parser.add_argument("--debug", action="store_true", help="Print healing decisions")
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
    records = json.loads(args.data.read_text(encoding="utf-8"))[: args.limit]
    url = args.url or skill.start_url or get_settings().sandbox_url

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
    )
    print_report(report)
    print(f"\nReport saved to reports/{report['run_id']}.json")


if __name__ == "__main__":
    main()
