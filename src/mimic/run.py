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

from mimic.agent import ReplayAgent, StepResult
from mimic.config import get_settings
from mimic.llm import OllamaClient
from mimic.mapping import apply_mapping, map_inputs
from mimic.recorder import launch_browser
from mimic.skills import Skill

FIELDS = ["vendor", "invoice_number", "date", "total", "category"]


def _with_layout(url: str, layout: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path or "/", urlencode({"layout": layout}), "")
    )


def _base(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class StepLogger:
    """Writes step telemetry to Postgres (training data for the Sprint 7 failure predictor)."""

    def __init__(self, run_id: str, enabled: bool = True):
        self.run_id = run_id
        self.enabled = enabled
        self.rows: list[tuple] = []

    def __call__(self, step: StepResult, layout_change: float = 0.0) -> None:
        self.rows.append(
            (
                self.run_id,
                step.index,
                step.action,
                step.confidence,
                layout_change,
                step.latency_ms,
                step.success,
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
                       layout_change, latency_ms, success) VALUES (%s,%s,%s,%s,%s,%s,%s)""",
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
) -> dict:
    settings = get_settings()
    mapping = map_inputs(skill.parameters, list(records[0]), client, settings.planner_model)
    base = _base(url)
    httpx.delete(f"{base}/api/entries").raise_for_status()

    run_id = f"{skill.name}-{layout}-{uuid.uuid4().hex[:8]}"
    logger = StepLogger(run_id, enabled=log_to_db)
    results = []
    t0 = time.monotonic()

    with sync_playwright() as p:
        browser = launch_browser(p, headless=not show)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        if show:
            page.set_default_timeout(5000)
        agent = ReplayAgent(page, on_step=logger)

        for i, record in enumerate(records, 1):
            page.goto(_with_layout(url, layout))
            actions = skill.render(apply_mapping(record, mapping))
            result = agent.run(actions)
            results.append(result)
            mark = "✅" if result.success else f"❌ step {len(result.steps) - 1}: {result.error}"
            print(f"  Task {i:>3}/{len(records)}  {result.duration_ms:>5} ms  {mark}")
        browser.close()

    logger.flush()
    entries = {e["invoice_number"]: e for e in httpx.get(f"{base}/api/entries").json()}
    correct = [_matches(entries.get(r["invoice_number"]), r) for r in records]

    steps = [s for r in results for s in r.steps]
    report = {
        "run_id": run_id,
        "skill": skill.name,
        "layout": layout,
        "tasks": len(records),
        "task_success_rate": round(sum(correct) / len(records), 4),
        "agent_reported_success_rate": round(sum(r.success for r in results) / len(records), 4),
        "step_success_rate": round(sum(s.success for s in steps) / max(len(steps), 1), 4),
        "avg_task_ms": int(sum(r.duration_ms for r in results) / len(results)),
        "total_s": round(time.monotonic() - t0, 1),
        "retries": sum(s.attempts - 1 for s in steps),
        "strategies": dict(Counter(s.strategy or "failed" for s in steps)),
        "failures": dict(Counter(r.error for r in results if not r.success)),
        "mapping": mapping,
        "steps": [asdict(s) for s in steps],
    }
    if out_dir:
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{run_id}.json").write_text(json.dumps(report, indent=2))
    return report


def print_report(report: dict) -> None:
    print(f"\n=== {report['skill']} on layout {report['layout']} ===")
    print(f"Task success      : {report['task_success_rate']:.0%}  ({report['tasks']} tasks)")
    print(f"Step success      : {report['step_success_rate']:.0%}")
    print(f"Avg time per task : {report['avg_task_ms']} ms   (total {report['total_s']} s)")
    print(f"Retries           : {report['retries']}")
    print(f"Locator strategies: {report['strategies']}")
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
    args = parser.parse_args()

    from mimic.store import SkillStore

    skill = SkillStore().get(args.skill)
    if skill is None:
        raise SystemExit(f"Unknown skill '{args.skill}'. Run: python -m mimic.learn list")
    records = json.loads(args.data.read_text(encoding="utf-8"))[: args.limit]
    url = args.url or skill.start_url or get_settings().sandbox_url

    print(f"Running '{skill.name}' on {len(records)} tasks ({args.layout})...")
    report = run_batch(skill, records, url, args.layout, args.show, client=OllamaClient())
    print_report(report)
    print(f"\nReport saved to reports/{report['run_id']}.json")


if __name__ == "__main__":
    main()
