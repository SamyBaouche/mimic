"""End-to-end: the agent replays a learned skill on the sandbox and is graded by the app."""

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

pytest.importorskip("playwright")
pytest.importorskip("langgraph")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.agent import ReplayAgent, _same_value  # noqa: E402
from mimic.models import Recording  # noqa: E402
from mimic.recorder import launch_browser  # noqa: E402
from mimic.run import run_batch  # noqa: E402
from mimic.sandbox.app import app  # noqa: E402
from mimic.sandbox.generate_invoices import generate  # noqa: E402
from mimic.skills import apply_refinement, draft_skill  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"


@pytest.fixture(scope="module")
def url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True


@pytest.fixture(scope="module")
def browser_available():
    with sync_playwright() as p:
        try:
            launch_browser(p, headless=True).close()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")


@pytest.fixture
def skill():
    """Skill with LLM-style parameter names, so the batch also exercises input mapping."""
    rec = Recording.model_validate_json(FIXTURE.read_text())
    names = ["vendor_name", "invoice_number", "invoice_date", "total_amount", "category"]
    return apply_refinement(
        draft_skill(rec),
        {
            "description": "Enters a supplier invoice.",
            "steps": [{"index": i, "parameter": n} for i, n in enumerate(names)],
        },
    )


def test_replay_on_original_layout_is_perfect(url, skill, browser_available):
    report = run_batch(skill, generate(5), url, "v1", log_to_db=False, out_dir=None)
    assert report["task_success_rate"] == 1.0
    assert report["mapping"]["total_amount"] == "total"
    assert report["strategies"] == {"selector": 30}


def test_redesign_breaks_the_button_but_fields_are_found_by_name(url, skill, browser_available):
    """Baseline for Sprint 5: without vision, the renamed and moved button cannot be found."""
    report = run_batch(skill, generate(2), url, "v2", log_to_db=False, out_dir=None)
    assert report["task_success_rate"] == 0.0
    assert report["strategies"] == {"name": 10, "failed": 2}
    failed = [s for s in report["steps"] if not s["success"]]
    assert all(s["action"] == "click" for s in failed)


def test_agent_reports_each_step(url, skill, browser_available):
    seen = []
    with sync_playwright() as p:
        browser = launch_browser(p, headless=True)
        page = browser.new_page()
        page.goto(url)
        actions = skill.render(
            {
                "vendor_name": "AquaNet",
                "invoice_number": "INV-1",
                "invoice_date": "2026-02-02",
                "total_amount": 10,
                "category": "Utilities",
            }
        )
        result = ReplayAgent(page, on_step=seen.append).run(actions)
        browser.close()
    assert result.success
    assert [s.index for s in seen] == list(range(6))
    assert all(s.latency_ms >= 0 for s in seen)


@pytest.mark.parametrize(
    "actual, expected, same",
    [("10", "10", True), ("10.50", "10.5", True), ("10", "11", False), ("a", "b", False)],
)
def test_value_comparison(actual, expected, same):
    assert _same_value(actual, expected) is same
