"""Human-in-the-loop: a scripted 'human' approves, edits, skips and declines submits."""

from pathlib import Path

import httpx
import pytest

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.approval import ConsoleApprover, format_plan  # noqa: E402
from mimic.models import Recording  # noqa: E402
from mimic.recorder import launch_browser  # noqa: E402
from mimic.run import run_batch  # noqa: E402
from mimic.sandbox.generate_invoices import generate  # noqa: E402
from mimic.skills import draft_skill  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"


@pytest.fixture(scope="module")
def browser_ok():
    with sync_playwright() as p:
        try:
            launch_browser(p, headless=True).close()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")


@pytest.fixture
def skill():
    return draft_skill(Recording.model_validate_json(FIXTURE.read_text()))


@pytest.fixture
def renamed_skill():
    from mimic.skills import apply_refinement

    names = ["vendor_name", "invoice_number", "invoice_date", "total_amount", "category"]
    return apply_refinement(
        draft_skill(Recording.model_validate_json(FIXTURE.read_text())),
        {
            "description": "Enters a supplier invoice.",
            "steps": [{"index": i, "parameter": n} for i, n in enumerate(names)],
        },
    )


def human(*answers):
    """Returns scripted answers in order, like a person typing at the prompt."""
    queue = list(answers)
    asked = []

    def _input(prompt):
        asked.append(prompt)
        return queue.pop(0)

    _input.asked = asked
    return _input


def entries(url):
    return httpx.get(f"{url}api/entries").json()


def _run(skill, url, approver, n):
    return run_batch(
        skill, generate(n), url, "v1", log_to_db=False, out_dir=None, approver=approver
    )


def test_plan_shows_values_and_final_click(skill):
    inv = generate(1)[0]
    plan = format_plan(1, 3, skill.render(inv), "Bureau Express")
    assert "Task 1/3" in plan
    assert f"Vendor             <- {inv['vendor']}" in plan
    assert "then click   'Save invoice'" in plan


def test_approve_skip_then_approve_all(skill, sandbox_server, browser_ok):
    ask = human("a", "s", "t")  # approve #1, skip #2, approve all the rest
    approver = ConsoleApprover(input_fn=ask, print_fn=lambda *_: None)
    report = _run(skill, sandbox_server, approver, n=5)
    assert len(ask.asked) == 3  # never asked again after "approve all"
    assert report["executed"] == 4 and report["skipped_by_user"] == 1
    assert report["task_success_rate"] == 1.0
    skipped_invoice = generate(5)[1]["invoice_number"]
    assert skipped_invoice not in {e["invoice_number"] for e in entries(sandbox_server)}


def test_edit_changes_what_is_typed(skill, sandbox_server, browser_ok):
    ask = human("e", "total=999.99", "", "a")
    approver = ConsoleApprover(input_fn=ask, print_fn=lambda *_: None)
    report = _run(skill, sandbox_server, approver, n=1)
    assert report["task_success_rate"] == 1.0  # graded against the edited value
    assert entries(sandbox_server)[0]["total"] == 999.99
    assert approver.stats["edited"] == 1


@pytest.mark.parametrize("name", ["Total amount", "total", "TOTAL_AMOUNT", "total amount"])
def test_edit_accepts_screen_label_data_field_or_parameter(
    renamed_skill, sandbox_server, browser_ok, name
):
    """Regression: the LLM renamed 'total' to 'total_amount'; users type what they see."""
    ask = human("e", f"{name}=123.45", "", "a")
    approver = ConsoleApprover(input_fn=ask, print_fn=lambda *_: None)
    report = _run(renamed_skill, sandbox_server, approver, n=1)
    assert report["task_success_rate"] == 1.0
    assert entries(sandbox_server)[0]["total"] == 123.45


def test_unknown_field_is_refused(skill, sandbox_server, browser_ok):
    printed = []
    ask = human("e", "colour=red", "", "q")
    approver = ConsoleApprover(input_fn=ask, print_fn=printed.append)
    report = _run(skill, sandbox_server, approver, n=1)
    assert any("unknown field 'colour'" in p for p in printed)
    assert report["executed"] == 0


def test_quit_stops_the_batch(skill, sandbox_server, browser_ok):
    approver = ConsoleApprover(input_fn=human("a", "q"), print_fn=lambda *_: None)
    report = _run(skill, sandbox_server, approver, n=5)
    assert report["executed"] == 1
    assert len(entries(sandbox_server)) == 1


def test_declined_submit_fills_but_never_submits(skill, sandbox_server, browser_ok):
    ask = human("t", "n", "y")  # approve all plans; decline 1st submit, accept 2nd
    approver = ConsoleApprover(input_fn=ask, print_fn=lambda *_: None, confirm_submit=True)
    report = _run(skill, sandbox_server, approver, n=2)
    assert report["skipped_by_user"] == 1 and report["executed"] == 1
    assert report["task_success_rate"] == 1.0
    assert [e["invoice_number"] for e in entries(sandbox_server)] == [
        generate(2)[1]["invoice_number"]
    ]
    assert approver.stats["declined_submit"] == 1
