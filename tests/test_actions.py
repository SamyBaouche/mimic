"""Checkbox, radio, file upload, Enter and a two-page flow: record once, replay with new data."""

import httpx
import pytest
from playwright.sync_api import sync_playwright

from mimic.agent import ReplayAgent, as_bool
from mimic.approval import format_plan, plan_rows
from mimic.recorder import Recorder, launch_browser
from mimic.skills import draft_skill


def _browser(p):
    try:
        return launch_browser(p, headless=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No Chromium available: {exc}")


def record(url: str, out, name: str, demo) -> Recorder:
    """Run `demo(page)` like a person would, with the recorder attached."""
    recorder = Recorder(name, url, out)
    with sync_playwright() as p:
        browser = _browser(p)
        context = browser.new_context()
        recorder.attach(context)
        page = context.new_page()
        page.goto(url)
        demo(page)
        page.wait_for_timeout(250)
        recorder.capture_outcome(page)
        browser.close()
    return recorder


@pytest.fixture(scope="module")
def receipts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("receipts")
    for name in ("taxi.pdf", "lunch.png", "license.pdf"):
        (folder / name).write_bytes(b"%PDF-1.4 demo receipt " + name.encode())
    return folder


@pytest.fixture(scope="module")
def expense_skill(sandbox_server, receipts, tmp_path_factory):
    def demo(page):
        page.fill("#employee", "Alice Martin")
        page.fill("#spent_on", "2026-11-03")
        page.fill("#amount", "42.50")
        page.check("input[name=kind][value=Meals]")  # Travel is the default
        page.check("#billable")
        page.set_input_files("#receipt", str(receipts / "taxi.pdf"))
        page.click("#next")  # page 2: review
        page.wait_for_url("**/expenses/review")
        page.click("#confirm")  # confirmation page
        page.wait_for_url("**/expenses/EXP-*")

    recorder = record(
        sandbox_server + "expenses/new", tmp_path_factory.mktemp("rec"), "file_expense", demo
    )
    return draft_skill(recorder.recording), recorder.recording


def test_demo_is_recorded_with_every_kind_of_action(expense_skill):
    skill, recording = expense_skill
    kinds = [s.action for s in skill.steps]
    assert kinds == ["fill", "fill", "fill", "choose", "check", "upload", "click", "click"]
    choose = skill.steps[3]
    assert choose.value == "Meals" and choose.element.options == [
        "Travel",
        "Meals",
        "Office",
        "Software",
    ]
    assert skill.steps[4].value == "true"
    assert skill.steps[5].value == "taxi.pdf"
    assert [s.navigates for s in skill.steps[-2:]] == [True, True]
    assert skill.parameters == ["employee", "spent_on", "amount", "kind", "billable", "receipt"]
    assert "Expense report EXP-" in " ".join(recording.outcome.texts)


@pytest.mark.parametrize(
    "task",
    [
        {"employee": "Bob Tremblay", "kind": "Travel", "billable": "no", "receipt": "lunch.png"},
        {"employee": "Chloe Roy", "kind": "software", "billable": "yes", "receipt": "license.pdf"},
    ],
)
def test_replay_with_other_options_files_and_pages(expense_skill, task, sandbox_server, receipts):
    skill, _ = expense_skill
    inputs = {**task, "spent_on": "2026-11-10", "amount": "18.25"}
    inputs["receipt"] = str(receipts / task["receipt"])
    actions = skill.render(inputs)

    plan = format_plan(1, 1, actions)
    assert task["receipt"] in plan and str(receipts) not in plan  # file name, not full path
    assert plan_rows(actions)[4]["value"] in ("yes", "no")

    httpx.delete(sandbox_server + "api/expenses")
    with sync_playwright() as p:
        browser = _browser(p)
        page = browser.new_page()
        page.goto(sandbox_server + "expenses/new")
        result = ReplayAgent(page).run(actions)
        final_text = page.inner_text("body")
        browser.close()

    assert result.success, result.error
    [saved] = httpx.get(sandbox_server + "api/expenses").json()
    assert saved["employee"] == task["employee"]
    assert saved["kind"].lower() == task["kind"].lower()
    assert saved["billable"] is as_bool(task["billable"])
    assert saved["receipt"] == task["receipt"] and saved["receipt_bytes"] > 0
    assert f"Expense report {saved['id']} submitted" in final_text


def test_missing_file_is_a_clear_failure(expense_skill, sandbox_server):
    skill, _ = expense_skill
    actions = skill.render(
        {
            "employee": "X",
            "spent_on": "2026-11-10",
            "amount": "1",
            "kind": "Meals",
            "billable": "no",
            "receipt": "nope/missing.pdf",
        }
    )
    with sync_playwright() as p:
        browser = _browser(p)
        page = browser.new_page()
        page.goto(sandbox_server + "expenses/new")
        result = ReplayAgent(page).run(actions)
        browser.close()
    assert not result.success
    assert "file not found" in result.error


def test_enter_in_a_search_box(sandbox_server, tmp_path):
    def demo(page):
        page.fill("#q", "Alice")
        page.press("#q", "Enter")
        page.wait_for_url("**/expenses?q=Alice")

    recorder = record(sandbox_server + "expenses", tmp_path, "find_expense", demo)
    skill = draft_skill(recorder.recording)
    assert [s.action for s in skill.steps] == ["fill", "press"]
    assert skill.steps[1].value == "Enter" and skill.steps[1].navigates

    with sync_playwright() as p:
        browser = _browser(p)
        page = browser.new_page()
        page.goto(sandbox_server + "expenses")
        result = ReplayAgent(page).run(skill.render({"q": "Bob"}))
        url = page.url
        browser.close()
    assert result.success, result.error
    assert url.endswith("/expenses?q=Bob")


# ---------------------------------------------------------------- success signal


def test_signal_learned_from_the_end_of_the_demo(expense_skill):
    skill, _ = expense_skill
    sig = skill.success
    assert sig.path == r"/expenses/[^/]*\d[^/]*"
    assert sig.example.startswith("Expense report EXP-0001 submitted")
    import re

    other = "Expense report EXP-0042 submitted for Bob Tremblay."
    assert re.search(sig.text, other, re.IGNORECASE)
    assert not re.search(sig.text, "Expense report EXP-0042 rejected.", re.IGNORECASE)


def test_learn_signal_rules():
    from mimic.models import Outcome
    from mimic.outcome import learn_signal

    # Same page, no message, no new query key: nothing to learn.
    assert learn_signal(Outcome(url="http://x/form", texts=["Form"]), "http://x/form", []) is None
    # Redirect with ?saved=1 counts, even without a message.
    sig = learn_signal(Outcome(url="http://x/?layout=v1&saved=1"), "http://x/?layout=v1", [])
    assert sig.query_keys == ["saved"] and sig.text is None
    # Django-style message: the demo value becomes a wildcard.
    sig = learn_signal(
        Outcome(
            url="http://x/admin/app/expense/", texts=["The expense “Alice” was added successfully."]
        ),
        "http://x/admin/app/expense/add/",
        ["Alice"],
    )
    import re

    assert re.search(sig.text, "The expense “Zoé Roy” was added successfully.", re.IGNORECASE)


def _records(receipts, n):
    kinds = ["Travel", "Meals", "Office", "Software"]
    return [
        {
            "employee": f"Employee {i}",
            "spent_on": f"2026-10-{i + 1:02d}",
            "amount": f"{10 + i}.50",
            "kind": kinds[i % 4],
            "billable": "yes" if i % 2 else "no",
            "receipt": str(receipts / ("taxi.pdf", "lunch.png", "license.pdf")[i % 3]),
        }
        for i in range(n)
    ]


def test_run_is_graded_with_the_confirmation(expense_skill, sandbox_server, receipts):
    from mimic.run import run_batch

    skill, _ = expense_skill
    httpx.delete(sandbox_server + "api/expenses")
    report = run_batch(
        skill,
        _records(receipts, 4),
        sandbox_server + "expenses/new",
        "v1",
        log_to_db=False,
        out_dir=None,
    )
    assert report["graded_by"] == "confirmation"
    assert report["task_success_rate"] == 1.0
    assert len(httpx.get(sandbox_server + "api/expenses").json()) == 4  # really saved


def test_a_task_that_stops_early_is_not_counted_as_done(expense_skill, sandbox_server, receipts):
    from mimic.run import run_batch

    skill, _ = expense_skill
    truncated = skill.model_copy(update={"steps": skill.steps[:-1]})  # never confirms
    report = run_batch(
        truncated,
        _records(receipts, 2),
        sandbox_server + "expenses/new",
        "v1",
        log_to_db=False,
        out_dir=None,
    )
    assert report["agent_reported_success_rate"] == 0.0
    assert report["task_success_rate"] == 0.0
    assert "not on the page the demo ended on" in next(iter(report["failures"]))
