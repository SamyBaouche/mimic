"""Dashboard: WebApprover decisions and the FastAPI/WebSocket bridge (no browser, no LLM)."""

import threading
import time

from fastapi.testclient import TestClient

from mimic.approval import WebApprover
from mimic.events import EventBus
from mimic.server import RunManager, create_app


def _render(inputs):
    return [
        {
            "index": 0,
            "action": "fill",
            "parameter": "total",
            "value": inputs["total"],
            "element": _El("Total amount"),
        },
        {"index": 1, "action": "click", "parameter": None, "value": None, "element": _El("Save")},
    ]


class _El:
    def __init__(self, label):
        self.label, self.text, self.name, self.selector = label, "", "", "#x"


def _review_in_thread(approver, out, events=None):
    def go():
        out.append(approver.review(1, 3, {"total": "10.00"}, _render, title="inv-1"))

    t = threading.Thread(target=go)
    t.start()
    if events is not None:
        _wait_for(lambda: events)
    return t


def _wait_for(cond, timeout=5.0):
    """Poll instead of a fixed sleep: the first call may import slow modules."""
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def test_web_approver_approve_and_edit():
    events = []
    ap = WebApprover(events.append, poll_s=0.01)
    ap.aliases = {"totalamount": "total"}  # set by run_batch from the screen labels
    out = []
    t = _review_in_thread(ap, out, events)
    assert events[-1]["type"] == "review"
    assert events[-1]["plan"][0]["label"] == "Total amount"
    ap.decide({"action": "edit", "edits": {"Total amount": "99.50"}, "then": "approve"})
    t.join(2)
    decision, actions = out[0]
    assert decision.action == "approve"
    assert decision.edits == {"total": "99.50"}
    assert actions[0]["value"] == "99.50"
    assert ap.stats["edited"] == 1


def test_web_approver_unknown_field_keeps_waiting():
    events = []
    ap = WebApprover(events.append, poll_s=0.01)
    out = []
    t = _review_in_thread(ap, out, events)
    ap.decide({"action": "edit", "edits": {"nope": "1"}, "then": "approve"})
    _wait_for(lambda: len(events) == 2)
    assert not out and events[-1]["unknown_fields"] == ["nope"]
    ap.decide({"action": "skip"})
    t.join(2)
    assert out[0][0].action == "skip"


def test_web_approver_approve_all_then_auto():
    ap = WebApprover(lambda e: None, poll_s=0.01)
    out = []
    t = _review_in_thread(ap, out)
    time.sleep(0.05)
    ap.decide({"action": "approve_all"})
    t.join(2)
    decision, _ = ap.review(2, 3, {"total": "1"}, _render)
    assert decision.action == "approve" and ap.stats["approved"] == 2


def test_pause_resume_stop():
    ap = WebApprover(lambda e: None, poll_s=0.01)
    ap.decide({"action": "pause"})
    result = []
    t = threading.Thread(target=lambda: result.append(ap.checkpoint()))
    t.start()
    time.sleep(0.05)
    assert t.is_alive()
    ap.decide({"action": "resume"})
    t.join(1)
    assert result == [True]
    ap.decide({"action": "stop"})
    assert ap.checkpoint() is False
    assert ap.review(1, 1, {"total": "1"}, _render)[0].action == "quit"


def fake_runner(req, approver, emit):
    emit(
        {
            "type": "run_started",
            "run_id": "r",
            "skill": req.skill,
            "total": 1,
            "layout": req.layout,
            "url": "http://x",
        }
    )
    emit({"type": "frame", "data": "AAAA"})
    decision, _ = approver.review(1, 1, {"total": "5"}, _render, title="doc")
    emit({"type": "run_done", "report": {"decision": decision.action}})


def test_websocket_review_roundtrip():
    manager = RunManager(EventBus(), runner=fake_runner)
    client = TestClient(create_app(manager))
    assert "Mimic" in client.get("/").text
    with client.websocket_connect("/ws") as ws:
        assert client.post("/api/runs", json={"skill": "demo"}).status_code == 202
        seen = []
        while not seen or seen[-1]["type"] != "review":
            seen.append(ws.receive_json())
        assert {"run_started", "frame"} <= {e["type"] for e in seen}
        assert client.post("/api/runs", json={}).status_code == 409  # one run at a time
        ws.send_json({"action": "approve"})
        while (e := ws.receive_json())["type"] != "run_done":
            pass
        assert e["report"]["decision"] == "approve"
    manager.thread.join(2)
    assert client.get("/api/state").json() == {"running": False}


def test_runner_error_is_reported():
    def boom(req, approver, emit):
        raise ValueError("Unknown skill 'x'")

    manager = RunManager(EventBus(), runner=boom)
    client = TestClient(create_app(manager))
    client.post("/api/runs", json={"skill": "x"})
    manager.thread.join(2)
    errors = [e for e in manager.bus.history() if e["type"] == "error"]
    assert "Unknown skill" in errors[0]["message"]


def _item(error=None):
    from mimic.workflow import Item, PlannedStep

    actions = [
        {
            "index": 0,
            "action": "fill",
            "parameter": "total",
            "value": "3000",
            "element": _El("Total"),
        },
        {"index": 1, "action": "click", "parameter": None, "value": None, "element": _El("Save")},
    ]
    return Item(
        id="mail-1",
        title="Invoice INV-1 from Acme",
        context={},
        flags={"invoice.total": "confidence 0.40"},
        plan=[
            PlannedStep("enter_invoice", True, actions=actions),
            PlannedStep("request_approval", False, why="invoice.total > 2500 is false"),
        ],
        error=error,
    )


def test_workflow_review_publishes_grouped_plan_and_waits():
    from mimic.approval import WebWorkflowReview

    events = []
    ap = WebApprover(events.append, poll_s=0.01)
    review, out = WebWorkflowReview(ap), []
    t = threading.Thread(target=lambda: out.append(review("text", item=_item(), index=1, total=2)))
    t.start()
    while not any(e["type"] == "item_review" for e in events):
        time.sleep(0.01)
    event = next(e for e in events if e["type"] == "item_review")
    assert (event["index"], event["total"], event["title"]) == (1, 2, "Invoice INV-1 from Acme")
    first, second = event["steps"]
    assert first["runs"] and [r["index"] for r in first["plan"]] == [0, 1]
    assert first["plan"][0]["flag"] == "confidence 0.40"  # "invoice.total" -> field "total"
    assert not second["runs"] and second["plan"] == [] and "2500" in second["why"]
    ap.decide({"action": "edit", "values": {"total": "1"}})  # not offered for workflows: ignored
    ap.decide({"action": "skip"})
    t.join(2)
    assert out == ["skip"]


def test_workflow_review_approve_all_stops_asking():
    from mimic.approval import WebWorkflowReview

    events = []
    ap = WebApprover(events.append, poll_s=0.01)
    review, out = WebWorkflowReview(ap), []
    t = threading.Thread(target=lambda: out.append(review("t", item=_item(), index=1, total=3)))
    t.start()
    while not events:
        time.sleep(0.01)
    ap.decide({"action": "approve_all"})
    t.join(2)
    assert out == ["approve"]
    assert review("t", item=_item(), index=2, total=3) == "approve"  # no second question
    assert sum(e["type"] == "item_review" for e in events) == 1


def test_workflows_endpoint_lists_workflow_files(tmp_path, monkeypatch):
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "wf.yml").write_text(
        "name: emails\ndescription: Invoices by mail\nfor_each:\n  emails: inbox\n"
        "steps:\n  - read: attachment\n    fields_of: enter_invoice\n    as: invoice\n"
        "  - run: enter_invoice\n  - run: reply_to_supplier\n"
    )
    (tmp_path / "notes.yml").write_text("just: a config file\n")
    monkeypatch.chdir(tmp_path)
    client = TestClient(create_app(RunManager(EventBus(), runner=fake_runner)))
    listed = client.get("/api/workflows").json()
    assert listed == [
        {
            "path": str(__import__("pathlib").Path("examples") / "wf.yml"),
            "name": "emails",
            "description": "Invoices by mail",
            "skills": ["enter_invoice", "reply_to_supplier"],
        }
    ]


def test_a_workflow_request_goes_to_the_workflow_runner(monkeypatch):
    import mimic.server as server

    calls = []
    monkeypatch.setattr(server, "workflow_runner", lambda req, ap, emit: calls.append(req.workflow))
    server.default_runner(server.RunRequest(workflow="examples/wf.yml"), None, None)
    assert calls == ["examples/wf.yml"]
