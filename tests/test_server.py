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


def _review_in_thread(approver, out):
    def go():
        out.append(approver.review(1, 3, {"total": "10.00"}, _render, title="inv-1"))

    t = threading.Thread(target=go)
    t.start()
    return t


def test_web_approver_approve_and_edit():
    events = []
    ap = WebApprover(events.append, poll_s=0.01)
    ap.aliases = {"totalamount": "total"}  # set by run_batch from the screen labels
    out = []
    t = _review_in_thread(ap, out)
    time.sleep(0.05)
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
    t = _review_in_thread(ap, out)
    time.sleep(0.05)
    ap.decide({"action": "edit", "edits": {"nope": "1"}, "then": "approve"})
    time.sleep(0.05)
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
