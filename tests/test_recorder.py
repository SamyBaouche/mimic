"""End-to-end test: a scripted 'human' fills LedgerLite while the recorder listens."""

import json
import socket
import threading
import time

import pytest
import uvicorn

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.recorder import Recorder, launch_browser  # noqa: E402
from sandbox.app import ENTRIES, app  # noqa: E402

INVOICE = {
    "vendor": "Bureau Express",
    "invoice_number": "INV-20260001",
    "date": "2026-03-14",
    "total": "249.99",
    "category": "Software",
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def sandbox_url():
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


def _demo(page):
    """What a human does: click each field, type, pick a category, save."""
    for field in ["vendor", "invoice_number", "date", "total"]:
        page.click(f"#{field}")
        page.fill(f"#{field}", INVOICE[field])
    page.click("#category")
    page.select_option("#category", INVOICE["category"])
    page.click("#save-btn")
    page.wait_for_url("**saved=1")


def test_recorder_captures_full_demonstration(sandbox_url, tmp_path):
    ENTRIES.clear()
    recorder = Recorder("enter_invoice", sandbox_url, tmp_path)

    with sync_playwright() as p:
        try:
            browser = launch_browser(p, headless=True)
        except Exception as exc:  # no browser installed in this environment
            pytest.skip(f"No Chromium available: {exc}")
        context = browser.new_context()
        recorder.attach(context)
        page = context.new_page()
        page.goto(f"{sandbox_url}/?layout=v1")
        _demo(page)
        page.wait_for_timeout(300)
        browser.close()

    events = recorder.recording.events
    assert [e.index for e in events] == list(range(len(events)))
    shots = [e.screenshot for e in events if e.screenshot]
    assert len(shots) == len(set(shots))

    fills = {e.element.id: e.value for e in events if e.kind == "fill"}
    assert fills == {k: v for k, v in INVOICE.items() if k != "category"}

    selects = [e for e in events if e.kind == "select"]
    assert selects[-1].value == "Software"
    assert selects[-1].element.label == "Category"

    clicks = [e for e in events if e.kind == "click"]
    assert clicks[-1].element.id == "save-btn"
    assert clicks[-1].element.text == "Save invoice"

    # Labels, bounding boxes and screenshots are captured for later self-healing.
    vendor = next(e for e in events if e.element.id == "vendor")
    assert vendor.element.label == "Vendor"
    assert vendor.element.bbox.width > 0
    assert (tmp_path / vendor.screenshot).exists()

    # The demonstration really happened in the app.
    assert ENTRIES[-1]["vendor"] == "Bureau Express"

    saved = json.loads(recorder.save().read_text())
    assert saved["name"] == "enter_invoice"
    assert len(saved["events"]) == len(events)
