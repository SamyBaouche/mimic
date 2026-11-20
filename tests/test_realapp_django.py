"""A real app Mimic was not built for: a stock Django admin (sign-in, date, select, radio,
checkbox, file upload, success message). Record once, replay rows from Excel, check the DB."""

import os
import socket
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from openpyxl import Workbook
from playwright.sync_api import sync_playwright

from mimic.recorder import Recorder, launch_browser
from mimic.run import run_batch
from mimic.skills import draft_skill
from mimic.tables import load_records

django = pytest.importorskip("django")

TESTS = Path(__file__).parent
PASSWORD = "ledger-demo-2026"
ADD = "admin/expenses/expense/add/"


def _manage(env, *args):
    subprocess.run(
        [sys.executable, "-m", "django", *args], env=env, check=True, capture_output=True
    )


@pytest.fixture(scope="module")
def django_admin(tmp_path_factory):
    data = tmp_path_factory.mktemp("django")
    env = {
        **os.environ,
        "PYTHONPATH": str(TESTS),
        "DJANGO_SETTINGS_MODULE": "realapps.expenses_site.settings",
        "REALAPP_DATA": str(data),
        "DJANGO_SUPERUSER_PASSWORD": PASSWORD,
    }
    _manage(env, "migrate", "--run-syncdb", "-v", "0")
    _manage(env, "createsuperuser", "--noinput", "--username", "demo", "--email", "d@x.io")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "django", "runserver", f"127.0.0.1:{port}", "--noreload"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/"
    for _ in range(100):
        try:
            httpx.get(base + "admin/login/", timeout=1)
            break
        except httpx.HTTPError:
            time.sleep(0.1)
    yield base, data
    server.terminate()


@pytest.fixture(scope="module")
def receipts(tmp_path_factory):
    folder = tmp_path_factory.mktemp("receipts")
    for name in ("taxi.pdf", "hotel.pdf", "lunch.png"):
        (folder / name).write_bytes(b"receipt " + name.encode())
    return folder


@pytest.fixture(scope="module")
def skill(django_admin, receipts, tmp_path_factory):
    base, _ = django_admin
    recorder = Recorder("django_expense", base + ADD, tmp_path_factory.mktemp("rec"))
    with sync_playwright() as p:
        try:
            browser = launch_browser(p, headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")
        context = browser.new_context()
        recorder.attach(context)
        page = context.new_page()
        page.goto(base + ADD)  # not signed in: Django redirects to its login page
        page.fill("#id_username", "demo")
        page.fill("#id_password", PASSWORD)
        page.click("input[type=submit]")
        page.wait_for_url("**/" + ADD)
        page.fill("#id_title", "Taxi to the airport")
        page.fill("#id_spent_on", "2026-11-03")
        page.fill("#id_amount", "42.50")
        page.select_option("#id_category", label="Travel")
        page.check("text=Bank transfer")
        page.check("#id_reimbursable")
        page.set_input_files("#id_receipt", str(receipts / "taxi.pdf"))
        page.click("input[name=_save]")
        page.wait_for_url("**/admin/expenses/expense/")
        recorder.capture_outcome(page)
        browser.close()
    return draft_skill(recorder.recording)


def test_demo_learns_secret_login_and_a_success_message(skill):
    assert skill.secrets == ["password"]
    actions = [s.action for s in skill.steps]
    assert actions[:3] == ["fill", "fill", "click"]  # sign in
    # Fields can be recorded in another order (a number is committed when you leave it).
    assert sorted(actions[3:-1]) == sorted(
        ["fill", "fill", "fill", "select", "choose", "check", "upload"]
    )
    assert actions[-1] == "click"
    assert skill.steps[2].element.text == "Log in" and skill.steps[-1].element.text == "Save"
    assert skill.steps[[s.action for s in skill.steps].index("upload")].element.label == "Receipt:"
    assert skill.success.path == "/admin/expenses/expense/"
    assert "was added successfully" in skill.success.example


def test_replays_rows_from_excel_into_the_real_app(
    skill, django_admin, receipts, tmp_path, monkeypatch
):
    base, data = django_admin
    rows = [
        ["Title", "Date", "Amount", "Category", "Payment", "Reimbursable", "Receipt"],
        [
            "Hotel Montclair",
            datetime(2026, 10, 12),
            389.9,
            "Travel",
            "Company card",
            "yes",
            str(receipts / "hotel.pdf"),
        ],
        [
            "Team lunch",
            datetime(2026, 10, 14),
            86.4,
            "Meals",
            "Cash",
            "no",
            str(receipts / "lunch.png"),
        ],
        [
            "Figma licence",
            datetime(2026, 10, 20),
            180,
            "Software",
            "Bank transfer",
            "no",
            str(receipts / "taxi.pdf"),
        ],
        [
            "Printer paper",
            datetime(2026, 10, 21),
            23.75,
            "Office",
            "Company card",
            "yes",
            str(receipts / "hotel.pdf"),
        ],
    ]
    wb = Workbook()
    for r in rows:
        wb.active.append(r)
    wb.save(tmp_path / "expenses.xlsx")
    records = load_records(tmp_path / "expenses.xlsx")

    monkeypatch.setenv("MIMIC_SECRET_PASSWORD", PASSWORD)
    report = run_batch(skill, records, base + ADD, "v1", log_to_db=False, out_dir=None)

    assert report["graded_by"] == "confirmation"
    assert report["task_success_rate"] == 1.0, report["failures"]
    assert report["strategies"]["not needed"] == 3 * 3  # signed in once, then skipped login

    with sqlite3.connect(data / "db.sqlite3") as db:
        saved = db.execute(
            "SELECT title, spent_on, amount, category, payment, reimbursable, receipt "
            "FROM expenses_expense ORDER BY id"
        ).fetchall()
    saved = saved[1:]  # the first one is the demonstration itself
    assert [(t, d, float(a), c, p, bool(r)) for t, d, a, c, p, r, _ in saved] == [
        ("Hotel Montclair", "2026-10-12", 389.9, "travel", "card", True),
        ("Team lunch", "2026-10-14", 86.4, "meals", "cash", False),
        ("Figma licence", "2026-10-20", 180.0, "software", "transfer", False),
        ("Printer paper", "2026-10-21", 23.75, "office", "card", True),
    ]
    assert all(row[6].startswith("receipts/") for row in saved)
