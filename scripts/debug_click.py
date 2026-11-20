"""Diagnose why the Save click fails in a visible browser covered by another window.

    uv run python scripts/debug_click.py

Reproduces a Mimic run: visible browser, 1 s click timeout like the agent, and the browser
COVERED by your terminal while it clicks. Prints the full Playwright log on failure.
"""

import sys
import time

from playwright.sync_api import sync_playwright

sys.path.insert(0, "src")
from mimic.recorder import NO_THROTTLING, launch_browser  # noqa: E402

URL = "http://localhost:8001/?layout=v1"


def fill(page) -> None:
    page.fill("#vendor", "Debug Vendor")
    page.fill("#invoice_number", f"DEBUG-{time.time_ns()}")
    page.fill("#date", "2026-01-01")
    page.fill("#total", "1")
    page.select_option("#category", "Software")


def click(page, label: str) -> None:
    visible = page.evaluate("document.visibilityState")
    try:
        with page.expect_navigation(timeout=5000):
            page.locator("#save-btn").click(timeout=1000)  # same timeout as the agent
        print(f"  {label}: click OK  (page visibility: {visible})")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label}: click FAILED  (page visibility: {visible})")
        print("    " + str(exc)[:900].replace("\n", "\n    "))


with sync_playwright() as p:
    print("Browser flags:", " ".join(NO_THROTTLING))
    browser = launch_browser(p, headless=False)
    page = browser.new_page(viewport={"width": 1280, "height": 800})

    page.goto(URL)
    fill(page)
    click(page, "1. browser in front")

    page.goto(URL)
    fill(page)
    print("\n>>> NOW click on this PowerShell window so it COVERS the browser.")
    for s in range(8, 0, -1):
        print(f"    clicking in {s} s...", end="\r")
        time.sleep(1)
    print()
    click(page, "2. browser covered ")
    browser.close()
