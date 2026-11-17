"""Recorder: capture a single human demonstration of a workflow.

Usage:
    uv run python -m mimic.recorder --name enter_invoice --url http://localhost:8001

A browser opens. Do the task once, then come back to the terminal and press Enter.
The demonstration is saved to recordings/<name>/recording.json with one screenshot per step.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

from mimic.models import ElementInfo, RecordedEvent, Recording

# Injected into every page (survives navigations). Reports user actions to Python.
CAPTURE_JS = r"""
(() => {
  if (window.__mimicInstalled) return;
  window.__mimicInstalled = true;

  const cssPath = (el) => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    const parts = [];
    while (el && el.nodeType === 1 && el !== document.body) {
      let part = el.tagName.toLowerCase();
      const parent = el.parentElement;
      if (parent) {
        const same = [...parent.children].filter(c => c.tagName === el.tagName);
        if (same.length > 1) part += `:nth-of-type(${same.indexOf(el) + 1})`;
      }
      parts.unshift(part);
      el = parent;
    }
    return parts.length ? 'body > ' + parts.join(' > ') : 'body';
  };

  const labelOf = (el) => {
    if (el.id) {
      const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
      if (l) return l.innerText.trim();
    }
    const wrap = el.closest('label');
    if (wrap) return wrap.innerText.trim();
    return el.getAttribute('aria-label') || null;
  };

  const describe = (el) => {
    const r = el.getBoundingClientRect();
    return {
      tag: el.tagName.toLowerCase(),
      selector: cssPath(el),
      id: el.id || null,
      name: el.getAttribute('name'),
      type: el.getAttribute('type'),
      label: labelOf(el),
      text: (el.innerText || '').trim().slice(0, 80) || null,
      placeholder: el.getAttribute('placeholder'),
      bbox: { x: r.x, y: r.y, width: r.width, height: r.height },
    };
  };

  const isField = (el) =>
    (el.tagName === 'INPUT' && !['submit', 'button', 'hidden'].includes(el.type)) ||
    el.tagName === 'TEXTAREA';

  // Clicks on buttons/links (fields are captured through "change" instead).
  document.addEventListener('click', (e) => {
    const el = e.target.closest('button, a, input[type=submit], input[type=button], [role=button]');
    if (!el) return;
    window.__mimicReport({ kind: 'click', element: describe(el), value: null, url: location.href });
  }, true);

  // Final value of a field, fired when the user leaves it.
  document.addEventListener('change', (e) => {
    const el = e.target;
    const kind = el.tagName === 'SELECT' ? 'select' : isField(el) ? 'fill' : null;
    if (!kind) return;
    window.__mimicReport({ kind, element: describe(el), value: el.value, url: location.href });
  }, true);
})();
"""


class Recorder:
    """Attaches to a browser context and turns user actions into a Recording."""

    def __init__(self, name: str, start_url: str, out_dir: Path, screenshots: bool = True):
        self.recording = Recording(name=name, start_url=start_url)
        self.out_dir = out_dir
        self.screenshots = screenshots
        self._t0 = time.monotonic()

    def attach(self, context: BrowserContext) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        context.expose_binding("__mimicReport", self._on_event)
        context.add_init_script(CAPTURE_JS)

    def _on_event(self, source: dict, payload: dict) -> None:
        # Register the event first: taking a screenshot yields control, and the
        # next event can arrive meanwhile. Reserving the index keeps them unique.
        index = len(self.recording.events)
        event = RecordedEvent(
            index=index,
            kind=payload["kind"],
            element=ElementInfo(**payload["element"]),
            value=payload.get("value"),
            url=payload["url"],
            timestamp_ms=int((time.monotonic() - self._t0) * 1000),
        )
        self.recording.events.append(event)

        el = payload["element"]
        target = el.get("label") or el.get("text") or el["selector"]
        value = f" = {payload['value']!r}" if payload.get("value") is not None else ""
        print(f"  [{index:02d}] {payload['kind']:<6} {target}{value}")

        if self.screenshots:
            page: Page = source["page"]
            shot = f"step_{index:03d}.png"
            try:
                page.screenshot(path=str(self.out_dir / shot))
                event.screenshot = shot
            except Exception:  # page may be navigating after a submit click
                pass

    def save(self) -> Path:
        path = self.out_dir / "recording.json"
        path.write_text(self.recording.model_dump_json(indent=2))
        return path


def launch_browser(p: Playwright, headless: bool) -> Browser:
    """Use Playwright's Chromium, or a local Chrome if MIMIC_CHROME_PATH is set."""
    chrome = os.environ.get("MIMIC_CHROME_PATH")
    return p.chromium.launch(headless=headless, executable_path=chrome or None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a single demonstration.")
    parser.add_argument("--name", required=True, help="Skill name, e.g. enter_invoice")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--out", default="recordings")
    args = parser.parse_args()

    out_dir = Path(args.out) / args.name
    recorder = Recorder(args.name, args.url, out_dir)

    with sync_playwright() as p:
        browser = launch_browser(p, headless=False)
        context = browser.new_context(viewport=recorder.recording.viewport)
        recorder.attach(context)
        page = context.new_page()
        page.goto(args.url)

        print(f"\nRecording '{args.name}'. Do the task once in the browser.")
        print("When finished, come back here and press Enter.\n")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass
        page.wait_for_timeout(300)  # flush last events
        browser.close()

    path = recorder.save()
    print(f"\nSaved {len(recorder.recording.events)} steps to {path}")


if __name__ == "__main__":
    main()
