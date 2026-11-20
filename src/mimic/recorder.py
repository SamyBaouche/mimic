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

from mimic.models import ElementInfo, Outcome, RecordedEvent, Recording

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
    if (el.labels && el.labels.length) return el.labels[0].innerText.trim() || null;
    const by = el.getAttribute('aria-labelledby');
    if (by) {
      const t = by.split(/\s+/).map(id => document.getElementById(id)?.innerText || '').join(' ');
      if (t.trim()) return t.trim();
    }
    if (el.getAttribute('aria-label')) return el.getAttribute('aria-label');
    // A field alone in a fieldset is named by its legend.
    const fs = el.closest('fieldset');
    if (fs && fs.querySelector('legend') &&
        fs.querySelectorAll('input:not([type=hidden]), select, textarea').length === 1) {
      return fs.querySelector('legend').innerText.trim() || null;
    }
    // A label right next to the field but not tied to it (common in admin screens).
    let box = el.parentElement;
    for (let depth = 0; box && box !== document.body && depth < 3; depth++) {
      const loose = [...box.querySelectorAll('label')].filter(
        l => !l.htmlFor && !l.querySelector('input, select, textarea'));
      if (loose.length === 1 && box.querySelectorAll('input, select, textarea').length <= 2) {
        return loose[0].innerText.trim() || null;
      }
      box = box.parentElement;
    }
    return null;
  };
  // Buttons written as <input type=submit value="Save"> have no inner text: use the value.
  const textOf = (el) => (el.innerText || '').trim().slice(0, 80) ||
    (el.tagName === 'INPUT' && ['submit', 'button', 'reset'].includes(el.type) ? el.value : '') ||
    null;

  const describe = (el) => {
    const r = el.getBoundingClientRect();
    return {
      tag: el.tagName.toLowerCase(),
      selector: cssPath(el),
      id: el.id || null,
      name: el.getAttribute('name'),
      type: el.getAttribute('type'),
      label: labelOf(el),
      text: textOf(el),
      placeholder: el.getAttribute('placeholder'),
      options: el.tagName === 'SELECT' ? [...el.options].map(o => o.value || o.text) : [],
      bbox: { x: r.x, y: r.y, width: r.width, height: r.height },
    };
  };

  // Passwords, one-time codes, card numbers: the value never leaves the page.
  const SECRET_NAME =
    /password|passwd|passcode|pwd|otp|one.?time|cvv|cvc|card.?num|ssn|secret|token|(^|[^a-z])pin([^a-z]|$)/i;
  const isSecret = (el) =>
    el.type === 'password' ||
    /current-password|new-password|one-time-code|cc-number|cc-csc/.test(el.autocomplete || '') ||
    SECRET_NAME.test([el.name, el.id, el.getAttribute('aria-label')].join(' '));

  const isField = (el) =>
    (el.tagName === 'INPUT' && !['submit', 'button', 'hidden'].includes(el.type)) ||
    el.tagName === 'TEXTAREA';

  // Clicks on buttons/links (fields are captured through "change" instead).
  document.addEventListener('click', (e) => {
    const el = e.target.closest('button, a, input[type=submit], input[type=button], [role=button]');
    if (!el) return;
    window.__mimicReport({ kind: 'click', element: describe(el), value: null, url: location.href });
  }, true);

  const kindOf = (el) => {
    if (el.tagName === 'SELECT') return 'select';
    if (el.type === 'checkbox') return 'check';
    if (el.type === 'radio') return 'choose';
    if (el.type === 'file') return 'upload';
    return isField(el) ? 'fill' : null;
  };

  const valueOf = (el, kind) => {
    if (kind === 'check') return String(el.checked);
    if (kind === 'upload') return el.files && el.files.length ? el.files[0].name : '';
    return el.value;
  };

  const report = (el, kind) => {
    const secret = kind === 'fill' && isSecret(el);
    const value = valueOf(el, kind);
    el.__mimicLast = value;
    const element = describe(el);
    if (kind === 'choose') {
      // A radio group: remember every option, so another task can pick another one.
      const group = el.form && el.name
        ? [...el.form.querySelectorAll(`input[type=radio][name="${CSS.escape(el.name)}"]`)]
        : [el];
      element.options = group.map(r => r.value);
    }
    window.__mimicReport({
      kind, element, value: secret ? null : value, secret, url: location.href,
    });
  };

  // Final value of a field, fired when the user leaves it (or ticks / picks / attaches).
  document.addEventListener('change', (e) => {
    const el = e.target;
    const kind = kindOf(el);
    if (!kind) return;
    if (kind === 'fill' && el.__mimicLast === el.value) return;  // already sent on Enter
    report(el, kind);
  }, true);

  // Enter in a text field (search boxes, quick-add forms): the value first, then the key.
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.isComposing) return;
    const el = e.target;
    if (kindOf(el) !== 'fill' || el.tagName === 'TEXTAREA') return;
    if (el.__mimicLast !== el.value) report(el, 'fill');
    window.__mimicReport({
      kind: 'press', element: describe(el), value: 'Enter', secret: false, url: location.href,
    });
  }, true);
})();
"""


# Confirmation-like messages visible at the end of the demo (alerts, toasts, headings).
OUTCOME_JS = r"""
() => {
  const sel = '[role=alert], [role=status], .alert, .toast, .notice, .success, .messagelist li, ' +
              '.message, .notification, .flash, h1, h2';
  const seen = new Set();
  return [...document.querySelectorAll(sel)]
    .filter(el => el.offsetParent !== null || el.getClientRects().length)
    .map(el => (el.innerText || '').trim().replace(/\s+/g, ' '))
    .filter(t => t && t.length <= 200 && !seen.has(t) && seen.add(t))
    .slice(0, 6);
}
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
            value=None if payload.get("secret") else payload.get("value"),
            secret=bool(payload.get("secret")),
            url=payload["url"],
            timestamp_ms=int((time.monotonic() - self._t0) * 1000),
        )
        self.recording.events.append(event)

        el = payload["element"]
        target = el.get("label") or el.get("text") or el["selector"]
        if payload.get("secret"):
            value = " = (secret, not recorded)"
        else:
            value = f" = {payload['value']!r}" if payload.get("value") is not None else ""
        print(f"  [{index:02d}] {payload['kind']:<6} {target}{value}")

        if self.screenshots and not event.secret:
            page: Page = source["page"]
            shot = f"step_{index:03d}.png"
            try:
                page.screenshot(path=str(self.out_dir / shot))
                event.screenshot = shot
            except Exception:  # page may be navigating after a submit click
                pass

    def capture_outcome(self, page: Page) -> None:
        """Remember where the demo ended: the success signal for every replay."""
        try:
            texts = page.evaluate(OUTCOME_JS)
            self.recording.outcome = Outcome(url=page.url, texts=texts)
        except Exception:  # noqa: BLE001 - the page may already be closed
            pass

    def save(self) -> Path:
        path = self.out_dir / "recording.json"
        path.write_text(self.recording.model_dump_json(indent=2))
        return path


# A visible browser that goes behind another window (e.g. while you answer in the terminal)
# is throttled by the OS: animation frames stop, and Playwright's "is the element stable?"
# check before a click never completes. These flags keep the page running at full speed.
NO_THROTTLING = [
    "--disable-renderer-backgrounding",
    "--disable-backgrounding-occluded-windows",
    "--disable-background-timer-throttling",
    # Windows: Chrome detects natively when another window covers it and stops rendering.
    "--disable-features=CalculateNativeWinOcclusion",
]


def launch_browser(p: Playwright, headless: bool) -> Browser:
    """Use Playwright's Chromium, or a local Chrome if MIMIC_CHROME_PATH is set."""
    chrome = os.environ.get("MIMIC_CHROME_PATH")
    return p.chromium.launch(headless=headless, executable_path=chrome or None, args=NO_THROTTLING)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record a single demonstration.")
    parser.add_argument("--name", required=True, help="Skill name, e.g. enter_invoice")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--out", default="recordings")
    args = parser.parse_args(argv)

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
        pages = [pg for pg in context.pages if not pg.is_closed()]
        if pages:
            recorder.capture_outcome(pages[-1])
        browser.close()

    path = recorder.save()
    print(f"\nSaved {len(recorder.recording.events)} steps to {path}")


if __name__ == "__main__":
    main()
