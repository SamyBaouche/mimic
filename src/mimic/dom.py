"""DOM helpers used by the healer: list candidate elements, and resolve a screen point."""

from __future__ import annotations

from playwright.sync_api import Page

from mimic.models import ElementInfo

KIND_SELECTORS = {
    "click": "button, a[href], input[type=submit], input[type=button], [role=button]",
    "fill": (
        "input:not([type=hidden]):not([type=submit]):not([type=button])"
        ":not([type=checkbox]):not([type=radio]), textarea"
    ),
    "select": "select",
    "check": "input[type=checkbox], [role=checkbox], [role=switch]",
    "choose": "input[type=radio], [role=radio]",
    "upload": "input[type=file]",
    "press": (
        "input:not([type=hidden]):not([type=submit]):not([type=button])"
        ":not([type=checkbox]):not([type=radio]), textarea"
    ),
}

_DESCRIBE = r"""
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
  return 'body > ' + parts.join(' > ');
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
    tag: el.tagName.toLowerCase(), selector: cssPath(el), id: el.id || null,
    name: el.getAttribute('name'), type: el.getAttribute('type'), label: labelOf(el),
    text: textOf(el),
    placeholder: el.getAttribute('placeholder'),
    options: el.tagName === 'SELECT' ? [...el.options].map(o => o.value || o.text) : [],
    bbox: { x: r.x, y: r.y, width: r.width, height: r.height },
  };
};
const visible = (el) => {
  const r = el.getBoundingClientRect();
  const s = getComputedStyle(el);
  return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
};
"""

CANDIDATES_JS = (
    "(sel) => {"
    + _DESCRIBE
    + "return [...document.querySelectorAll(sel)].filter(visible).map(describe); }"
)

AT_POINT_JS = (
    "([x, y, sel]) => {"
    + _DESCRIBE
    + """
  let el = document.elementFromPoint(x, y);
  if (!el) return null;
  if (el.tagName === 'LABEL' && el.control) el = el.control;
  const hit = el.closest(sel);
  if (hit && visible(hit)) return describe(hit);
  // Nearest matching element within 60 px of the point.
  let best = null, bestD = 60;
  for (const c of document.querySelectorAll(sel)) {
    if (!visible(c)) continue;
    const r = c.getBoundingClientRect();
    const dx = Math.max(r.left - x, 0, x - r.right), dy = Math.max(r.top - y, 0, y - r.bottom);
    const d = Math.hypot(dx, dy);
    if (d < bestD) { best = c; bestD = d; }
  }
  return best ? describe(best) : null;
}"""
)


def list_candidates(page: Page, action: str) -> list[ElementInfo]:
    return [ElementInfo(**d) for d in page.evaluate(CANDIDATES_JS, KIND_SELECTORS[action])]


def element_at(page: Page, x: float, y: float, action: str) -> ElementInfo | None:
    found = page.evaluate(AT_POINT_JS, [x, y, KIND_SELECTORS[action]])
    return ElementInfo(**found) if found else None


def describe_text(el: ElementInfo, action: str) -> str:
    """Short natural-language description used for embeddings and prompts."""
    kind = {
        "click": "button",
        "fill": "input field",
        "select": "dropdown",
        "check": "checkbox",
        "choose": "radio option",
        "upload": "file upload",
        "press": "input field",
    }[action]
    name = el.label or el.text or el.placeholder or el.name or el.id or "unnamed"
    extra = f" ({el.type})" if el.type and el.type not in ("text", "submit") else ""
    return f"{kind} '{name}'{extra}"
