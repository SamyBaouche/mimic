"""Failure risk: features observed just before acting, the XGBoost model, and mitigations.

Before each action the agent looks at the element it is about to use:
    covered      another element (e.g. a cookie banner) sits on top of it
    enabled      it is not disabled (pages often disable buttons while loading)
    in_viewport  it is on screen
    page_age_ms  how long ago the page started loading
    + how it was found (strategy, confidence, number of fallbacks) and the action type.

The model predicts the probability that the action fails. Mitigations (dismiss overlays,
wait until enabled) cost time, so the agent only pays for them when the risk is high.
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import Locator, Page

STRATEGIES = ["selector", "label", "name", "text", "memory", "semantic", "vision"]
ACTIONS = ["fill", "select", "click"]

FEATURE_NAMES = (
    [f"action_{a}" for a in ACTIONS]
    + [f"strategy_{s}" for s in STRATEGIES]
    + [
        "is_submit",
        "confidence",
        "fallbacks",
        "step_index",
        "covered",
        "enabled",
        "in_viewport",
        "page_age_ms",
        "n_candidates",
    ]
)

_PROBE_JS = """(el) => {
  const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
  const top = document.elementFromPoint(cx, cy);
  const isButton = el.tagName === 'BUTTON' || el.type === 'submit';
  const kind = el.tagName === 'SELECT' ? 'select' : isButton ? 'button' : 'input';
  return {
    covered: !!top && top !== el && !el.contains(top),
    enabled: !el.disabled,
    in_viewport: r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth,
    page_age_ms: Math.round(performance.now()),
    n_candidates: document.querySelectorAll(
      kind === 'button' ? 'button, [type=submit]' : kind).length,
  };
}"""


def probe(locator: Locator, action: dict, strategy: str, confidence: float, fallbacks: int) -> dict:
    """Observe the element right before acting. Never raises: missing data -> defaults."""
    try:
        raw = locator.evaluate(_PROBE_JS)
    except Exception:  # noqa: BLE001
        raw = {}
    return {
        "action": action["action"],
        "strategy": strategy,
        "is_submit": action["element"].type == "submit",
        "confidence": float(confidence),
        "fallbacks": int(fallbacks),
        "step_index": int(action.get("index", 0)),
        "covered": bool(raw.get("covered", False)),
        "enabled": bool(raw.get("enabled", True)),
        "in_viewport": bool(raw.get("in_viewport", True)),
        "page_age_ms": float(raw.get("page_age_ms", 0)),
        "n_candidates": int(raw.get("n_candidates", 0)),
    }


def vectorize(f: dict) -> list[float]:
    row = [1.0 if f.get("action") == a else 0.0 for a in ACTIONS]
    row += [1.0 if f.get("strategy") == s else 0.0 for s in STRATEGIES]
    row += [
        float(f.get("is_submit", False)),
        float(f.get("confidence", 0.0)),
        float(f.get("fallbacks", 0)),
        float(f.get("step_index", 0)),
        float(f.get("covered", False)),
        float(f.get("enabled", True)),
        float(f.get("in_viewport", True)),
        float(f.get("page_age_ms", 0.0)),
        float(f.get("n_candidates", 0)),
    ]
    return row


class RiskModel:
    """Loads the trained XGBoost model and scores one step."""

    def __init__(self, booster, threshold: float = 0.5):
        self.booster = booster
        self.threshold = threshold

    @classmethod
    def load(cls, path: Path = Path("models/risk_model.json")) -> RiskModel:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        if meta.get("features", FEATURE_NAMES) != FEATURE_NAMES:
            raise ValueError("Model was trained on different features: retrain it.")
        return cls(booster, meta.get("threshold", 0.5))

    def score(self, features: dict) -> float:
        import numpy as np
        import xgboost as xgb

        matrix = xgb.DMatrix(np.array([vectorize(features)]), feature_names=FEATURE_NAMES)
        return float(self.booster.predict(matrix)[0])


# ---------- mitigations ----------

_DISMISS_WORDS = ["accept", "agree", "got it", "ok", "close", "dismiss", "continue", "×"]

_COVER_JS = """(el, words) => {
  const r = el.getBoundingClientRect();
  const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  if (!top || top === el || el.contains(top)) return null;
  // Walk up to the overlay container (fixed/absolute positioned, or a dialog).
  let c = top;
  while (c && c !== document.body) {
    const pos = getComputedStyle(c).position;
    if (pos === 'fixed' || pos === 'absolute' || c.getAttribute('role') === 'dialog') break;
    c = c.parentElement;
  }
  for (const b of (c || top).querySelectorAll('button, [role=button], a')) {
    const t = (b.innerText || b.getAttribute('aria-label') || '').trim().toLowerCase();
    if (words.some(w => t.includes(w))) {
      b.click();
      return t;
    }
  }
  return 'escape';
}"""


SETTLE_MS = 400  # cost of "stabilizing" a page: production agents pay this on every mitigation


def mitigate(page: Page, locator: Locator, timeout_ms: int = 3000) -> list[str]:
    """Generic fixes for the usual reasons an action fails. Returns what was done.

    Not free: it waits for the network to go idle and lets the page settle, which is why the
    agent only mitigates when the risk model says it is worth it.
    """
    done = []
    try:
        page.wait_for_load_state("networkidle", timeout=timeout_ms)
    except Exception:  # noqa: BLE001
        pass
    page.wait_for_timeout(SETTLE_MS)
    done.append("settled")
    try:
        clicked = locator.evaluate(_COVER_JS, _DISMISS_WORDS)
        if clicked == "escape":
            page.keyboard.press("Escape")
            done.append("escape")
        elif clicked:
            done.append(f"dismissed '{clicked}'")
    except Exception:  # noqa: BLE001
        pass
    try:
        if locator.is_disabled():
            handle = locator.element_handle(timeout=timeout_ms)
            page.wait_for_function("(el) => !el.disabled", arg=handle, timeout=timeout_ms)
            done.append("waited until enabled")
    except Exception:  # noqa: BLE001
        pass
    return done
