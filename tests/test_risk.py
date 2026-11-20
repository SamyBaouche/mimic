"""Chaos sandbox, pre-action features, mitigations and the XGBoost risk model."""

import random

import numpy as np
import pytest

from mimic.risk import FEATURE_NAMES, RiskModel, vectorize
from mimic.riskml import _best_threshold, _smart_rule, train
from mimic.sandbox.app import draw_chaos


def test_chaos_is_reproducible_with_a_seed():
    assert draw_chaos(0.5, random.Random(7)) == draw_chaos(0.5, random.Random(7))
    assert draw_chaos(0.0, random.Random(7)) == {}


def test_vector_matches_feature_names():
    f = {"action": "click", "strategy": "vision", "covered": True, "enabled": False}
    row = vectorize(f)
    assert len(row) == len(FEATURE_NAMES)
    named = dict(zip(FEATURE_NAMES, row, strict=True))
    assert named["action_click"] == 1 and named["strategy_vision"] == 1
    assert named["covered"] == 1 and named["enabled"] == 0


def _synthetic(n=600, seed=0):
    """Clicks on covered or disabled elements fail; a little label noise like real servers."""
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        f = {
            "action": rng.choice(["fill", "fill", "select", "click"]),
            "strategy": "selector",
            "is_submit": False,
            "confidence": 1.0,
            "fallbacks": rng.choice([0, 0, 1]),
            "step_index": rng.randint(0, 5),
            "covered": rng.random() < 0.3,
            "enabled": rng.random() > 0.2,
            "in_viewport": True,
            "page_age_ms": rng.uniform(100, 3000),
            "n_candidates": rng.randint(1, 6),
        }
        f["is_submit"] = f["action"] == "click"
        failed = _smart_rule(f) == 1.0
        if rng.random() < 0.03:
            failed = not failed
        samples.append({"run_id": f"run{i % 6}", "features": f, "failed": failed})
    return samples


def test_training_learns_the_interaction_and_saves_a_usable_model(tmp_path):
    out = tmp_path / "risk_model.json"
    meta = train(_synthetic(), out=out)
    xgb = meta["test_metrics"]["xgboost"]
    assert xgb["roc_auc"] > 0.85  # 3% label noise on purpose
    assert xgb["roc_auc"] > meta["test_metrics"]["rule_covered_or_disabled"]["roc_auc"]
    assert out.exists() and out.with_suffix(".meta.json").exists()

    model = RiskModel.load(out)
    blocked = {"action": "click", "is_submit": True, "covered": True, "enabled": True}
    fine = {"action": "fill", "covered": True, "enabled": True}
    assert model.score(blocked) > model.score(fine)


def test_training_refuses_too_little_data():
    with pytest.raises(SystemExit):
        train(_synthetic(n=20))


def test_threshold_favours_recall():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.55, 0.6, 0.9])
    assert _best_threshold(y, scores) <= 0.6  # never sacrifices a failure for precision


def test_model_with_other_features_is_rejected(tmp_path):
    out = tmp_path / "m.json"
    train(_synthetic(), out=out)
    meta = out.with_suffix(".meta.json")
    meta.write_text(meta.read_text().replace('"covered"', '"something_else"'))
    with pytest.raises(ValueError, match="retrain"):
        RiskModel.load(out)


# ---------------------------------------------------------------- browser


pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.agent import ReplayAgent  # noqa: E402
from mimic.recorder import launch_browser  # noqa: E402
from mimic.risk import mitigate, probe  # noqa: E402


def _seed_with(**wanted):
    for s in range(2000):
        c = draw_chaos(0.5, random.Random(s))
        if all(c.get(k) == v for k, v in wanted.items()):
            return s
    raise AssertionError("no seed found")


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = launch_browser(p, headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")
        yield b
        b.close()


def test_probe_sees_the_cookie_banner_and_mitigation_removes_it(browser, sandbox_server):
    seed = _seed_with(overlay=True, overlay_escape=False)
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(f"{sandbox_server}?chaos=0.5&seed={seed}")
    button = page.locator("#save-btn")
    action = {"action": "click", "element": type("E", (), {"type": "submit"})(), "index": 5}
    assert probe(button, action, "selector", 1.0, 0)["covered"] is True

    done = mitigate(page, button)
    assert any("accept" in d for d in done)
    assert probe(button, action, "selector", 1.0, 0)["covered"] is False
    page.close()


def test_mitigation_waits_for_a_disabled_submit(browser, sandbox_server):
    seed = None
    for s in range(2000):
        c = draw_chaos(0.5, random.Random(s))
        if not c["overlay"] and c["slow_submit_ms"] > 1500:
            seed = s
            break
    page = browser.new_page()
    page.goto(f"{sandbox_server}?chaos=0.5&seed={seed}")
    button = page.locator("#save-btn")
    assert button.is_disabled()
    assert "waited until enabled" in mitigate(page, button)
    assert button.is_enabled()
    page.close()


def test_predicted_policy_requires_a_model(browser):
    page = browser.new_page()
    with pytest.raises(ValueError, match="trained risk model"):
        ReplayAgent(page, risk_policy="predicted")
    page.close()
