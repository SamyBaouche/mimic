"""Element matcher: dataset, training pipeline, and self-healing with the matcher."""

import sys
from pathlib import Path

import pytest

from mimic.matcher_data import (
    CONCEPTS,
    HELD_OUT,
    make_examples,
    sandbox_v2_examples,
    to_pairs,
)

sys.path.insert(0, str(Path(__file__).parent))


def test_held_out_labels_never_appear_in_training():
    seen = {el.label for ex in make_examples(3000, seed=5) for el in [ex.target, *ex.candidates]}
    assert not seen & HELD_OUT


def test_every_held_out_label_exists_in_the_banks():
    banks = {label for _, _, labels in CONCEPTS.values() for label in labels}
    assert HELD_OUT <= banks


def test_examples_have_one_right_answer_of_the_same_kind():
    for ex in make_examples(200, seed=3):
        assert ex.candidates[ex.answer].concept == ex.target.concept
        assert (
            ex.candidates[ex.answer].label != ex.target.label
            or len(CONCEPTS[ex.target.concept][2]) == 1
        )
        assert all(c.kind == ex.target.kind for c in ex.candidates)
        assert sum(c.concept == ex.target.concept for c in ex.candidates) == 1


def test_sandbox_examples_point_at_the_real_v2_elements():
    answers = {ex.target.label: ex.candidates[ex.answer].label for ex in sandbox_v2_examples()}
    assert answers["Save invoice"] == "Record"
    assert answers["Vendor"] == "Supplier name"
    assert answers["Total amount"] == "Amount (CAD)"


def test_pairs_have_one_positive_per_example():
    ex = make_examples(50, seed=4)
    pairs = to_pairs(ex)
    assert sum(y for _, _, y in pairs) == 50
    assert len(pairs) == sum(len(e.candidates) for e in ex)


def test_string_similarity_baseline():
    from mimic.matcher import string_similarity

    scores = string_similarity("button 'Save'", ["button 'Saved'", "button 'Export'"])
    assert scores[0] > scores[1]


# ---------------------------------------------------------------- with torch

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tiny_model import build_tiny  # noqa: E402

from mimic.matcher import ElementMatcher, train  # noqa: E402


@pytest.fixture(scope="module")
def tiny_trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("matcher")
    model, tok = build_tiny()
    meta = train(
        model,
        tok,
        make_examples(2000, seed=1),
        make_examples(300, seed=2, split="test"),
        epochs=4,
        lr=1e-3,
        out=out,
        base_name="tiny",
        log=lambda *_: None,
    )
    return out, meta


def test_training_pipeline_learns_and_reports(tiny_trained):
    """Mechanics only. Model QUALITY is measured by `mimic.matcher train` on the real
    pre-trained model: a random tiny model's accuracy varies across platforms."""
    out, meta = tiny_trained
    losses = meta["loss_per_epoch"]
    assert losses[-1] < losses[0]  # it learns
    r = meta["results"]
    assert {"fine_tuned", "before_fine_tuning", "string_similarity"} <= set(r)
    assert 0 <= r["fine_tuned"]["synthetic_top1"] <= 1
    assert r["fine_tuned"]["ms_per_decision_cpu"] < 500
    assert (out / "meta.json").exists()


def test_saved_matcher_reloads_with_identical_scores(tiny_trained):
    out, _ = tiny_trained
    pair = ("button 'Save invoice'", ["button 'Export CSV' (button)", "button 'Record'"])
    first = ElementMatcher.load(out).scores(*pair)
    second = ElementMatcher.load(out).scores(*pair)
    assert first == pytest.approx(second)
    assert all(0 <= x <= 1 for x in first)


# ---------------------------------------------------------------- healing

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.healing import Healer  # noqa: E402
from mimic.models import Recording  # noqa: E402
from mimic.recorder import launch_browser  # noqa: E402
from mimic.run import run_batch  # noqa: E402
from mimic.sandbox.generate_invoices import generate  # noqa: E402
from mimic.skills import draft_skill  # noqa: E402
from mimic.vision import VisionChoice  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"


@pytest.fixture(scope="module")
def browser_ok():
    with sync_playwright() as p:
        try:
            launch_browser(p, headless=True).close()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")


class CountingVision:
    def __init__(self):
        self.calls = 0

    def choose(self, png, target, action, candidates, reference=None):
        self.calls += 1
        for i, c in enumerate(candidates):
            if c.selector == "#btn-submit-bill":
                return VisionChoice(i, "fake", {"choice": i}, 0.9)
        return VisionChoice(-1, "fake", {})


class ConfidentMatcher:
    """Behaves like a well-trained matcher: sure that 'Record' replaces 'Save invoice'."""

    threshold = 0.5

    def scores(self, target, candidates):
        return [0.95 if "Record" in c else 0.05 for c in candidates]


class FlatMatcher:
    """A matcher that cannot tell candidates apart."""

    threshold = 0.5

    def scores(self, target, candidates):
        return [0.6] * len(candidates)


def _run(url, healer, n=3):
    skill = draft_skill(Recording.model_validate_json(FIXTURE.read_text()))
    return run_batch(skill, generate(n), url, "v2", log_to_db=False, out_dir=None, healer=healer)


def test_confident_matcher_heals_v2_without_the_vision_model(sandbox_server, browser_ok):
    vision = CountingVision()
    healer = Healer(matcher=ConfidentMatcher(), vision=vision)
    report = _run(sandbox_server, healer)
    assert report["task_success_rate"] == 1.0
    assert vision.calls == 0  # the 7B model was never needed
    assert report["healing"]["healed"] == {"matcher": 1, "memory": 2}


def test_unsure_matcher_falls_back_to_vision(sandbox_server, browser_ok):
    vision = CountingVision()
    report = _run(sandbox_server, Healer(matcher=FlatMatcher(), vision=vision), n=2)
    assert report["task_success_rate"] == 1.0
    assert vision.calls == 1
    assert report["healing"]["healed"] == {"vision": 1, "memory": 1}
