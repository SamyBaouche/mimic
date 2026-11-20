"""Self-healing on the redesigned layout (v2), with controllable fake embedding and vision models.

The real models (nomic-embed-text, qwen2.5vl) run through the same interfaces on your machine.
"""

import io
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("playwright")
pytest.importorskip("langgraph")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.healing import Healer, HealMemory  # noqa: E402
from mimic.models import ElementInfo, Recording  # noqa: E402
from mimic.recorder import launch_browser  # noqa: E402
from mimic.run import run_batch  # noqa: E402
from mimic.sandbox.generate_invoices import generate  # noqa: E402
from mimic.skills import draft_skill  # noqa: E402
from mimic.vision import VisionChoice, VisionLocator, annotate_candidates, annotate_reference  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"


@pytest.fixture(scope="module")
def v2_centers(sandbox_server):
    """Skip cleanly when no browser is available; selectors of the v2 buttons."""
    with sync_playwright() as p:
        try:
            launch_browser(p, headless=True).close()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")
    return {s: s for s in ["#btn-submit-bill", "#btn-export", "#btn-clear"]}


@pytest.fixture
def skill():
    return draft_skill(Recording.model_validate_json(FIXTURE.read_text()))


class FakeVision:
    """Picks the numbered candidate with the given selector (or none)."""

    def __init__(self, selector, confidence=0.4):
        self.selector = selector
        self.confidence = confidence
        self.calls = 0

    def choose(self, png, target, action, candidates, reference=None):
        self.calls += 1
        for i, c in enumerate(candidates):
            if c.selector == self.selector:
                return VisionChoice(i, "fake", {"choice": i}, self.confidence)
        return VisionChoice(-1, "fake", {"choice": -1})


def fake_embedder(texts):
    """'Save invoice' and 'Record' mean the same thing; Export and Clear do not."""
    out = []
    for t in texts:
        t = t.lower()
        if "save" in t or "record" in t:
            out.append([1.0, 0.0, 0.0])
        elif "export" in t:
            out.append([0.0, 1.0, 0.0])
        elif "clear" in t:
            out.append([0.0, 0.0, 1.0])
        else:
            out.append([0.3, 0.3, 0.3])
    return out


def _run(skill, url, healer, n=5):
    return run_batch(skill, generate(n), url, "v2", log_to_db=False, out_dir=None, healer=healer)


def test_vision_heals_v2_and_memory_reuses_the_fix(skill, sandbox_server, v2_centers):
    vision = FakeVision(v2_centers["#btn-submit-bill"])
    healer = Healer(vision=vision)
    report = _run(skill, sandbox_server, healer)
    assert report["task_success_rate"] == 1.0
    assert vision.calls == 1  # only the first task pays for vision
    assert report["healing"]["healed"] == {"vision": 1, "memory": 4}
    clicks = [s for s in report["steps"] if s["action"] == "click"]
    assert [s["strategy"] for s in clicks] == ["vision"] + ["memory"] * 4
    assert all(s["healed"] for s in clicks)


def test_wrong_vision_pick_is_detected_and_never_remembered(skill, sandbox_server, v2_centers):
    healer = Healer(vision=FakeVision(v2_centers["#btn-export"]))
    report = _run(skill, sandbox_server, healer, n=2)
    assert report["task_success_rate"] == 0.0  # Export does not submit: caught, not "success"
    assert report["agent_reported_success_rate"] == 0.0
    assert healer.memory.cache == {}
    assert healer.stats.failed == 2


def test_semantic_match_heals_without_vision(skill, sandbox_server, v2_centers):
    vision = FakeVision(v2_centers["#btn-export"])  # would be wrong, must not be needed
    healer = Healer(embedder=fake_embedder, vision=vision)
    report = _run(skill, sandbox_server, healer, n=3)
    assert report["task_success_rate"] == 1.0
    assert vision.calls == 0
    assert report["healing"]["healed"] == {"semantic": 1, "memory": 2}


def test_ambiguous_semantics_fall_through_to_vision(skill, sandbox_server, v2_centers):
    vision = FakeVision(v2_centers["#btn-submit-bill"])
    healer = Healer(embedder=lambda texts: [[1.0, 1.0] for _ in texts], vision=vision)
    report = _run(skill, sandbox_server, healer, n=2)
    assert report["task_success_rate"] == 1.0
    assert vision.calls == 1


def test_no_healer_is_the_baseline(skill, sandbox_server, v2_centers):
    report = _run(skill, sandbox_server, None, n=2)
    assert report["task_success_rate"] == 0.0


def padded_embedder(texts):
    """Same meaning as fake_embedder, at the real 768 dimensions of nomic-embed-text."""
    return [v + [0.0] * 765 for v in fake_embedder(texts)]


@pytest.mark.parametrize("embedder", [padded_embedder, fake_embedder], ids=["768d", "wrong-dim"])
def test_healing_memory_persists_in_storage(
    skill, sandbox_server, v2_centers, test_database, embedder
):
    from mimic.storage import get_storage

    get_storage().reset()
    first = Healer(embedder=embedder, vision=None, memory=HealMemory(use_db=True))
    _run(skill, sandbox_server, first, n=1)
    assert get_storage().stats()["healed_elements"] == 1

    # A brand-new run (empty in-run cache) finds the fix in storage without any model.
    second = Healer(embedder=embedder, vision=None, memory=HealMemory(use_db=True))
    second.semantic_scores = lambda *a: pytest.fail("memory should answer first")
    report = _run(skill, sandbox_server, second, n=2)
    assert report["task_success_rate"] == 1.0
    assert report["healing"]["healed"] == {"memory": 2}


# ---------- vision locator unit tests ----------


def _png(w=200, h=100) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


class FakeVLM:
    def __init__(self, answer):
        self.answer = answer
        self.images = None

    def chat_json(self, model, system, user, schema, images=None):
        self.images, self.schema, self.prompt = images, schema, user
        return self.answer


TARGET = ElementInfo(
    tag="button",
    selector="#save-btn",
    text="Save invoice",
    type="submit",
    bbox={"x": 10, "y": 10, "width": 50, "height": 20},
)


CANDIDATES = [
    ElementInfo(
        tag="button",
        selector=f"#b{i}",
        text=t,
        bbox={"x": 20 + 60 * i, "y": 40, "width": 50, "height": 20},
    )
    for i, t in enumerate(["Export CSV", "Clear", "Record"])
]


@pytest.mark.parametrize(
    "answer, expected",
    [
        ({"choice": 2, "reason": "submit", "confidence": 0.7}, 2),
        ({"choice": 7, "reason": "", "confidence": 0.9}, -1),
        ({"choice": "two", "reason": "", "confidence": 0.9}, -1),
    ],
)
def test_vision_choice_is_validated(answer, expected):
    pick = VisionLocator(FakeVLM(answer), "m").choose(_png(), TARGET, "click", CANDIDATES)
    assert pick.index == expected


def test_vision_schema_forces_a_numbered_choice():
    vlm = FakeVLM({"choice": 0, "reason": "", "confidence": 0.5})
    VisionLocator(vlm, "m").choose(_png(), TARGET, "click", CANDIDATES)
    assert vlm.schema["properties"]["choice"]["enum"] == [0, 1, 2]
    assert "will NOT have the same text" in vlm.prompt


def test_vision_sends_demo_and_numbered_screens(tmp_path):
    ref = tmp_path / "step.png"
    ref.write_bytes(_png())
    vlm = FakeVLM({"choice": 0, "reason": ""})
    VisionLocator(vlm, "m").choose(_png(), TARGET, "click", CANDIDATES, reference=ref)
    assert len(vlm.images) == 2
    marked = Image.open(io.BytesIO(annotate_reference(ref, TARGET))).convert("RGB")
    assert marked.getpixel((8, 8)) == (255, 0, 0)  # red box around the demo element
    som = Image.open(io.BytesIO(annotate_candidates(_png(), CANDIDATES))).convert("RGB")
    assert som.getpixel((17, 50)) != (255, 255, 255)  # numbered box drawn around candidate 0


def test_vision_is_skipped_without_candidates():
    assert VisionLocator(FakeVLM({"choice": 0}), "m").choose(_png(), TARGET, "click", []) is None


def test_uncertain_vision_pick_is_refused_for_fields(sandbox_server, v2_centers):
    """A low-confidence guess on a field could silently fill the wrong input."""
    from playwright.sync_api import sync_playwright as sp

    target = ElementInfo(
        tag="input",
        selector="#vendor",
        label="Vendor",
        bbox={"x": 0, "y": 0, "width": 10, "height": 10},
    )
    with sp() as p:
        browser = launch_browser(p, headless=True)
        page = browser.new_page()
        page.goto(f"{sandbox_server}?layout=v2")
        action = {"index": 0, "action": "fill", "element": target, "value": "X"}
        unsure = Healer(vision=FakeVision("#f-supplier", confidence=0.3))
        sure = Healer(vision=FakeVision("#f-supplier", confidence=0.9))
        assert unsure.heal(page, action, "s") is None
        assert sure.heal(page, action, "s").selector == "#f-supplier"
        browser.close()


def test_failing_step_stops_calling_models(skill, sandbox_server, v2_centers):
    vision = FakeVision("#btn-export")  # always wrong
    healer = Healer(vision=vision)
    report = _run(skill, sandbox_server, healer, n=6)
    assert report["task_success_rate"] == 0.0
    assert vision.calls == 3  # gave up after 3 rejected heals instead of paying 6 times
