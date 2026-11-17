from pathlib import Path

import pytest

from mimic.models import Recording
from mimic.skills import apply_refinement, draft_skill, refine_skill

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"
INVOICE = {
    "vendor": "PaperPro Inc.",
    "invoice_number": "INV-20260042",
    "date": "2026-05-01",
    "total": 1200.5,
    "category": "Office supplies",
}


@pytest.fixture
def recording() -> Recording:
    return Recording.model_validate_json(FIXTURE.read_text())


def test_draft_turns_typed_values_into_parameters(recording):
    skill = draft_skill(recording)
    assert skill.parameters == ["vendor", "invoice_number", "date", "total", "category"]
    assert [s.action for s in skill.steps] == ["fill"] * 4 + ["select", "click"]
    assert skill.steps[-1].parameter is None  # the Save click has no value


def test_render_substitutes_a_new_invoice(recording):
    actions = draft_skill(recording).render(INVOICE)
    assert actions[0]["value"] == "PaperPro Inc."
    assert actions[3]["value"] == "1200.5"
    assert actions[-1]["action"] == "click" and actions[-1]["value"] is None


def test_render_reports_missing_inputs(recording):
    with pytest.raises(KeyError, match="total"):
        draft_skill(recording).render({k: v for k, v in INVOICE.items() if k != "total"})


def test_repeated_edits_of_the_same_field_are_collapsed(recording):
    first = recording.events[0].model_copy(update={"value": "Typo Vendr"})
    recording.events.insert(0, first)
    skill = draft_skill(recording)
    assert skill.steps[0].value == "Bureau Express"
    assert len(skill.steps) == 6


def test_duplicate_field_names_get_unique_parameters(recording):
    recording.events[1].element.name = "vendor"
    recording.events[1].element.id = "vendor"
    assert draft_skill(recording).parameters[:2] == ["vendor", "vendor_2"]


def _llm_answer(**overrides):
    answer = {
        "description": "Records a supplier invoice in the accounts payable app.",
        "steps": [
            {"index": 0, "parameter": "supplier"},
            {"index": 1, "parameter": "invoice_number"},
            {"index": 2, "parameter": "invoice_date"},
            {"index": 3, "parameter": "amount"},
            {"index": 4, "parameter": "category"},
        ],
    }
    answer.update(overrides)
    return answer


def test_valid_llm_answer_renames_parameters(recording):
    skill = apply_refinement(draft_skill(recording), _llm_answer())
    assert skill.refined_by == "llm"
    assert skill.parameters[0] == "supplier"
    assert skill.description.startswith("Records a supplier invoice")


def test_llm_answer_for_click_steps_is_ignored(recording):
    """Regression: qwen2.5 also answered for the Save click (index 5)."""
    answer = _llm_answer()
    answer["steps"].append({"index": 5, "parameter": None})
    skill = apply_refinement(draft_skill(recording), answer)
    assert skill.refined_by == "llm"
    assert skill.steps[5].action == "click" and skill.steps[5].parameter is None


def test_prompt_only_lists_value_steps(recording):
    from mimic.skills import _describe_for_llm

    prompt = _describe_for_llm(draft_skill(recording))
    assert "index=5" not in prompt
    assert "Save invoice" in prompt


def test_llm_can_mark_a_value_as_constant(recording):
    answer = _llm_answer()
    answer["steps"][4]["parameter"] = None
    skill = apply_refinement(draft_skill(recording), answer)
    assert "category" not in skill.parameters
    assert (
        skill.render({"supplier": "X", "invoice_number": "1", "invoice_date": "d", "amount": 1})[4][
            "value"
        ]
        == "Software"
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"steps": [{"index": 0, "parameter": "vendor"}]},  # missing steps
        {"steps": [{"index": i, "parameter": "same"} for i in range(5)]},  # duplicates
        {"steps": [{"index": i, "parameter": f"Bad Name {i}"} for i in range(5)]},  # invalid
        {"description": "  "},  # empty description
    ],
)
def test_invalid_llm_answers_are_rejected(recording, bad):
    with pytest.raises(ValueError):
        apply_refinement(draft_skill(recording), _llm_answer(**bad))


class FakeClient:
    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error

    def chat_json(self, *args):
        if self.error:
            raise self.error
        return self.answer


def test_refine_falls_back_to_draft_when_llm_fails(recording):
    draft = draft_skill(recording)
    skill = refine_skill(draft, FakeClient(error=ConnectionError("ollama down")), "m")
    assert skill == draft


def test_refine_falls_back_when_llm_answer_is_invalid(recording):
    draft = draft_skill(recording)
    assert refine_skill(draft, FakeClient(answer={"description": "x", "steps": []}), "m") == draft
