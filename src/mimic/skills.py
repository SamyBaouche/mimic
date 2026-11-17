"""Turn a recorded demonstration into a reusable, parametrised skill.

Recording:  fill Vendor = 'Bureau Express'
Skill:      fill Vendor <- {vendor}

Two passes:
1. Deterministic draft: every typed or selected value becomes a parameter named after its field.
2. LLM refinement (Ollama): writes a description, renames parameters, and decides which values
   are constants. If the model is unavailable or returns something invalid, the draft is kept.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from mimic.models import ElementInfo, Recording


class SkillStep(BaseModel):
    index: int
    action: Literal["fill", "select", "click"]
    element: ElementInfo
    parameter: str | None = None  # value comes from the task input
    value: str | None = None  # constant value (or the example from the demo)
    screenshot: str | None = None


class Skill(BaseModel):
    name: str
    description: str = ""
    parameters: list[str] = []
    steps: list[SkillStep]
    source_recording: str | None = None
    refined_by: Literal["heuristic", "llm"] = "heuristic"

    def render(self, inputs: dict) -> list[dict]:
        """Resolve parameters into concrete actions for one task (e.g. one invoice)."""
        missing = [p for p in self.parameters if p not in inputs]
        if missing:
            raise KeyError(f"Missing inputs for skill '{self.name}': {missing}")
        actions = []
        for step in self.steps:
            value = str(inputs[step.parameter]) if step.parameter else step.value
            actions.append({"action": step.action, "element": step.element, "value": value})
        return actions

    def embedding_text(self) -> str:
        fields = ", ".join(s.element.label or s.element.text or "" for s in self.steps)
        return f"{self.name.replace('_', ' ')}: {self.description}. Steps on: {fields}"


# ---------- Pass 1: deterministic draft ----------


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug or slug[0].isdigit():
        slug = f"field_{slug}"
    return slug


def _collapse(events):
    """Keep only the last value when the user edits the same field several times in a row."""
    out = []
    for e in events:
        if out and e.kind != "click" and out[-1].kind == e.kind:
            if out[-1].element.selector == e.element.selector:
                out[-1] = e
                continue
        out.append(e)
    return out


def draft_skill(recording: Recording, source: str | None = None) -> Skill:
    steps, params = [], []
    for i, event in enumerate(_collapse(recording.events)):
        param = None
        if event.kind in ("fill", "select"):
            el = event.element
            base = _slug(el.name or el.id or el.label or f"field_{i}")
            param, n = base, 2
            while param in params:
                param, n = f"{base}_{n}", n + 1
            params.append(param)
        steps.append(
            SkillStep(
                index=i,
                action=event.kind,
                element=event.element,
                parameter=param,
                value=event.value,
                screenshot=event.screenshot,
            )
        )
    return Skill(
        name=recording.name,
        description=f"Replay of the '{recording.name}' demonstration.",
        parameters=params,
        steps=steps,
        source_recording=source,
    )


# ---------- Pass 2: LLM refinement ----------

SYSTEM_PROMPT = """You convert a recorded UI demonstration into a reusable automation skill.
For each step that types or selects a value, decide:
- parameter: a short snake_case name if the value changes from one task to the next
  (e.g. names, numbers, dates, amounts, categories), or null if it is always the same.
Also write a one-sentence description of what the skill accomplishes.
Answer only with JSON matching the schema."""

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "parameter": {"type": ["string", "null"]},
                },
                "required": ["index", "parameter"],
            },
        },
    },
    "required": ["description", "steps"],
}

_IDENT = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def _describe_for_llm(skill: Skill) -> str:
    lines = [
        f"Skill name: {skill.name}",
        "Steps that enter a value (answer for exactly these indexes):",
    ]
    for s in skill.steps:
        if s.action == "click":
            continue
        el = s.element
        target = el.label or el.text or el.name or el.selector
        lines.append(f"- index={s.index} action={s.action} target={target!r} example={s.value!r}")
    clicks = [s.element.text or s.element.selector for s in skill.steps if s.action == "click"]
    if clicks:
        lines.append(f"The skill then clicks: {', '.join(clicks)} (no answer needed).")
    return "\n".join(lines)


def apply_refinement(draft: Skill, answer: dict) -> Skill:
    """Validate the LLM answer and merge it into the draft. Raises ValueError if invalid."""
    value_steps = {s.index for s in draft.steps if s.action != "click"}
    # Models sometimes also answer for click steps: ignore anything that is not a value step.
    proposed = {
        s["index"]: s["parameter"] for s in answer.get("steps", []) if s["index"] in value_steps
    }
    if set(proposed) != value_steps:
        raise ValueError(f"LLM covered steps {sorted(proposed)}, expected {sorted(value_steps)}")

    names = [p for p in proposed.values() if p]
    if len(names) != len(set(names)) or not all(_IDENT.match(p) for p in names):
        raise ValueError(f"Invalid parameter names: {names}")

    description = (answer.get("description") or "").strip()
    if not description:
        raise ValueError("Empty description")

    steps = [
        s.model_copy(update={"parameter": proposed[s.index]}) if s.index in proposed else s
        for s in draft.steps
    ]
    return draft.model_copy(
        update={
            "description": description,
            "parameters": names,
            "steps": steps,
            "refined_by": "llm",
        }
    )


def refine_skill(draft: Skill, client, model: str) -> Skill:
    """Ask the LLM to refine the draft; fall back to the draft on any failure."""
    try:
        answer = client.chat_json(model, SYSTEM_PROMPT, _describe_for_llm(draft), REFINE_SCHEMA)
        return apply_refinement(draft, answer)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! LLM refinement skipped ({type(exc).__name__}: {exc}). Using heuristic draft.")
        return draft
