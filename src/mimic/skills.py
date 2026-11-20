"""Turn a recorded demonstration into a reusable, parametrised skill.

Recording:  fill Vendor = 'Bureau Express'
Skill:      fill Vendor <- {vendor}

Two passes:
1. Deterministic draft: every typed or selected value becomes a parameter named after its field.
2. LLM refinement (Ollama): writes a description, renames parameters, and decides which values
   are constants. If the model is unavailable or returns something invalid, the draft is kept.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel

from mimic.models import ElementInfo, Recording
from mimic.outcome import SuccessSignal, learn_signal

ACTIONS = ("fill", "select", "check", "choose", "upload", "press", "click")
VALUE_ACTIONS = ("fill", "select", "check", "choose", "upload")  # value can vary per task


class SkillStep(BaseModel):
    index: int
    action: Literal["fill", "select", "check", "choose", "upload", "press", "click"]
    element: ElementInfo
    parameter: str | None = None  # value comes from the task input
    value: str | None = None  # constant value (or the example from the demo)
    screenshot: str | None = None
    secret: bool = False  # password-like: never stored, given at run time (see secret_env)
    navigates: bool = False  # in the demo, this click / key press led to another page
    page: str | None = None  # URL path the step happened on in the demo


class Skill(BaseModel):
    name: str
    description: str = ""
    parameters: list[str] = []
    steps: list[SkillStep]
    source_recording: str | None = None
    start_url: str | None = None
    refined_by: Literal["heuristic", "llm"] = "heuristic"
    success: SuccessSignal | None = None  # what the end of a successful task looks like

    @property
    def secrets(self) -> list[str]:
        return [s.parameter for s in self.steps if s.secret and s.parameter]

    def render(self, inputs: dict) -> list[dict]:
        """Resolve parameters into concrete actions for one task (e.g. one invoice).

        Secret parameters come from the inputs if present, else from the environment
        variable MIMIC_SECRET_<NAME> (e.g. MIMIC_SECRET_PASSWORD). They are never stored.
        """
        secrets = set(self.secrets)
        missing = [p for p in self.parameters if p not in inputs and p not in secrets]
        if missing:
            raise KeyError(f"Missing inputs for skill '{self.name}': {missing}")
        demo_dir = Path(self.source_recording).parent if self.source_recording else None
        actions = []
        for step in self.steps:
            if step.secret:
                raw = inputs.get(step.parameter) or os.environ.get(secret_env(step.parameter))
                value = None if raw is None else str(raw)
            elif step.parameter:
                raw = inputs[step.parameter]
                value = None if raw is None else str(raw)
            else:
                value = step.value
            shot = str(demo_dir / step.screenshot) if demo_dir and step.screenshot else None
            actions.append(
                {
                    "index": step.index,
                    "parameter": step.parameter,
                    "action": step.action,
                    "element": step.element,
                    "value": value,
                    "screenshot": shot,
                    "secret": step.secret,
                    "navigates": step.navigates,
                    "page": step.page,
                }
            )
        return actions

    def embedding_text(self) -> str:
        fields = ", ".join(s.element.label or s.element.text or "" for s in self.steps)
        return f"{self.name.replace('_', ' ')}: {self.description}. Steps on: {fields}"


def secret_env(parameter: str) -> str:
    return f"MIMIC_SECRET_{parameter.upper()}"


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
        if out and e.kind not in ("click", "press") and out[-1].kind == e.kind:
            same = out[-1].element.selector == e.element.selector
            if e.kind == "choose":  # another option of the same radio group
                same = bool(e.element.name) and out[-1].element.name == e.element.name
            if same:
                out[-1] = e
                continue
        out.append(e)
    return out


def _same_page(a: str, b: str) -> bool:
    return a.split("#")[0] == b.split("#")[0]


def draft_skill(recording: Recording, source: str | None = None) -> Skill:
    steps, params = [], []
    events = _collapse(recording.events)
    final_url = recording.outcome.url if recording.outcome else None
    for i, event in enumerate(events):
        next_url = events[i + 1].url if i + 1 < len(events) else final_url
        navigates = (
            event.kind in ("click", "press")
            and next_url is not None
            and not _same_page(event.url, next_url)
        )
        param = None
        if event.kind in VALUE_ACTIONS:
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
                value=None if event.secret else event.value,
                screenshot=event.screenshot,
                secret=event.secret,
                navigates=navigates,
                page=urlsplit(event.url).path or "/",
            )
        )
    demo_values = [e.value for e in events if e.value and not e.secret]
    return Skill(
        name=recording.name,
        success=learn_signal(recording.outcome, recording.start_url, demo_values),
        description=f"Replay of the '{recording.name}' demonstration.",
        parameters=params,
        steps=steps,
        source_recording=source,
        start_url=recording.start_url,
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
        if s.action not in VALUE_ACTIONS or s.secret:
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
    value_steps = {s.index for s in draft.steps if s.action in VALUE_ACTIONS and not s.secret}
    # Models sometimes also answer for click steps: ignore anything that is not a value step.
    proposed = {
        s["index"]: s["parameter"] for s in answer.get("steps", []) if s["index"] in value_steps
    }
    if set(proposed) != value_steps:
        raise ValueError(f"LLM covered steps {sorted(proposed)}, expected {sorted(value_steps)}")

    names = [p for p in proposed.values() if p]
    if len(names) != len(set(names)) or not all(_IDENT.match(p) for p in names):
        raise ValueError(f"Invalid parameter names: {names}")
    if set(names) & set(draft.secrets):
        raise ValueError(f"Parameter names clash with secret fields: {names}")

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
            "parameters": [s.parameter for s in steps if s.parameter],
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
