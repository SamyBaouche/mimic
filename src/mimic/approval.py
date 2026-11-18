"""Human-in-the-loop approval: Mimic shows its plan and waits for you before acting.

Two checkpoints:
  1. plan review   before each task: the values it will type and the button it will press
                   [A]pprove  [E]dit  [S]kip  [T] approve all remaining  [Q]uit
  2. submit guard  (optional, --confirm-submit) right before a submit button is pressed,
                   once the form is filled: best used with --show to see the form
Safe by default: the CLI always reviews unless you pass --yes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Decision:
    action: Literal["approve", "skip", "quit"]
    edits: dict = field(default_factory=dict)


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def build_aliases(skill, mapping: dict) -> dict[str, str]:
    """Every way a person may name a value -> skill parameter.

    'Total amount' (label on screen), 'total' (field in the data), 'total_amount' (parameter).
    """
    aliases: dict[str, str] = {}
    for step in skill.steps:
        if not step.parameter:
            continue
        el = step.element
        names = [step.parameter, mapping.get(step.parameter), el.label, el.name, el.id]
        for name in filter(None, names):
            aliases.setdefault(_norm(name), step.parameter)
    return aliases


def _target(action: dict) -> str:
    el = action["element"]
    return el.label or el.text or el.name or el.selector


def format_plan(index: int, total: int, actions: list[dict], title: str = "") -> str:
    lines = [f"\nTask {index}/{total}{'  — ' + title if title else ''}"]
    for a in actions:
        if a["action"] == "click":
            lines.append(f"  then click   '{_target(a)}'")
        else:
            lines.append(f"  {_target(a):<18} <- {a['value']}")
    return "\n".join(lines)


class ConsoleApprover:
    def __init__(self, input_fn=input, print_fn=print, confirm_submit: bool = False):
        self.input = input_fn
        self.print = print_fn
        self.confirm_submit_enabled = confirm_submit
        self.auto = False
        self.aliases: dict[str, str] = {}  # set by the runner, see build_aliases
        self.stats = {"approved": 0, "edited": 0, "skipped": 0, "declined_submit": 0}

    def review(
        self, index: int, total: int, inputs: dict, render, title: str = ""
    ) -> tuple[Decision, list[dict]]:
        """Show the plan; return the decision and the (possibly edited) actions."""
        edits: dict = {}
        actions = render({**inputs, **edits})
        if self.auto:
            self.stats["approved"] += 1
            return Decision("approve"), actions

        while True:
            self.print(format_plan(index, total, actions, title))
            answer = self.input("[A]pprove  [E]dit  [S]kip  [T] approve all  [Q]uit > ")
            answer = answer.strip().lower()
            if answer in ("", "a"):
                break
            if answer == "t":
                self.auto = True
                break
            if answer == "s":
                self.stats["skipped"] += 1
                return Decision("skip"), actions
            if answer == "q":
                return Decision("quit"), actions
            if answer == "e":
                self.print("  Type the field as shown above, e.g.  Total amount=999.99")
                while line := self.input("  field=value (empty line to finish) > ").strip():
                    name, sep, value = line.partition("=")
                    param = self._resolve(name, inputs)
                    if not sep or param is None:
                        self.print(f"  ! unknown field '{name.strip()}'")
                        continue
                    edits[param] = value.strip()
                actions = render({**inputs, **edits})

        self.stats["approved"] += 1
        if edits:
            self.stats["edited"] += 1
        return Decision("approve", edits), actions

    def _resolve(self, name: str, inputs: dict) -> str | None:
        key = _norm(name)
        if not key:
            return None
        for param in inputs:
            if _norm(param) == key:
                return param
        return self.aliases.get(key)

    def confirm_submit(self, action: dict) -> bool:
        if not self.confirm_submit_enabled:
            return True
        answer = self.input(f"  Form filled. Press '{_target(action)}' now? [y/N] > ")
        ok = answer.strip().lower() in ("y", "yes", "o", "oui")
        if not ok:
            self.stats["declined_submit"] += 1
        return ok
