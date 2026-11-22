"""Human-in-the-loop approval: Mimic shows its plan and waits for you before acting.

Two checkpoints:
  1. plan review   before each task: the values it will type and the button it will press
                   [A]pprove  [E]dit  [S]kip  [T] approve all remaining  [Q]uit
  2. submit guard  (optional, --confirm-submit) right before a submit button is pressed,
                   once the form is filled: best used with --show to see the form
Safe by default: the CLI always reviews unless you pass --yes.
"""

from __future__ import annotations

import queue
import threading
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


SECRET_MASK = "********"


def display_value(a: dict) -> str:
    """How a planned value is shown to a person: masked, yes/no, file name..."""
    from pathlib import Path

    from mimic.models import as_bool

    if a["value"] in (None, "None"):
        return "(missing)"
    if a.get("secret"):
        return SECRET_MASK
    if a["action"] == "check":
        return "yes" if as_bool(a["value"]) else "no"
    if a["action"] == "upload":
        return Path(str(a["value"])).name
    return str(a["value"])


def format_plan(
    index: int, total: int, actions: list[dict], title: str = "", flags: dict | None = None
) -> str:
    flags = flags or {}
    lines = [f"\nTask {index}/{total}{'  — ' + title if title else ''}"]
    for a in actions:
        if a["action"] == "click":
            lines.append(f"  then click   '{_target(a)}'")
            continue
        if a["action"] == "press":
            lines.append(f"  then press   {a['value']} in '{_target(a)}'")
            continue
        value = display_value(a)
        note = flags.get(a.get("parameter"), "")
        warn = f"   <- check: {note}" if note else ""
        lines.append(f"  {_target(a):<18} <- {value}{warn}")
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
        self, index: int, total: int, inputs: dict, render, title: str = "", flags=None
    ) -> tuple[Decision, list[dict]]:
        """Show the plan; return the decision and the (possibly edited) actions."""
        edits: dict = {}
        actions = render({**inputs, **edits})
        if self.auto:
            self.stats["approved"] += 1
            return Decision("approve"), actions

        while True:
            self.print(format_plan(index, total, actions, title, flags))
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


# ---------------------------------------------------------------- dashboard


def plan_label(action: dict) -> str:
    el = action["element"]
    return el.label or el.text or el.name or el.selector


def plan_rows(actions: list[dict], flags: dict | None = None) -> list[dict]:
    """The plan in a JSON-friendly shape, for the dashboard."""
    flags = flags or {}
    return [
        {
            "index": a["index"],
            "action": a["action"],
            "label": plan_label(a),
            "parameter": a.get("parameter"),
            "value": None if a["value"] in (None, "None") else display_value(a),
            "secret": bool(a.get("secret")),
            "flag": flags.get(a.get("parameter"), ""),
        }
        for a in actions
    ]


class WebApprover(ConsoleApprover):
    """Same contract as ConsoleApprover, but the decisions come from the dashboard buttons.

    The agent thread blocks in review() until the page sends a decision over the WebSocket:
    {"action": "approve" | "approve_all" | "skip" | "quit" | "edit", "edits": {...}}
    Pause / resume / stop are handled between tasks by checkpoint().
    """

    def __init__(self, publish, confirm_submit: bool = False, poll_s: float = 0.2):
        super().__init__(input_fn=None, print_fn=lambda *_: None, confirm_submit=confirm_submit)
        self.publish = publish
        self.poll_s = poll_s
        self._decisions: queue.Queue = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self.stopped = False

    # --- called by the server (asyncio side) ---
    def decide(self, message: dict) -> None:
        action = message.get("action")
        if action == "pause":
            self._running.clear()
            self.publish({"type": "paused", "paused": True})
        elif action == "resume":
            self._running.set()
            self.publish({"type": "paused", "paused": False})
        elif action == "stop":
            self.stopped = True
            self._running.set()
            self._decisions.put({"action": "quit"})
        else:
            self._decisions.put(message)

    # --- called by the agent thread ---
    def _wait(self) -> dict:
        while True:
            if self.stopped:
                return {"action": "quit"}
            try:
                return self._decisions.get(timeout=self.poll_s)
            except queue.Empty:
                continue

    def checkpoint(self) -> bool:
        """Between tasks: wait while paused. False means stop the run."""
        while not self._running.wait(timeout=self.poll_s):
            if self.stopped:
                break
        return not self.stopped

    def review(self, index, total, inputs, render, title="", flags=None):
        actions = render(inputs)
        if self.auto:
            self.stats["approved"] += 1
            return Decision("approve"), actions
        edits: dict = {}
        self.publish(
            {
                "type": "review",
                "index": index,
                "total": total,
                "title": title,
                "plan": plan_rows(actions, flags),
            }
        )
        while True:
            msg = self._wait()
            action = msg.get("action")
            if action in ("approve", "approve_all"):
                self.auto = action == "approve_all"
                break
            if action == "skip":
                self.stats["skipped"] += 1
                return Decision("skip"), actions
            if action == "quit":
                return Decision("quit"), actions
            if action == "edit":
                unknown = []
                for name, value in (msg.get("edits") or {}).items():
                    param = self._resolve(name, inputs)
                    if param is None:
                        unknown.append(name)
                    else:
                        edits[param] = str(value)
                actions = render({**inputs, **edits})
                self.publish(
                    {
                        "type": "review",
                        "index": index,
                        "total": total,
                        "title": title,
                        "plan": plan_rows(actions, flags),
                        "unknown_fields": unknown,
                    }
                )
                if msg.get("then") == "approve" and not unknown:
                    break
        self.stats["approved"] += 1
        if edits:
            self.stats["edited"] += 1
        return Decision("approve", edits), actions

    def confirm_submit(self, action: dict) -> bool:
        if not self.confirm_submit_enabled:
            return True
        self.publish({"type": "confirm_submit", "label": plan_label(action)})
        ok = self._wait().get("action") in ("approve", "yes")
        if not ok:
            self.stats["declined_submit"] += 1
        return ok


class WebWorkflowReview:
    """Approve each email of a workflow from the dashboard (same buttons, same pause / stop)."""

    def __init__(self, approver: WebApprover):
        self.approver = approver

    def checkpoint(self) -> bool:
        return self.approver.checkpoint()

    def __call__(self, text: str, item=None, index: int = 0, total: int = 0) -> str:
        from mimic.workflow import plan_groups

        if self.approver.auto or item is None:
            return "approve"
        self.approver.publish(
            {
                "type": "item_review",
                "index": index,
                "total": total,
                "title": item.title,
                "flags": item.flags,
                "error": item.error,
                "steps": plan_groups(item),
            }
        )
        while True:
            action = self.approver._wait().get("action")
            if action in ("approve", "approve_all"):
                self.approver.auto = action == "approve_all"
                return "approve"
            if action in ("skip", "quit"):
                return action
