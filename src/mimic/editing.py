"""Change a learned skill without recording it again.

    mimic skills edit enter_invoice                       # interactive
    mimic skills edit enter_invoice --fix category=Software --remove 6 --rename vendor=supplier
    mimic skills export enter_invoice --out enter_invoice.json
    mimic skills import enter_invoice.json                # share a skill, restore a backup

Every edit is checked before it is saved, and the previous version is kept in
~/.mimic/backups so it can be restored with `mimic skills import`.
"""

from __future__ import annotations

import json
import shlex
from datetime import datetime
from pathlib import Path

from mimic.skills import _IDENT, VALUE_ACTIONS, Skill


class EditError(ValueError):
    pass


def _with_steps(skill: Skill, steps) -> Skill:
    params = list(dict.fromkeys(s.parameter for s in steps if s.parameter))
    return skill.model_copy(update={"steps": steps, "parameters": params})


def _step(skill: Skill, index: int):
    found = next((s for s in skill.steps if s.index == index), None)
    if found is None:
        raise EditError(f"no step {index} (steps: {', '.join(str(s.index) for s in skill.steps)})")
    return found


def _check_name(skill: Skill, name: str) -> None:
    if not _IDENT.match(name):
        raise EditError(f"'{name}' is not a valid input name (lowercase, digits, _)")
    if name in skill.parameters:
        raise EditError(f"an input is already called '{name}'")


def remove_steps(skill: Skill, indices: list[int]) -> Skill:
    for i in indices:
        _step(skill, i)
    steps = [s for s in skill.steps if s.index not in indices]
    if not steps:
        raise EditError("a skill needs at least one step")
    return _with_steps(skill, steps)


def fix_value(skill: Skill, parameter: str, value: str) -> Skill:
    """The input `parameter` becomes a constant: every task types `value`."""
    if parameter not in skill.parameters:
        raise EditError(f"no input '{parameter}' (inputs: {', '.join(skill.parameters)})")
    if parameter in skill.secrets:
        raise EditError(f"'{parameter}' is a secret: it is never stored in a skill")
    steps = [
        s.model_copy(update={"parameter": None, "value": value}) if s.parameter == parameter else s
        for s in skill.steps
    ]
    return _with_steps(skill, steps)


def make_input(skill: Skill, index: int, name: str) -> Skill:
    """Step `index` takes its value from the input `name` (each task can give another one)."""
    step = _step(skill, index)
    if step.action not in VALUE_ACTIONS:
        raise EditError(f"step {index} is a {step.action}: it has no value to vary")
    if step.secret:
        raise EditError(f"step {index} is already a secret input ('{step.parameter}')")
    if step.parameter == name:
        return skill
    _check_name(skill, name)
    steps = [
        s.model_copy(update={"parameter": name}) if s.index == index else s for s in skill.steps
    ]
    return _with_steps(skill, steps)


def rename_input(skill: Skill, old: str, new: str) -> Skill:
    if old not in skill.parameters:
        raise EditError(f"no input '{old}' (inputs: {', '.join(skill.parameters)})")
    _check_name(skill, new)
    steps = [
        s.model_copy(update={"parameter": new}) if s.parameter == old else s for s in skill.steps
    ]
    return _with_steps(skill, steps)


def describe(skill: Skill, text: str) -> Skill:
    if not text.strip():
        raise EditError("the description cannot be empty")
    return skill.model_copy(update={"description": text.strip()})


def apply_edits(skill: Skill, remove=(), fix=(), inputs=(), rename=(), description=None) -> Skill:
    """The flags of `mimic skills edit`, in a safe order (fix and rename before removing)."""
    for pair in rename:
        old, new = _split(pair, "--rename OLD=NEW")
        skill = rename_input(skill, old, new)
    for pair in fix:
        param, value = _split(pair, "--fix INPUT=VALUE")
        skill = fix_value(skill, param, value)
    for pair in inputs:
        index, name = _split(pair, "--input STEP=NAME")
        skill = make_input(skill, _int(index), name)
    if remove:
        skill = remove_steps(skill, [_int(i) for i in remove])
    if description:
        skill = describe(skill, description)
    return skill


def _split(pair: str, usage: str) -> tuple[str, str]:
    if "=" not in pair:
        raise EditError(f"expected {usage}, got '{pair}'")
    a, b = pair.split("=", 1)
    return a.strip(), b.strip()


def _int(text) -> int:
    try:
        return int(text)
    except ValueError:
        raise EditError(f"'{text}' is not a step number") from None


def changes(before: Skill, after: Skill) -> list[str]:
    """What changed, in words, for the confirmation."""
    out = []
    old = {s.index: s for s in before.steps}
    new = {s.index: s for s in after.steps}
    for i in sorted(set(old) - set(new)):
        out.append(f"removed step {i} ({old[i].action} {_label(old[i])})")
    for i in sorted(set(old) & set(new)):
        a, b = old[i], new[i]
        if a.parameter and not b.parameter:
            out.append(f"step {i} ({_label(b)}) always uses {b.value!r}")
        elif b.parameter and a.parameter != b.parameter:
            was = f"'{a.parameter}'" if a.parameter else f"constant {a.value!r}"
            out.append(f"step {i} ({_label(b)}) takes input '{b.parameter}' (was {was})")
    if before.description != after.description:
        out.append(f"description: {after.description!r}")
    return out


def _label(step) -> str:
    return step.element.label or step.element.text or step.element.selector


# ---------------------------------------------------------------- saving


def backup(skill: Skill, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{skill.name}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps(skill.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def save(skill: Skill, store, embedder=None, previous: Skill | None = None) -> None:
    """Save, with a fresh embedding when the model is available, else the previous one."""
    embedding = None
    if embedder is not None:
        try:
            embedding = embedder([skill.embedding_text()])[0]
        except Exception:  # noqa: BLE001 - Ollama off: keep searching with the old vector
            embedding = None
    if embedding is None and previous is not None:
        embedding = store.embedding(previous.name)
    store.save(skill, embedding)


def import_skill(path: Path, name: str | None = None) -> Skill:
    skill = Skill.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if name:
        if not _IDENT.match(name):
            raise EditError(f"'{name}' is not a valid skill name")
        skill = skill.model_copy(update={"name": name})
    return skill


# ---------------------------------------------------------------- interactive

HELP = """Commands:
  remove N             delete step N
  fix INPUT VALUE      INPUT becomes a constant: every task types VALUE
  input N NAME         step N takes its value from the input NAME
  rename OLD NEW       rename an input
  describe TEXT        new description
  show                 show the steps again
  save                 save and quit        quit   leave without saving"""


def interactive(skill: Skill, show, input_fn=input, print_fn=print) -> Skill | None:
    """Edit until 'save' (returns the new skill) or 'quit' (returns None)."""
    current = skill
    show(current)
    print_fn(HELP)
    while True:
        try:
            words = shlex.split(input_fn("edit> "))
        except ValueError as exc:
            print_fn(f"  ! {exc}")
            continue
        if not words:
            continue
        cmd, args = words[0].lower(), words[1:]
        before = current
        try:
            if cmd == "save":
                return current
            if cmd in ("quit", "exit"):
                return None
            if cmd == "show":
                show(current)
            elif cmd == "remove" and args:
                current = remove_steps(current, [_int(a) for a in args])
            elif cmd == "fix" and len(args) >= 2:
                current = fix_value(current, args[0], " ".join(args[1:]))
            elif cmd == "input" and len(args) == 2:
                current = make_input(current, _int(args[0]), args[1])
            elif cmd == "rename" and len(args) == 2:
                current = rename_input(current, args[0], args[1])
            elif cmd == "describe" and args:
                current = describe(current, " ".join(args))
            else:
                print_fn(HELP)
                continue
            if cmd != "show":
                print_fn("  ok: " + ("; ".join(changes(before, current)) or "no change"))
        except EditError as exc:
            print_fn(f"  ! {exc}")
