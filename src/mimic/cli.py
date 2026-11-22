"""The `mimic` command. Each subcommand hands off to the module that does the work."""

from __future__ import annotations

import sys
from pathlib import Path

from mimic import __version__

HELP = f"""Mimic {__version__}: show a web task once, the agent repeats it.

Get started
  mimic doctor                     Check browser, local models and storage
  mimic sandbox                    Start the demo accounting app on http://localhost:8001

Teach and run
  mimic record NAME [--url URL]    Do the task once in a browser; Mimic records it
  mimic learn RECORDING.json       Turn a recording into a reusable skill
  mimic run NAME [options]         Repeat the skill on a data file or on documents
  mimic serve                      Live dashboard: watch the agent, approve each plan

Workflows (several skills, several apps)
  mimic workflow check FILE.yml    Validate it, show the rules its conditions became
  mimic workflow run FILE.yml      For every email / row: read, decide, run each skill

Skills
  mimic skills                     List learned skills
  mimic skills show NAME           Steps and parameters of a skill
  mimic skills search "TEXT"       Find a skill by what it does
  mimic skills delete NAME         Forget a skill
  mimic skills edit NAME           Change a skill without recording it again
  mimic skills export NAME         Save a skill to a .json file (share it, keep it)
  mimic skills import FILE.json    Add a skill from a file (or undo an edit)

Documents and models
  mimic read FILE [--skill NAME]   Extract the skill's fields from a PDF or photo
  mimic extract eval               Measure extraction accuracy on the demo documents
  mimic risk collect|train|evaluate     Failure-risk model (XGBoost)
  mimic matcher train|evaluate          Fine-tuned UI element matcher

Run `mimic COMMAND --help` for the options of a command."""


# Optional features live in extras so the base install stays small.
EXTRAS = {
    "xgboost": "ml",
    "sklearn": "ml",
    "torch": "matcher",
    "transformers": "matcher",
    "psycopg": "postgres",
}


def _positional(rest: list[str], flag: str) -> list[str]:
    """Allow `mimic run enter_invoice` as well as `mimic run --skill enter_invoice`."""
    if rest and not rest[0].startswith("-"):
        return [flag, rest[0], *rest[1:]]
    return rest


def cmd_sandbox(rest: list[str]) -> int:
    import argparse

    import uvicorn

    from mimic.sandbox import app as sandbox_app
    from mimic.sandbox import generate_documents, generate_invoices, generate_mail

    parser = argparse.ArgumentParser(prog="mimic sandbox", description="Demo accounting app")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("sandbox/data"),
        help="Where to write the demo invoices and PDFs",
    )
    args = parser.parse_args(rest)

    if not (args.data / "invoices.json").exists():
        generate_invoices.main(args.data / "invoices.json")
    if not (args.data / "documents" / "truth.json").exists():
        generate_documents.main(args.data / "documents")
    if not (args.data / "mail" / "truth.json").exists():
        generate_mail.main(args.data / "mail")
    sandbox_app.set_mail_dir(args.data / "mail")
    print(f"LedgerLite demo app on http://localhost:{args.port}  (Ctrl+C to stop)")
    print(f"  /?layout=v2 for the redesigned interface, demo data in {args.data}")
    print("  WorkSuite: /mail (supplier emails), /approvals/new, /expenses/new")
    uvicorn.run(sandbox_app.app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


def cmd_skills(rest: list[str]) -> int:
    from mimic import learn

    if not rest or rest[0] == "list":
        from mimic.store import SkillStore

        names = SkillStore().list_names()
        print("\n".join(names) if names else "No skills yet. Record one: mimic record NAME")
        return 0
    action, args = rest[0], rest[1:]
    if action == "show" and args:
        from mimic.store import SkillStore

        skill = SkillStore().get(args[0])
        if skill is None:
            print(f"No skill named '{args[0]}'. See them with: mimic skills")
            return 1
        learn.print_skill(skill)
        return 0
    if action == "delete" and args:
        from mimic.store import SkillStore

        if SkillStore().delete(args[0]):
            print(f"Deleted '{args[0]}'.")
            return 0
        print(f"No skill named '{args[0]}'.")
        return 1
    if action == "edit" and args:
        return _edit(args)
    if action == "export" and args:
        return _export(args)
    if action == "import" and args:
        return _import(args)
    if action == "search" and args:
        learn.main(["search", " ".join(args)])
        return 0
    print(
        "Usage: mimic skills [list | show NAME | search TEXT | delete NAME | edit NAME"
        " | export NAME | import FILE]"
    )
    return 2


def _edit(args: list[str]) -> int:
    import argparse

    from mimic import editing, learn
    from mimic.config import get_settings
    from mimic.store import SkillStore

    parser = argparse.ArgumentParser(prog="mimic skills edit")
    parser.add_argument("name")
    parser.add_argument("--remove", action="append", default=[], metavar="STEP")
    parser.add_argument("--fix", action="append", default=[], metavar="INPUT=VALUE")
    parser.add_argument("--input", action="append", default=[], metavar="STEP=NAME")
    parser.add_argument("--rename", action="append", default=[], metavar="OLD=NEW")
    parser.add_argument("--describe", default=None, metavar="TEXT")
    parser.add_argument("--dry-run", action="store_true", help="Show the changes, save nothing")
    a = parser.parse_args(args)

    store = SkillStore()
    skill = store.get(a.name)
    if skill is None:
        print(f"No skill named '{a.name}'. See them with: mimic skills")
        return 1
    try:
        if any([a.remove, a.fix, a.input, a.rename, a.describe]):
            edited = editing.apply_edits(skill, a.remove, a.fix, a.input, a.rename, a.describe)
        else:
            edited = editing.interactive(skill, learn.print_skill)
            if edited is None:
                print("Left without saving.")
                return 0
    except editing.EditError as exc:
        print(f"Cannot edit '{a.name}': {exc}")
        return 1
    done = editing.changes(skill, edited)
    if not done:
        print("Nothing changed.")
        return 0
    print("\n".join(f"  - {c}" for c in done))
    if a.dry_run:
        print("Dry run: nothing saved.")
        return 0
    saved_to = editing.backup(skill, get_settings().mimic_home / "backups")
    from mimic.llm import OllamaClient

    editing.save(edited, store, OllamaClient().embed, previous=skill)
    print(f"Saved. Previous version: {saved_to}")
    print(f'  (undo with: mimic skills import "{saved_to}")')
    return 0


def _export(args: list[str]) -> int:
    import argparse
    import json

    from mimic.store import SkillStore

    parser = argparse.ArgumentParser(prog="mimic skills export")
    parser.add_argument("name")
    parser.add_argument("--out", type=Path, default=None)
    a = parser.parse_args(args)
    skill = SkillStore().get(a.name)
    if skill is None:
        print(f"No skill named '{a.name}'.")
        return 1
    out = a.out or Path(f"{a.name}.json")
    out.write_text(json.dumps(skill.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"Exported '{a.name}' to {out} (secrets are never part of a skill).")
    return 0


def _import(args: list[str]) -> int:
    import argparse

    from mimic import editing
    from mimic.llm import OllamaClient
    from mimic.store import SkillStore

    parser = argparse.ArgumentParser(prog="mimic skills import")
    parser.add_argument("file", type=Path)
    parser.add_argument("--name", default=None, help="Save it under another name")
    a = parser.parse_args(args)
    try:
        skill = editing.import_skill(a.file, a.name)
    except (OSError, ValueError) as exc:
        print(f"Cannot import {a.file}: {exc}")
        return 1
    store = SkillStore()
    existing = store.get(skill.name)
    editing.save(skill, store, OllamaClient().embed, previous=existing)
    verb = "Replaced" if existing else "Imported"
    print(
        f"{verb} '{skill.name}' ({len(skill.steps)} steps, inputs: {', '.join(skill.parameters)})."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"mimic {__version__}")
        return 0

    cmd, rest = argv[0], argv[1:]
    sys.argv[0] = f"mimic {cmd}"  # so each --help shows the right command name

    if cmd == "doctor":
        from mimic import doctor

        return doctor.main(rest)
    if cmd == "sandbox":
        return cmd_sandbox(rest)
    if cmd == "skills":
        return cmd_skills(rest)
    if cmd == "init":
        from mimic.db import init_db

        init_db()
        return 0

    handlers = {
        "record": ("mimic.recorder", lambda r: _positional(r, "--name")),
        "learn": ("mimic.learn", lambda r: ["learn", *r]),
        "run": ("mimic.run", lambda r: _positional(r, "--skill")),
        "serve": ("mimic.server", lambda r: r),
        "read": ("mimic.extract", lambda r: ["read", *r]),
        "extract": ("mimic.extract", lambda r: r),
        "risk": ("mimic.riskml", lambda r: r),
        "matcher": ("mimic.matcher", lambda r: r),
        "workflow": ("mimic.workflow", lambda r: r),
    }
    if cmd not in handlers:
        print(f"Unknown command '{cmd}'. Run `mimic --help` to see the commands.")
        return 2
    import importlib

    module, to_argv = handlers[cmd]
    try:
        result = importlib.import_module(module).main(to_argv(rest))
    except ModuleNotFoundError as exc:
        extra = EXTRAS.get((exc.name or "").split(".")[0])
        if extra is None:
            raise
        print(f'This needs an optional part of Mimic: pip install "mimic-agent[{extra}]"')
        return 1
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
