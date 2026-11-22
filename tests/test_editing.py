"""Change a learned skill without recording it again; export / import; undo from a backup."""

from pathlib import Path

import httpx
import pytest

from mimic import editing
from mimic.editing import EditError
from mimic.models import Recording
from mimic.skills import draft_skill

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"


@pytest.fixture
def skill():
    return draft_skill(Recording.model_validate_json(FIXTURE.read_text()))


def test_fix_a_value(skill):
    edited = editing.fix_value(skill, "category", "Software")
    assert "category" not in edited.parameters
    actions = edited.render(
        {"vendor": "A", "invoice_number": "1", "date": "2026-01-01", "total": "5"}
    )
    assert [a["value"] for a in actions if a["action"] == "select"] == ["Software"]


def test_rename_and_make_input(skill):
    edited = editing.rename_input(skill, "vendor", "supplier")
    assert edited.parameters[0] == "supplier"
    fixed = editing.fix_value(edited, "category", "Software")
    back = editing.make_input(fixed, 4, "expense_type")
    assert "expense_type" in back.parameters
    assert editing.changes(fixed, back) == [
        "step 4 (Category) takes input 'expense_type' (was constant 'Software')"
    ]


def test_remove_steps_keeps_the_other_indices(skill):
    edited = editing.remove_steps(skill, [3])
    assert [s.index for s in edited.steps] == [0, 1, 2, 4, 5]  # memory of healed steps still fits
    assert "total" not in edited.parameters


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda s: editing.fix_value(s, "nope", "x"), "no input 'nope'"),
        (lambda s: editing.rename_input(s, "vendor", "Bad Name"), "not a valid input name"),
        (lambda s: editing.rename_input(s, "vendor", "total"), "already called 'total'"),
        (lambda s: editing.make_input(s, 5, "button"), "is a click"),
        (lambda s: editing.remove_steps(s, [9]), "no step 9"),
        (lambda s: editing.remove_steps(s, [0, 1, 2, 3, 4, 5]), "at least one step"),
        (lambda s: editing.describe(s, "  "), "cannot be empty"),
    ],
)
def test_edits_are_checked(skill, edit, message):
    with pytest.raises(EditError, match=message):
        edit(skill)


def test_secrets_are_never_fixed_into_a_skill(skill):
    secret = skill.model_copy(
        update={
            "steps": [
                s.model_copy(update={"secret": True}) if s.index == 0 else s for s in skill.steps
            ]
        }
    )
    with pytest.raises(EditError, match="never stored"):
        editing.fix_value(secret, "vendor", "hunter2")


def test_flags_in_a_safe_order(skill):
    edited = editing.apply_edits(
        skill,
        remove=["3"],
        fix=["category=Software"],
        rename=["vendor=supplier"],
        description="Enter a supplier invoice in LedgerLite",
    )
    assert edited.parameters == ["supplier", "invoice_number", "date"]
    with pytest.raises(EditError, match="--fix INPUT=VALUE"):
        editing.apply_edits(skill, fix=["category"])


def test_interactive_session(skill):
    answers = iter(
        [
            "fix category Office supplies",
            "rename vendor supplier",
            "frobnicate",
            "remove 99",
            "save",
        ]
    )
    out = []
    edited = editing.interactive(skill, lambda s: None, lambda _: next(answers), out.append)
    assert edited.parameters == ["supplier", "invoice_number", "date", "total"]
    text = "\n".join(out)
    assert "always uses 'Office supplies'" in text and "no step 99" in text and "Commands:" in text
    quit_answers = iter(["fix category X", "quit"])
    assert (
        editing.interactive(skill, lambda s: None, lambda _: next(quit_answers), out.append) is None
    )


def test_cli_edit_backup_undo_export_import(skill, tmp_path, monkeypatch, capsys):
    from mimic.cli import main
    from mimic.config import get_settings
    from mimic.store import SkillStore

    monkeypatch.setattr(get_settings(), "mimic_home", tmp_path / "home")
    monkeypatch.chdir(tmp_path)

    def ollama_is_off(self, texts):
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr("mimic.llm.OllamaClient.embed", ollama_is_off)
    store = SkillStore()
    vector = [1.0] + [0.0] * 767
    store.save(skill, vector)

    assert main(["skills", "edit", "enter_invoice", "--fix", "category=Software", "--dry-run"]) == 0
    assert "category" in store.get("enter_invoice").parameters  # dry run saved nothing

    assert main(["skills", "edit", "enter_invoice", "--fix", "category=Software"]) == 0
    assert "category" not in store.get("enter_invoice").parameters
    assert store.embedding("enter_invoice") == vector  # no Ollama here: search keeps working
    [backup] = (tmp_path / "home" / "backups").glob("enter_invoice-*.json")
    assert "undo with: mimic skills import" in capsys.readouterr().out

    assert main(["skills", "import", str(backup)]) == 0  # undo
    assert "category" in store.get("enter_invoice").parameters

    assert main(["skills", "export", "enter_invoice", "--out", "shared.json"]) == 0
    assert main(["skills", "import", "shared.json", "--name", "enter_invoice_copy"]) == 0
    assert store.get("enter_invoice_copy").steps == store.get("enter_invoice").steps
    assert main(["skills", "edit", "enter_invoice", "--fix", "nope=1"]) == 1


def test_an_edited_skill_still_replays(skill, sandbox_server):
    from playwright.sync_api import sync_playwright

    from mimic.agent import ReplayAgent
    from mimic.recorder import launch_browser

    edited = editing.apply_edits(skill, fix=["category=Utilities"], rename=["vendor=supplier"])
    httpx.delete(sandbox_server + "api/entries")
    with sync_playwright() as p:
        try:
            browser = launch_browser(p, headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")
        page = browser.new_page()
        page.goto(sandbox_server)
        actions = edited.render(
            {
                "supplier": "Hydro Services",
                "invoice_number": "INV-EDIT-1",
                "date": "2026-11-21",
                "total": "88.10",
            }
        )
        result = ReplayAgent(page).run(actions)
        browser.close()
    assert result.success, result.error
    [entry] = httpx.get(sandbox_server + "api/entries").json()
    assert (entry["vendor"], entry["category"]) == ("Hydro Services", "Utilities")
