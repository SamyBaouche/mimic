"""Passwords typed during a demonstration are never recorded, stored or shown."""

import httpx
import pytest
from playwright.sync_api import sync_playwright

from mimic.agent import ReplayAgent
from mimic.approval import SECRET_MASK, format_plan, plan_rows
from mimic.recorder import Recorder, launch_browser
from mimic.run import MissingSecret, check_secrets
from mimic.skills import apply_refinement, draft_skill

PASSWORD = "ledger-demo-2026"


@pytest.fixture(scope="module")
def recording(sandbox_server, tmp_path_factory):
    out = tmp_path_factory.mktemp("rec")
    recorder = Recorder("sign_in", sandbox_server + "login", out)
    with sync_playwright() as p:
        try:
            browser = launch_browser(p, headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")
        context = browser.new_context()
        recorder.attach(context)
        page = context.new_page()
        page.goto(sandbox_server + "login")
        page.fill("#username", "demo")
        page.fill("#password", PASSWORD)
        page.click("#sign-in")
        page.wait_for_url("**/?signed_in=1")
        page.wait_for_timeout(200)
        browser.close()
    recorder.save()
    return recorder, out


def test_password_value_never_reaches_the_recording(recording):
    recorder, out = recording
    kinds = [(e.kind, e.secret, e.value) for e in recorder.recording.events]
    assert kinds[:2] == [("fill", False, "demo"), ("fill", True, None)]
    saved = (out / "recording.json").read_text()
    assert PASSWORD not in saved
    # No screenshot is taken while the secret field is being filled.
    assert recorder.recording.events[1].screenshot is None


def test_skill_keeps_a_secret_parameter_without_its_value(recording):
    recorder, _ = recording
    skill = draft_skill(recorder.recording)
    assert skill.secrets == ["password"]
    assert PASSWORD not in skill.model_dump_json()

    # The LLM only sees (and can only rename) the non-secret fields.
    refined = apply_refinement(
        skill, {"description": "Signs in.", "steps": [{"index": 0, "parameter": "user_name"}]}
    )
    assert refined.parameters == ["user_name", "password"]
    with pytest.raises(ValueError):
        apply_refinement(
            skill, {"description": "x", "steps": [{"index": 0, "parameter": "password"}]}
        )


def test_secret_comes_from_the_environment_and_is_masked(recording, monkeypatch):
    skill = draft_skill(recording[0].recording)
    monkeypatch.delenv("MIMIC_SECRET_PASSWORD", raising=False)
    with pytest.raises(MissingSecret, match="MIMIC_SECRET_PASSWORD"):
        check_secrets(skill, {"username": "demo"})

    monkeypatch.setenv("MIMIC_SECRET_PASSWORD", PASSWORD)
    check_secrets(skill, {"username": "demo"})
    actions = skill.render({"username": "demo"})
    assert actions[1]["value"] == PASSWORD
    assert PASSWORD not in format_plan(1, 1, actions)
    assert SECRET_MASK in format_plan(1, 1, actions)
    assert plan_rows(actions)[1]["value"] == SECRET_MASK


def test_replay_signs_in_with_the_secret(recording, sandbox_server, monkeypatch):
    skill = draft_skill(recording[0].recording)
    monkeypatch.setenv("MIMIC_SECRET_PASSWORD", PASSWORD)
    with sync_playwright() as p:
        browser = launch_browser(p, headless=True)
        page = browser.new_page()
        page.goto(sandbox_server + "login")
        result = ReplayAgent(page).run(skill.render({"username": "demo"}))
        browser.close()
    assert result.success
    assert httpx.get(sandbox_server + "api/logins").json()[-1] == "demo"
