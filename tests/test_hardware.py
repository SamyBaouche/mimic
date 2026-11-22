"""Model profiles chosen from the hardware, and `mimic doctor --pull` (no Ollama needed)."""

import contextlib
import json

import pytest

from mimic import doctor, hardware
from mimic.config import Settings
from mimic.hardware import Hardware, recommend


@pytest.mark.parametrize(
    "hw, profile",
    [
        (Hardware(ram_gb=32, gpu="RTX 4070", vram_gb=12), "standard"),  # big GPU
        (Hardware(ram_gb=8, gpu="RTX 4070", vram_gb=12), "standard"),  # the GPU carries it
        (Hardware(ram_gb=16), "standard"),  # enough RAM, no GPU
        (Hardware(ram_gb=8), "light"),  # small laptop
        (Hardware(ram_gb=8, gpu="GTX 1650", vram_gb=4), "light"),  # small GPU
        (Hardware(ram_gb=8, gpu="Apple Silicon", vram_gb=8, unified_memory=True), "light"),
        (Hardware(ram_gb=24, gpu="Apple Silicon", vram_gb=24, unified_memory=True), "standard"),
    ],
)
def test_recommend(hw, profile):
    assert recommend(hw)[0] == profile


def _settings(monkeypatch, hw, **env):
    monkeypatch.setattr(hardware, "detect", lambda: hw)
    for key in ("MIMIC_PROFILE", "VISION_MODEL", "PLANNER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_auto_picks_models_for_this_machine(monkeypatch):
    small = _settings(monkeypatch, Hardware(ram_gb=8))
    assert (small.profile_used, small.vision_model, small.planner_model) == (
        "light",
        "qwen2.5vl:3b",
        "qwen2.5:3b",
    )
    big = _settings(monkeypatch, Hardware(ram_gb=32, gpu="RTX 4070", vram_gb=12))
    assert (big.vision_model, big.planner_model) == ("qwen2.5vl:7b", "qwen2.5:7b")


def test_profile_and_single_model_overrides(monkeypatch):
    forced = _settings(monkeypatch, Hardware(ram_gb=64), MIMIC_PROFILE="light")
    assert forced.vision_model == "qwen2.5vl:3b" and forced.profile_reason == "MIMIC_PROFILE"
    one = _settings(monkeypatch, Hardware(ram_gb=8), VISION_MODEL="llava:13b")
    assert (one.vision_model, one.planner_model) == ("llava:13b", "qwen2.5:3b")
    both = _settings(monkeypatch, Hardware(ram_gb=8), VISION_MODEL="a", PLANNER_MODEL="b")
    assert both.profile_used == "custom"
    with pytest.raises(ValueError):
        _settings(monkeypatch, Hardware(ram_gb=8), MIMIC_PROFILE="huge")


def test_doctor_warns_when_models_are_too_big(monkeypatch):
    monkeypatch.setattr(hardware, "detect", lambda: Hardware(ram_gb=8))
    s = doctor.get_settings()
    monkeypatch.setattr(s, "profile_used", "custom")  # VISION_MODEL / PLANNER_MODEL set by hand
    monkeypatch.setattr(s, "vision_model", "qwen2.5vl:7b")
    check = doctor.check_hardware()
    assert not check.ok and not check.required
    assert "MIMIC_PROFILE=light" in check.fix


def test_doctor_says_when_env_overrides_the_profile(monkeypatch):
    monkeypatch.setattr(hardware, "detect", lambda: Hardware(ram_gb=32))
    s = doctor.get_settings()
    monkeypatch.setattr(s, "profile_used", "light")
    monkeypatch.setattr(s, "vision_model", "qwen2.5vl:7b")  # left in .env
    monkeypatch.setattr(s, "planner_model", "qwen2.5:3b")
    check = doctor.check_hardware()
    assert not check.ok
    assert "VISION_MODEL=qwen2.5vl:7b" in check.detail and ".env" in check.fix


def test_pull_streams_progress(monkeypatch, capsys):
    lines = [
        {"status": "pulling manifest"},
        {"status": "downloading", "total": 3_000_000_000, "completed": 0},
        {"status": "downloading", "total": 3_000_000_000, "completed": 1_500_000_000},
        {"status": "downloading", "total": 3_000_000_000, "completed": 3_000_000_000},
        {"status": "success"},
    ]

    class Response:
        def iter_lines(self):
            return (json.dumps(x) for x in lines)

    calls = []

    @contextlib.contextmanager
    def fake_stream(method, url, json, timeout):
        calls.append((method, url, json["model"]))
        yield Response()

    monkeypatch.setattr(doctor.httpx, "stream", fake_stream)
    doctor.pull_models(["qwen2.5:3b"], "http://ollama")
    out = capsys.readouterr().out
    assert calls == [("POST", "http://ollama/api/pull", "qwen2.5:3b")]
    assert " 50%" in out and "100%" in out and "qwen2.5:3b ready" in out
