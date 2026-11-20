"""Publishing the matcher on the Hugging Face Hub, and finding it again (no network)."""

import sys
import types

from mimic import matcher
from mimic.config import get_settings

META = {
    "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "parameters": 22_700_000,
    "epochs": 2,
    "train_pairs": 21000,
    "train_examples": 3000,
    "train_seconds": 95.0,
    "results": {
        "fine_tuned": {"synthetic_top1": 0.97, "sandbox_v2_top1": 1.0, "ms_per_decision_cpu": 7.7},
        "before_fine_tuning": {"synthetic_top1": 0.41, "sandbox_v2_top1": 0.5},
        "string_similarity": {"synthetic_top1": 0.38, "sandbox_v2_top1": 0.25},
    },
}


def test_model_card_states_the_measured_numbers():
    card = matcher.model_card(META, "someone/mimic-element-matcher")
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: cross-encoder/")
    assert "| Fine-tuned (this model) | 97.0% | 100.0% |" in card
    assert "7.7 ms per decision" in card
    assert "held out of training" in card


def test_local_model_wins_then_hub_download(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_download(repo, local_dir):
        calls.append(repo)
        (local_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        return str(local_dir)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_download)
    )
    settings = get_settings()
    monkeypatch.setattr(settings, "mimic_home", tmp_path / "home")
    assert matcher.find_model(download=False) is None
    got = matcher.find_model()
    assert got == settings.mimic_home / "models" / "element_matcher"
    assert calls == [settings.matcher_repo]
    assert matcher.find_model() == got and len(calls) == 1  # cached: downloaded once

    (tmp_path / "models" / "element_matcher").mkdir(parents=True)
    (tmp_path / "models" / "element_matcher" / "config.json").write_text("{}")
    assert matcher.find_model() == matcher.MODEL_DIR  # your own trained model first


def test_hub_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(get_settings(), "matcher_repo", "")
    monkeypatch.setattr(get_settings(), "mimic_home", tmp_path / "home")
    assert matcher.find_model() is None
