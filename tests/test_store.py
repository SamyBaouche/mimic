"""Skill library on both storage backends: SQLite (always) and Postgres (if reachable)."""

import math
from pathlib import Path

import pytest

from mimic.config import get_settings
from mimic.models import Recording
from mimic.skills import draft_skill
from mimic.storage import get_storage
from mimic.store import SkillStore

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"
DIM = get_settings().embedding_dim


def _unit(axis: int, noise_axis: int | None = None) -> list[float]:
    """A unit vector pointing mostly along one axis: easy to reason about similarity."""
    v = [0.0] * DIM
    v[axis] = 1.0
    if noise_axis is not None:
        v[noise_axis] = 0.3
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@pytest.fixture(scope="module")
def store(test_database):
    get_storage().reset()
    return SkillStore()


def _skill(name: str):
    rec = Recording.model_validate_json(FIXTURE.read_text())
    rec.name = name
    return draft_skill(rec)


def test_save_and_get_round_trip(store):
    skill = _skill("enter_invoice")
    store.save(skill, _unit(0))
    loaded = store.get("enter_invoice")
    assert loaded == skill
    assert get_storage().stats()["skills"] >= 1


def test_relearning_replaces_the_skill(store):
    store.save(_skill("enter_invoice"), _unit(0))
    store.save(_skill("enter_invoice"), _unit(0))
    assert store.list_names().count("enter_invoice") == 1


def test_search_returns_closest_skill_first(store):
    store.save(_skill("enter_invoice"), _unit(0))
    store.save(_skill("create_customer"), _unit(1))
    results = store.search(_unit(0, noise_axis=1), k=2)
    assert [s.name for s, _ in results] == ["enter_invoice", "create_customer"]
    assert results[0][1] > results[1][1]


def test_skill_without_embedding_is_stored_but_not_searchable(store):
    store.save(_skill("no_vector"), None)
    assert store.get("no_vector") is not None
    assert "no_vector" not in [s.name for s, _ in store.search(_unit(0), k=10)]


def test_delete_skill(store):
    store.save(_skill("to_delete"), _unit(2))
    assert store.delete("to_delete") is True
    assert store.get("to_delete") is None
    assert store.delete("to_delete") is False


def test_step_logs_round_trip(store):
    rows = [
        {
            "run_id": "r1",
            "step_index": 0,
            "action": "fill",
            "match_score": 1.0,
            "layout_change": 0.0,
            "latency_ms": 12,
            "success": True,
            "strategy": "selector",
            "features": {"a": 1.0},
            "risk": 0.1,
            "mitigated": False,
            "error": None,
        },
        {
            "run_id": "r1",
            "step_index": 1,
            "action": "click",
            "match_score": None,
            "layout_change": 1.0,
            "latency_ms": 900,
            "success": False,
            "strategy": "failed",
            "features": {"a": 2.0},
            "risk": 0.9,
            "mitigated": True,
            "error": "timeout",
        },
    ]
    get_storage().log_steps(rows)
    samples = [s for s in get_storage().step_samples() if s["run_id"] == "r1"]
    assert samples == [{"run_id": "r1", "features": {"a": 1.0}, "failed": False}]
