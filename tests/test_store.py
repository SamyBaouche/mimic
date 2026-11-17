"""Integration tests against a real Postgres + pgvector (skipped if unavailable)."""

import math
from pathlib import Path

import psycopg
import pytest

from mimic.config import get_settings
from mimic.db import connect, init_db
from mimic.models import Recording
from mimic.skills import draft_skill
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
def store():
    """Use a separate 'mimic_test' database so tests never touch your learned skills."""
    settings = get_settings()
    original = settings.database_url
    base, _, _ = original.rpartition("/")
    try:
        with psycopg.connect(original, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'mimic_test'")
            if not exists.fetchone():
                conn.execute("CREATE DATABASE mimic_test")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not available: {exc}")

    settings.database_url = f"{base}/mimic_test"
    init_db()
    with connect() as conn:
        conn.execute("TRUNCATE skills RESTART IDENTITY CASCADE")
        conn.commit()
    yield SkillStore()
    settings.database_url = original


def _skill(name: str):
    rec = Recording.model_validate_json(FIXTURE.read_text())
    rec.name = name
    return draft_skill(rec)


def test_save_and_get_round_trip(store):
    skill = _skill("enter_invoice")
    store.save(skill, _unit(0))
    loaded = store.get("enter_invoice")
    assert loaded == skill
    with connect() as conn:
        n = conn.execute("SELECT count(*) FROM ui_elements").fetchone()[0]
    assert n == len(skill.steps)


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
