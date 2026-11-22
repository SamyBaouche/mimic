"""Skill library: save, load and search learned skills (semantic search by meaning)."""

from __future__ import annotations

import json

from mimic.skills import Skill
from mimic.storage import get_storage


class SkillStore:
    """Thin facade over the configured storage backend (SQLite by default, or Postgres)."""

    def save(self, skill: Skill, embedding: list[float] | None) -> int:
        """Insert or replace a skill (same name = re-learned) and its UI element memory."""
        return get_storage().save_skill(skill, embedding)

    def get(self, name: str) -> Skill | None:
        return get_storage().get_skill(name)

    def search(self, embedding: list[float], k: int = 3) -> list[tuple[Skill, float]]:
        """Return the k skills closest to the query, with a cosine similarity score."""
        return get_storage().search_skills(embedding, k)

    def list_names(self) -> list[str]:
        return get_storage().list_skills()

    def embedding(self, name: str) -> list[float] | None:
        return get_storage().get_embedding(name)

    def delete(self, name: str) -> bool:
        return get_storage().delete_skill(name)


def dumps(skill: Skill) -> str:
    return json.dumps(skill.model_dump(mode="json"), indent=2)
