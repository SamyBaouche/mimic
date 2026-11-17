"""Skill library backed by Postgres + pgvector (semantic search over learned skills)."""

from __future__ import annotations

import json

from psycopg.types.json import Jsonb

from mimic.db import connect
from mimic.skills import Skill


def _vec(embedding: list[float] | None) -> str | None:
    return None if embedding is None else "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


class SkillStore:
    def save(self, skill: Skill, embedding: list[float] | None) -> int:
        """Insert or replace a skill (same name = re-learned) and its UI element memory."""
        with connect() as conn:
            conn.execute("DELETE FROM skills WHERE name = %s", (skill.name,))
            skill_id = conn.execute(
                """INSERT INTO skills (name, description, steps, embedding)
                   VALUES (%s, %s, %s, %s::vector) RETURNING id""",
                (
                    skill.name,
                    skill.description,
                    Jsonb(skill.model_dump(mode="json")),
                    _vec(embedding),
                ),
            ).fetchone()[0]
            for step in skill.steps:
                el = step.element
                conn.execute(
                    """INSERT INTO ui_elements (skill_id, label, selector, bbox)
                       VALUES (%s, %s, %s, %s)""",
                    (
                        skill_id,
                        el.label or el.text or el.selector,
                        el.selector,
                        Jsonb(el.bbox.model_dump()),
                    ),
                )
            conn.commit()
        return skill_id

    def get(self, name: str) -> Skill | None:
        with connect() as conn:
            row = conn.execute("SELECT steps FROM skills WHERE name = %s", (name,)).fetchone()
        return Skill.model_validate(row[0]) if row else None

    def search(self, embedding: list[float], k: int = 3) -> list[tuple[Skill, float]]:
        """Return the k skills closest to the query, with a cosine similarity score."""
        vec = _vec(embedding)
        with connect() as conn:
            rows = conn.execute(
                """SELECT steps, 1 - (embedding <=> %s::vector) AS score
                   FROM skills WHERE embedding IS NOT NULL
                   ORDER BY embedding <=> %s::vector LIMIT %s""",
                (vec, vec, k),
            ).fetchall()
        return [(Skill.model_validate(r[0]), float(r[1])) for r in rows]

    def list_names(self) -> list[str]:
        with connect() as conn:
            return [r[0] for r in conn.execute("SELECT name FROM skills ORDER BY id")]


def dumps(skill: Skill) -> str:
    return json.dumps(skill.model_dump(mode="json"), indent=2)
