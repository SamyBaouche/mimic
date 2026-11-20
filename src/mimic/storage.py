"""Where Mimic keeps what it learns: skills, healed elements and step telemetry.

Two backends behind one small interface:

- SQLite (default): a single file, ~/.mimic/mimic.db. Nothing to install. Vector search is a
  numpy cosine over the stored embeddings, which is plenty for hundreds of skills.
- Postgres + pgvector: set DATABASE_URL=postgresql://... (team setups, large memories).
  Needs the optional extra:  pip install "mimic-agent[postgres]"
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from mimic.config import get_settings
from mimic.skills import Skill

STEP_COLUMNS = (
    "run_id",
    "step_index",
    "action",
    "match_score",
    "layout_change",
    "latency_ms",
    "success",
    "strategy",
    "features",
    "risk",
    "mitigated",
    "error",
)


def _cosine(query: list[float], rows: list[list[float]]) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(rows, dtype=np.float32)
    denom = np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0)
    return (m @ q) / np.where(denom == 0, 1.0, denom)


class Storage:
    """Interface shared by both backends."""

    kind = "base"
    location = ""

    def init(self) -> None: ...
    def save_skill(self, skill: Skill, embedding: list[float] | None) -> int: ...
    def get_skill(self, name: str) -> Skill | None: ...
    def search_skills(self, embedding: list[float], k: int = 3) -> list[tuple[Skill, float]]: ...
    def list_skills(self) -> list[str]: ...
    def delete_skill(self, name: str) -> bool: ...
    def heal_lookup(
        self, skill: str, step: int, path: str, embedding: list[float] | None
    ) -> list[str]: ...
    def heal_remember(
        self, skill: str, step: int, path: str, selector: str, strategy: str, embedding
    ) -> None: ...
    def log_steps(self, rows: list[dict]) -> None: ...
    def step_samples(self) -> list[dict]: ...
    def stats(self) -> dict: ...
    def reset(self) -> None:
        """Delete everything (tests only)."""


# ---------------------------------------------------------------- SQLite

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    steps       TEXT NOT NULL,
    embedding   TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ui_elements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_id    INTEGER REFERENCES skills(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    selector    TEXT,
    bbox        TEXT
);
CREATE TABLE IF NOT EXISTS healed_elements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    skill_name  TEXT NOT NULL,
    step_index  INTEGER NOT NULL,
    page_path   TEXT NOT NULL,
    selector    TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    embedding   TEXT,
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS healed_page ON healed_elements(page_path);
CREATE TABLE IF NOT EXISTS step_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    step_index    INTEGER NOT NULL,
    action        TEXT NOT NULL,
    match_score   REAL,
    layout_change REAL,
    latency_ms    INTEGER,
    success       INTEGER,
    strategy      TEXT,
    features      TEXT,
    risk          REAL,
    mitigated     INTEGER,
    error         TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class SQLiteStorage(Storage):
    kind = "sqlite"

    def __init__(self, path: Path):
        self.path = Path(path).expanduser()
        self.location = str(self.path)
        self._ready = False
        self._lock = threading.Lock()

    def _conn(self) -> sqlite3.Connection:
        if not self._ready:
            self.init()
        conn = sqlite3.connect(self.path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path, timeout=10) as conn:
                conn.execute("PRAGMA journal_mode = WAL")  # dashboard thread + CLI at once
                conn.executescript(SQLITE_SCHEMA)
            self._ready = True

    def save_skill(self, skill, embedding):
        with self._conn() as conn:
            conn.execute("DELETE FROM skills WHERE name = ?", (skill.name,))
            cur = conn.execute(
                "INSERT INTO skills (name, description, steps, embedding) VALUES (?, ?, ?, ?)",
                (
                    skill.name,
                    skill.description,
                    json.dumps(skill.model_dump(mode="json")),
                    json.dumps(embedding) if embedding is not None else None,
                ),
            )
            skill_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO ui_elements (skill_id, label, selector, bbox) VALUES (?, ?, ?, ?)",
                [
                    (
                        skill_id,
                        s.element.label or s.element.text or s.element.selector,
                        s.element.selector,
                        json.dumps(s.element.bbox.model_dump()),
                    )
                    for s in skill.steps
                ],
            )
        return skill_id

    def get_skill(self, name):
        with self._conn() as conn:
            row = conn.execute("SELECT steps FROM skills WHERE name = ?", (name,)).fetchone()
        return Skill.model_validate_json(row[0]) if row else None

    def search_skills(self, embedding, k=3):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT steps, embedding FROM skills WHERE embedding IS NOT NULL"
            ).fetchall()
        vectors = [json.loads(r[1]) for r in rows]
        rows = [(r, v) for r, v in zip(rows, vectors, strict=True) if len(v) == len(embedding)]
        if not rows:
            return []
        scores = _cosine(embedding, [v for _, v in rows])
        order = np.argsort(-scores)[:k]
        return [(Skill.model_validate_json(rows[i][0][0]), float(scores[i])) for i in order]

    def list_skills(self):
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT name FROM skills ORDER BY id")]

    def delete_skill(self, name):
        with self._conn() as conn:
            return conn.execute("DELETE FROM skills WHERE name = ?", (name,)).rowcount > 0

    def heal_lookup(self, skill, step, path, embedding):
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT selector, skill_name = ? AND step_index = ?, embedding
                   FROM healed_elements WHERE page_path = ? ORDER BY id DESC""",
                (skill, step, path),
            ).fetchall()
        if not rows:
            return []
        if embedding is None:
            same = [r[0] for r in rows if r[1]]
            return list(dict.fromkeys(same))[:3]

        # Same step first, then the most similar element healed on this page.
        def key(r):
            vec = json.loads(r[2]) if r[2] else None
            sim = float(_cosine(embedding, [vec])[0]) if vec and len(vec) == len(embedding) else -1
            return (-int(r[1]), -sim)

        return list(dict.fromkeys(r[0] for r in sorted(rows, key=key)))[:3]

    def heal_remember(self, skill, step, path, selector, strategy, embedding):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO healed_elements
                   (skill_name, step_index, page_path, selector, strategy, embedding)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    skill,
                    step,
                    path,
                    selector,
                    strategy,
                    json.dumps(embedding) if embedding else None,
                ),
            )

    def log_steps(self, rows):
        values = [
            tuple(
                json.dumps(r[c]) if c == "features" and r[c] is not None else r[c]
                for c in STEP_COLUMNS
            )
            for r in rows
        ]
        with self._conn() as conn:
            conn.executemany(
                f"INSERT INTO step_logs ({', '.join(STEP_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(STEP_COLUMNS))})",
                values,
            )

    def step_samples(self):
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT run_id, features, success FROM step_logs
                   WHERE features IS NOT NULL AND NOT COALESCE(mitigated, 0)"""
            ).fetchall()
        return [{"run_id": r[0], "features": json.loads(r[1]), "failed": not r[2]} for r in rows]

    def stats(self):
        with self._conn() as conn:
            return {
                t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("skills", "healed_elements", "step_logs")
            }

    def reset(self):
        with self._conn() as conn:
            for table in ("ui_elements", "skills", "healed_elements", "step_logs"):
                conn.execute(f"DELETE FROM {table}")


# ---------------------------------------------------------------- Postgres

POSTGRES_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS skills (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    steps       JSONB NOT NULL DEFAULT '[]',
    embedding   vector({dim}),
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ui_elements (
    id          SERIAL PRIMARY KEY,
    skill_id    INT REFERENCES skills(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    selector    TEXT,
    bbox        JSONB,
    embedding   vector({dim}),
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS healed_elements (
    id          SERIAL PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    step_index  INT NOT NULL,
    page_path   TEXT NOT NULL,
    selector    TEXT NOT NULL,
    strategy    TEXT NOT NULL,
    embedding   vector({dim}),
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS step_logs (
    id            SERIAL PRIMARY KEY,
    run_id        TEXT NOT NULL,
    skill_id      INT REFERENCES skills(id) ON DELETE SET NULL,
    step_index    INT NOT NULL,
    action        TEXT NOT NULL,
    match_score   REAL,
    layout_change REAL,
    latency_ms    INT,
    success       BOOLEAN,
    created_at    TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE step_logs ADD COLUMN IF NOT EXISTS strategy TEXT;
ALTER TABLE step_logs ADD COLUMN IF NOT EXISTS features JSONB;
ALTER TABLE step_logs ADD COLUMN IF NOT EXISTS risk REAL;
ALTER TABLE step_logs ADD COLUMN IF NOT EXISTS mitigated BOOLEAN;
ALTER TABLE step_logs ADD COLUMN IF NOT EXISTS error TEXT;
"""


def _vec(embedding: list[float] | None) -> str | None:
    return None if embedding is None else "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


class PostgresStorage(Storage):
    kind = "postgres"

    def __init__(self, url: str):
        self.url = url
        self.location = url.split("@")[-1]  # never print the password
        self.dim = get_settings().embedding_dim
        self._ready = False

    def _conn(self):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                'Postgres support is not installed: pip install "mimic-agent[postgres]"'
            ) from exc
        if not self._ready:
            self._ready = True
            self.init()
        return psycopg.connect(self.url, connect_timeout=3)

    def init(self):
        import psycopg

        with psycopg.connect(self.url, connect_timeout=3) as conn:
            conn.execute(POSTGRES_SCHEMA.format(dim=self.dim))
            conn.commit()
        self._ready = True

    def _fits(self, embedding):
        return embedding is not None and len(embedding) == self.dim

    def save_skill(self, skill, embedding):
        from psycopg.types.json import Jsonb

        with self._conn() as conn:
            conn.execute("DELETE FROM skills WHERE name = %s", (skill.name,))
            skill_id = conn.execute(
                """INSERT INTO skills (name, description, steps, embedding)
                   VALUES (%s, %s, %s, %s::vector) RETURNING id""",
                (
                    skill.name,
                    skill.description,
                    Jsonb(skill.model_dump(mode="json")),
                    _vec(embedding) if self._fits(embedding) else None,
                ),
            ).fetchone()[0]
            for step in skill.steps:
                el = step.element
                conn.execute(
                    "INSERT INTO ui_elements (skill_id, label, selector, bbox) "
                    "VALUES (%s, %s, %s, %s)",
                    (
                        skill_id,
                        el.label or el.text or el.selector,
                        el.selector,
                        Jsonb(el.bbox.model_dump()),
                    ),
                )
            conn.commit()
        return skill_id

    def get_skill(self, name):
        with self._conn() as conn:
            row = conn.execute("SELECT steps FROM skills WHERE name = %s", (name,)).fetchone()
        return Skill.model_validate(row[0]) if row else None

    def search_skills(self, embedding, k=3):
        if not self._fits(embedding):
            return []
        vec = _vec(embedding)
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT steps, 1 - (embedding <=> %s::vector) FROM skills
                   WHERE embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s""",
                (vec, vec, k),
            ).fetchall()
        return [(Skill.model_validate(r[0]), float(r[1])) for r in rows]

    def list_skills(self):
        with self._conn() as conn:
            return [r[0] for r in conn.execute("SELECT name FROM skills ORDER BY id")]

    def delete_skill(self, name):
        with self._conn() as conn:
            n = conn.execute("DELETE FROM skills WHERE name = %s", (name,)).rowcount
            conn.commit()
        return n > 0

    def heal_lookup(self, skill, step, path, embedding):
        with self._conn() as conn:
            if self._fits(embedding):
                rows = conn.execute(
                    """SELECT selector FROM healed_elements WHERE page_path = %s
                       ORDER BY (skill_name = %s AND step_index = %s) DESC,
                                embedding <=> %s::vector LIMIT 3""",
                    (path, skill, step, _vec(embedding)),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT selector FROM healed_elements
                       WHERE page_path = %s AND skill_name = %s AND step_index = %s
                       ORDER BY id DESC LIMIT 3""",
                    (path, skill, step),
                ).fetchall()
        return list(dict.fromkeys(r[0] for r in rows))

    def heal_remember(self, skill, step, path, selector, strategy, embedding):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO healed_elements
                   (skill_name, step_index, page_path, selector, strategy, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s::vector)""",
                (
                    skill,
                    step,
                    path,
                    selector,
                    strategy,
                    _vec(embedding) if self._fits(embedding) else None,
                ),
            )
            conn.commit()

    def log_steps(self, rows):
        from psycopg.types.json import Jsonb

        values = [
            tuple(
                Jsonb(r[c]) if c == "features" and r[c] is not None else r[c] for c in STEP_COLUMNS
            )
            for r in rows
        ]
        with self._conn() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO step_logs ({', '.join(STEP_COLUMNS)}) "
                f"VALUES ({', '.join(['%s'] * len(STEP_COLUMNS))})",
                values,
            )
            conn.commit()

    def step_samples(self):
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT run_id, features, success FROM step_logs
                   WHERE features IS NOT NULL AND NOT COALESCE(mitigated, false)"""
            ).fetchall()
        return [{"run_id": r[0], "features": r[1], "failed": not r[2]} for r in rows]

    def stats(self):
        with self._conn() as conn:
            return {
                t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("skills", "healed_elements", "step_logs")
            }

    def reset(self):
        with self._conn() as conn:
            conn.execute(
                "TRUNCATE skills, ui_elements, healed_elements, step_logs RESTART IDENTITY CASCADE"
            )
            conn.commit()


# ---------------------------------------------------------------- factory

_cache: dict[str, Storage] = {}


def get_storage() -> Storage:
    """The backend for the current DATABASE_URL (empty or sqlite:///... means SQLite)."""
    settings = get_settings()
    url = settings.database_url.strip()
    if url not in _cache:
        if url.startswith(("postgres://", "postgresql://")):
            _cache[url] = PostgresStorage(url)
        else:
            path = url.removeprefix("sqlite:///") if url else Path(settings.mimic_home) / "mimic.db"
            _cache[url] = SQLiteStorage(Path(path))
    return _cache[url]
