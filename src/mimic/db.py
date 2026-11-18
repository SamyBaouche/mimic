"""Database schema: learned skills (demonstrations) and UI element memory (pgvector)."""

import psycopg

from mimic.config import get_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

-- A skill = one workflow learned from a single demonstration.
CREATE TABLE IF NOT EXISTS skills (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    steps       JSONB NOT NULL DEFAULT '[]',
    embedding   vector({dim}),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Visual memory of UI elements, used to relocate a button when the layout changes.
CREATE TABLE IF NOT EXISTS ui_elements (
    id          SERIAL PRIMARY KEY,
    skill_id    INT REFERENCES skills(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    selector    TEXT,
    bbox        JSONB,
    embedding   vector({dim}),
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Healing memory: once an element is found again after a UI change, remember where.
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

-- Execution telemetry: training data for the XGBoost failure predictor.
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
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, connect_timeout=3)


def init_db() -> None:
    dim = get_settings().embedding_dim
    with connect() as conn:
        conn.execute(SCHEMA.format(dim=dim))
        conn.commit()
    print("Database initialised (skills, ui_elements, step_logs).")


if __name__ == "__main__":
    init_db()
