import socket
import threading
import time

import psycopg
import pytest
import uvicorn

from mimic.config import get_settings


@pytest.fixture(scope="session")
def test_database():
    """Point the app at a separate 'mimic_test' database so tests never touch your data."""
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

    from mimic.db import init_db

    settings.database_url = f"{base}/mimic_test"
    init_db()
    yield settings.database_url
    settings.database_url = original


@pytest.fixture(scope="session")
def sandbox_server():
    from sandbox.app import app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
