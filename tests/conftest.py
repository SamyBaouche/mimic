import socket
import threading
import time

import pytest
import uvicorn

from mimic.config import get_settings

# Captured before any test changes it: the Postgres URL from your .env, if you use Postgres.
CONFIGURED_DB_URL = get_settings().database_url


@pytest.fixture(scope="session", autouse=True)
def isolated_storage(tmp_path_factory):
    """No test ever reads or writes your real skills: default storage is a throwaway SQLite."""
    settings = get_settings()
    saved = settings.database_url, settings.mimic_home
    settings.mimic_home = tmp_path_factory.mktemp("mimic-home")
    settings.database_url = ""
    yield
    settings.database_url, settings.mimic_home = saved


@pytest.fixture(scope="session", params=["sqlite", "postgres"])
def test_database(request, tmp_path_factory):
    """Run storage tests on a throwaway SQLite file, and on a 'mimic_test' Postgres database
    when one is reachable (DATABASE_URL=postgresql://...). Never touches your own data."""
    from mimic import storage

    settings = get_settings()
    original = settings.database_url
    configured = CONFIGURED_DB_URL
    if request.param == "sqlite":
        url = f"sqlite:///{tmp_path_factory.mktemp('db') / 'mimic_test.db'}"
    else:
        if not configured.startswith("postgres"):
            pytest.skip("Postgres not configured (DATABASE_URL is not postgresql://...)")
        try:
            import psycopg

            with psycopg.connect(configured, autocommit=True, connect_timeout=3) as conn:
                exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'mimic_test'")
                if not exists.fetchone():
                    conn.execute("CREATE DATABASE mimic_test")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Postgres not available: {exc}")
        url = f"{configured.rpartition('/')[0]}/mimic_test"
    settings.database_url = url
    storage.get_storage().init()
    yield url
    settings.database_url = original


@pytest.fixture(scope="session")
def sandbox_server():
    from mimic.sandbox.app import app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/"
    server.should_exit = True
