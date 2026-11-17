from fastapi.testclient import TestClient

from mimic.api.main import app


def test_health_reports_status_and_checks():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert set(body["checks"]) == {"db", "ollama"}
