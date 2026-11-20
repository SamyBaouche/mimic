import pytest
from fastapi.testclient import TestClient

from mimic.sandbox.app import app
from mimic.sandbox.generate_invoices import generate

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset():
    client.delete("/api/entries")


@pytest.mark.parametrize("layout", ["v1", "v2"])
def test_both_layouts_render(layout):
    r = client.get(f"/?layout={layout}")
    assert r.status_code == 200
    assert "LedgerLite" in r.text


def test_v2_changes_ids_to_force_self_healing():
    v1 = client.get("/?layout=v1").text
    v2 = client.get("/?layout=v2").text
    assert 'id="save-btn"' in v1 and 'id="save-btn"' not in v2
    assert 'id="btn-submit-bill"' in v2


def test_submitting_invoice_records_entry():
    inv = generate(1)[0]
    r = client.post("/invoices", data={**inv, "layout": "v2"}, follow_redirects=False)
    assert r.status_code == 303
    entries = client.get("/api/entries").json()
    assert entries == [{**inv, "total": inv["total"]}]


def test_generator_is_deterministic():
    assert generate(5) == generate(5)
