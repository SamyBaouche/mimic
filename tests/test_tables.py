"""Excel / CSV in, results file out, retry only what failed."""

import csv
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
from openpyxl import Workbook, load_workbook

from mimic.tables import load_records, only_failed, strip_results, write_results

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"


def _xlsx(path: Path, rows: list[list]) -> Path:
    wb = Workbook()
    ws = wb.active
    for r in rows:
        ws.append(r)
    wb.save(path)
    return path


def test_excel_cells_become_what_a_person_would_type(tmp_path):
    path = _xlsx(
        tmp_path / "in.xlsx",
        [
            [" Vendor ", "Invoice number", "Date", "Total", "Paid", None],
            ["Bureau Express", "INV-1", datetime(2026, 5, 21), 1239.56, True, None],
            [None, None, None, None, None, None],  # blank row in the middle
            ["Atlas", 42, date(2026, 2, 1), 300.0, False, None],
        ],
    )
    assert load_records(path) == [
        {
            "Vendor": "Bureau Express",
            "Invoice number": "INV-1",
            "Date": "2026-05-21",
            "Total": "1239.56",
            "Paid": "yes",
        },
        {
            "Vendor": "Atlas",
            "Invoice number": "42",
            "Date": "2026-02-01",
            "Total": "300",
            "Paid": "no",
        },
    ]


def test_french_excel_csv_with_semicolons_and_accents(tmp_path):
    path = tmp_path / "in.csv"
    path.write_bytes("Fournisseur;Montant\r\nCafé Dépôt;12,50\r\n;\r\n".encode("cp1252"))
    assert load_records(path) == [{"Fournisseur": "Café Dépôt", "Montant": "12,50"}]


def test_results_round_trip_and_retry(tmp_path):
    records = [{"vendor": "A"}, {"vendor": "B"}, {"vendor": "C"}]
    rows = [
        {"index": 1, "status": "done", "error": None, "ms": 300},
        {"index": 2, "status": "failed", "error": "element not found", "ms": 900},
    ]  # row 3 never ran (run stopped)
    for name in ("out.csv", "out.xlsx"):
        out = write_results(tmp_path / name, records, rows)
        back = load_records(out)
        assert [r["mimic_status"] for r in back] == ["done", "failed", "not run"]
        retry = strip_results(only_failed(back))
        assert retry == [{"vendor": "B"}, {"vendor": "C"}]
    with (tmp_path / "out.csv").open(encoding="utf-8-sig") as f:
        assert next(csv.reader(f))[-3:] == ["mimic_status", "mimic_error", "mimic_ms"]
    wb = load_workbook(tmp_path / "out.xlsx")
    assert wb.active.title == "Results"


def test_only_failed_needs_a_results_file():
    with pytest.raises(ValueError):
        only_failed([{"vendor": "A"}])


def test_cli_runs_an_excel_file_and_retries_failures(sandbox_server, tmp_path, monkeypatch):
    """Excel headers as people write them, one bad row, then --only-failed."""
    from mimic.cli import main
    from mimic.models import Recording
    from mimic.skills import draft_skill
    from mimic.store import SkillStore

    pytest.importorskip("playwright")
    SkillStore().save(draft_skill(Recording.model_validate_json(FIXTURE.read_text())), None)
    data = _xlsx(
        tmp_path / "invoices.xlsx",
        [
            ["Vendor", "Invoice number", "Date", "Total", "Category"],
            ["Bureau Express", "INV-9001", datetime(2026, 3, 2), 120.5, "Software"],
            ["Atlas Analytics", "INV-9002", datetime(2026, 3, 3), 99, "Groceries"],  # not an option
            ["TeleLink", "INV-9003", datetime(2026, 3, 4), 45.25, "Utilities"],
        ],
    )
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results.xlsx"
    args = ["run", "enter_invoice", "--url", sandbox_server, "--yes", "--no-heal"]
    try:
        main([*args, "--data", str(data), "--results", str(results)])
    except Exception as exc:  # noqa: BLE001
        if "Executable doesn't exist" in str(exc):
            pytest.skip("No Chromium available")
        raise
    back = load_records(results)
    assert [r["mimic_status"] for r in back] == ["done", "failed", "done"]
    saved = {e["invoice_number"] for e in httpx.get(sandbox_server + "api/entries").json()}
    assert {"INV-9001", "INV-9003"} <= saved

    retry = tmp_path / "retry.csv"
    main([*args, "--data", str(results), "--only-failed", "--results", str(retry)])
    again = load_records(retry)
    assert [r["Invoice number"] for r in again] == ["INV-9002"]  # only the failed row ran
    assert again[0]["mimic_status"] == "failed"
