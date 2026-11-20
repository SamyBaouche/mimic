"""Read tasks from the files people actually have (Excel, CSV, JSON) and write the results back.

mimic run enter_invoice --data invoices.xlsx
-> reports/<run>-results.csv   every row + mimic_status (done / failed / skipped) + error
mimic run enter_invoice --data reports/<run>-results.csv --only-failed   # retry the rest
"""

from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime
from pathlib import Path

RESULT_COLUMNS = ["mimic_status", "mimic_error", "mimic_ms"]
SUPPORTED = (".json", ".csv", ".tsv", ".txt", ".xlsx", ".xlsm")


def _cell(value) -> str:
    """Excel cells as the text a person would type."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, datetime):
        return (
            value.date().isoformat() if value.time() == datetime.min.time() else value.isoformat()
        )
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(round(value, 10))
    return str(value).strip()


def _clean(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        row = {str(k).strip(): v for k, v in row.items() if k is not None and str(k).strip()}
        if any(str(v).strip() for v in row.values() if v is not None):
            out.append(row)
    return out


def _read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):  # Excel on Windows often saves CSV as cp1252
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix == ".tsv" else csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _read_xlsx(path: Path, sheet: str | None) -> list[dict]:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    rows = [r for r in rows if any(c not in (None, "") for c in r)]
    if not rows:
        return []
    header = [_cell(c) for c in rows[0]]
    return [{h: _cell(c) for h, c in zip(header, r, strict=False) if h} for r in rows[1:]]


def load_records(path: Path, sheet: str | None = None) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = [rows]
    elif suffix in (".csv", ".tsv", ".txt"):
        rows = _read_csv(path)
    elif suffix in (".xlsx", ".xlsm"):
        rows = _read_xlsx(path, sheet)
    else:
        raise ValueError(f"Unsupported data file {path.name}: use {', '.join(SUPPORTED)}")
    return _clean(rows)


def only_failed(records: list[dict]) -> list[dict]:
    """Rows of a results file that still need doing."""
    if records and "mimic_status" not in records[0]:
        raise ValueError("--only-failed needs a results file (it has a mimic_status column)")
    return [r for r in records if str(r.get("mimic_status", "")).strip().lower() != "done"]


def strip_results(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k not in RESULT_COLUMNS} for r in records]


def write_results(path: Path, records: list[dict], rows: list[dict]) -> Path:
    """Each input row + what happened to it, as CSV (opens in Excel) or XLSX."""
    by_index = {r["index"]: r for r in rows}
    columns = list(dict.fromkeys(k for rec in records for k in rec)) + RESULT_COLUMNS
    table = []
    for i, rec in enumerate(records, 1):
        out = by_index.get(i, {"status": "not run", "error": "", "ms": ""})
        table.append(
            {
                **{k: rec.get(k, "") for k in columns[: -len(RESULT_COLUMNS)]},
                "mimic_status": out["status"],
                "mimic_error": out.get("error") or "",
                "mimic_ms": out.get("ms") or "",
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Results"
        ws.append(columns)
        for row in table:
            ws.append([row[c] for c in columns])
        wb.save(path)
    else:
        with path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: accents in Excel
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
    return path
