"""Benchmark on real scanned receipts: ICDAR 2019 SROIE (626 receipts with their answers).

    mimic extract receipts --n 50

Nothing here was made for Mimic: shop receipts from Malaysia, scanned, sometimes faded,
creased or written on by hand. The receipts are downloaded at run time from a public GitHub
mirror (github.com/zzzDavid/ICDAR-2019-SROIE) into ~/.mimic/datasets and never redistributed.

Scoring, per receipt: the company name (the registered business, as SROIE labels it, which
is also what an accounting entry needs), the date and the total paid.
- total: same amount;  date: same day (any printed format);
- store name: "exact" ignores case, spaces and punctuation; "close" also accepts a one- or
  two-letter OCR slip (similarity >= 0.9), and is reported separately.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import subprocess
from pathlib import Path

from mimic.extract import FieldSpec, normalize_date, normalize_number

SROIE_REPO = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"

SPECS = [
    FieldSpec(
        "vendor",
        "company name: the registered business (often ending in SDN BHD, S/B, LTD, INC), "
        "not a brand logo, mall or branch name, and without its registration number",
        "text",
    ),
    FieldSpec("date", "date of the purchase", "date"),
    FieldSpec("total", "final total paid, after rounding (not the cash given or the change)",
              "number"),
]  # fmt: skip
TRUTH_KEYS = {"vendor": "company", "date": "date", "total": "total"}


def _git(*args: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    out = subprocess.run(
        ["git", *args], cwd=cwd, input=stdin, capture_output=True, text=True, check=True
    )
    return out.stdout


def sample(ids: list[str], n: int) -> list[str]:
    """n receipts spread over the whole set (same choice every time)."""
    ids = sorted(ids)
    if n >= len(ids):
        return ids
    step = len(ids) / n
    return [ids[int(i * step)] for i in range(n)]


def fetch_sroie(cache: Path, n: int, repo: str = SROIE_REPO) -> list[tuple[Path, dict]]:
    """Download only the answers and the n receipt images needed (sparse git checkout)."""
    cache = Path(cache)
    if not (cache / ".git").exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", "--filter=blob:none", "--no-checkout", repo, str(cache))
        _git("sparse-checkout", "init", "--no-cone", cwd=cache)
    _git("sparse-checkout", "set", "--no-cone", "--stdin", cwd=cache, stdin="/data/key/*\n")
    _git("checkout", cwd=cache)
    ids = sample([p.stem for p in (cache / "data" / "key").glob("*.json")], n)
    patterns = ["/data/key/*", *(f"/data/img/{i}.jpg" for i in ids)]
    _git("sparse-checkout", "set", "--no-cone", "--stdin", cwd=cache, stdin="\n".join(patterns))
    _git("checkout", cwd=cache)
    return load_local(cache, ids)


def load_local(folder: Path, ids: list[str]) -> list[tuple[Path, dict]]:
    items = []
    for i in ids:
        img = Path(folder) / "data" / "img" / f"{i}.jpg"
        key = Path(folder) / "data" / "key" / f"{i}.json"
        if img.exists() and key.exists():
            items.append((img, json.loads(key.read_text(encoding="utf-8"))))
    return items


# ---------------------------------------------------------------- scoring


def _letters(text) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def judge(field: str, read, truth) -> tuple[bool, bool]:
    """(exact, close) for one field."""
    if read in (None, "") or truth in (None, ""):
        return False, False
    if field == "total":
        a, b = normalize_number(str(read)), normalize_number(str(truth))
        ok = a is not None and b is not None and abs(float(a) - float(b)) < 0.005
        return ok, ok
    if field == "date":
        ok = normalize_date(str(read)) == normalize_date(str(truth)) is not None
        return ok, ok
    a, b = _letters(read), _letters(truth)
    exact = a == b
    return exact, exact or difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


def run(extractor, items: list[tuple[Path, dict]], log=print) -> tuple[dict, list[dict]]:
    fields = list(TRUTH_KEYS)
    exact = {f: 0 for f in fields}
    close = {f: 0 for f in fields}
    wrong = flagged_wrong = flagged_total = docs_ok = 0
    seconds = 0.0
    mistakes = []
    for img, truth in items:
        ex = extractor.extract(img, SPECS)
        seconds += ex.seconds
        uncertain = set(ex.uncertain())
        flagged_total += len(uncertain)
        all_ok = True
        for f in fields:
            got = ex.fields.get(f)
            value = got.value if got else None
            ok, near = judge(f, value, truth.get(TRUTH_KEYS[f]))
            exact[f] += ok
            close[f] += near
            if not ok:
                all_ok = False
                wrong += 1
                flagged_wrong += f in uncertain
                mistakes.append(
                    {
                        "receipt": img.name,
                        "field": f,
                        "truth": truth.get(TRUTH_KEYS[f]),
                        "read": value,
                        "printed_as": got.raw if got else None,
                        "close": near,
                        "flagged": f in uncertain,
                        "note": got.note if got else "",
                    }
                )
        docs_ok += all_ok
        log(f"  {'ok ' if all_ok else 'ERR'} {img.name:<10} {ex.seconds:5.1f}s  {ex.inputs()}")
    total = max(len(items), 1)
    report = {
        "dataset": "ICDAR 2019 SROIE, real scanned receipts",
        "receipts": len(items),
        "document_accuracy": round(docs_ok / total, 4),
        "field_accuracy": {f: round(exact[f] / total, 4) for f in fields},
        "field_accuracy_all": round(sum(exact.values()) / (total * len(fields)), 4),
        "vendor_close_match": round(close["vendor"] / total, 4),
        "errors_flagged_for_review": round(flagged_wrong / wrong, 4) if wrong else 1.0,
        "fields_flagged": flagged_total,
        "share_of_fields_sent_to_review": round(flagged_total / (total * len(fields)), 4),
        "avg_seconds_per_receipt": round(seconds / total, 2),
    }
    return report, mistakes


def write_mistakes(path: Path, mistakes: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["receipt", "field", "truth", "read", "printed_as", "close", "flagged", "note"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(mistakes)
    return path
