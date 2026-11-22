"""Real-receipts benchmark: scoring rules and the sparse download (with a local git repo)."""

import json
import shutil
import subprocess

import pytest
from PIL import Image

from mimic import receipts
from mimic.extract import Extracted, Extraction, normalize_date


@pytest.mark.parametrize(
    "printed, iso",
    [
        ("25/12/2018", "2018-12-25"),
        ("12-01-19", "2019-01-12"),  # two-digit year
        ("28 MAR 18", "2018-03-28"),
        ("02/JAN/2017", "2017-01-02"),
        ("20180304", "2018-03-04"),
        ("25032018", "2018-03-25"),
        ("(06/12/2016)", "2016-12-06"),
        ("25/12/2018 8:13:39 PM", "2018-12-25"),
        ("15 janv. 2026", "2026-01-15"),
        ("3 mars 2026", "2026-03-03"),
    ],
)
def test_real_receipt_date_formats(printed, iso):
    assert normalize_date(printed) == iso


@pytest.mark.parametrize(
    "field, read, truth, expected",
    [
        ("total", "RM 9.00", "9.00", (True, True)),
        ("total", "10.00", "9.00", (False, False)),
        ("date", "2018-12-25", "25/12/2018", (True, True)),
        (
            "vendor",
            "Book Ta.K (Taman Daya) Sdn Bhd",
            "BOOK TA .K (TAMAN DAYA) SDN BHD",
            (True, True),
        ),
        (
            "vendor",
            "BOOK TAK (TAMAN DAYA) SDN BHD.",
            "BOOK TA .K (TAMAN DAYA) SDN BHD",
            (True, True),
        ),
        (
            "vendor",
            "BOOK TA K (TAMAN DAYO) SDN BHD",
            "BOOK TA .K (TAMAN DAYA) SDN BHD",
            (False, True),
        ),
        ("vendor", "ABC TRADING", "BOOK TA .K (TAMAN DAYA) SDN BHD", (False, False)),
        ("vendor", None, "ABC", (False, False)),
    ],
)
def test_judge(field, read, truth, expected):
    assert receipts.judge(field, read, truth) == expected


def test_sample_is_spread_and_stable():
    ids = [f"{i:03d}" for i in range(626)]
    picked = receipts.sample(ids, 50)
    assert len(picked) == 50 and picked[0] == "000" and picked[-1] > "600"
    assert picked == receipts.sample(list(reversed(ids)), 50)


class Reader:
    """Stands in for the vision model: one wrong total (flagged), one wrong name (not)."""

    def extract(self, path, specs):
        values = {
            "a.jpg": ("SHOP A", "25/12/2018", "9.00", 0.9),
            "b.jpg": ("SHOP B", "01/01/2019", "99.00", 0.3),  # wrong total, low confidence
            "c.jpg": ("SHOP X", "02/01/2019", "5.00", 0.9),  # wrong name, confident
        }[path.name]
        fields = {
            "vendor": Extracted(values[0], 0.9, values[0]),
            "date": Extracted(normalize_date(values[1]), 0.9, values[1]),
            "total": Extracted(values[2], values[3], values[2]),
        }
        return Extraction(path.name, fields, 2.0)


def _dataset(root, rows):
    for name, truth in rows.items():
        (root / "data" / "key").mkdir(parents=True, exist_ok=True)
        (root / "data" / "img").mkdir(parents=True, exist_ok=True)
        (root / "data" / "key" / f"{name}.json").write_text(json.dumps(truth))
        Image.new("RGB", (20, 40), "white").save(root / "data" / "img" / f"{name}.jpg")


ROWS = {
    "a": {"company": "SHOP A", "date": "25/12/2018", "total": "9.00"},
    "b": {"company": "SHOP B", "date": "01-01-19", "total": "12.50"},
    "c": {"company": "SHOP C", "date": "02/01/2019", "total": "5.00"},
}


def test_run_scores_and_lists_every_mistake(tmp_path):
    _dataset(tmp_path, ROWS)
    items = receipts.load_local(tmp_path, ["a", "b", "c"])
    report, mistakes = receipts.run(Reader(), items, log=lambda *_: None)
    assert report["document_accuracy"] == round(1 / 3, 4)
    assert report["field_accuracy"] == {
        "vendor": round(2 / 3, 4),
        "date": 1.0,
        "total": round(2 / 3, 4),
    }
    assert report["errors_flagged_for_review"] == 0.5  # the total was flagged, the name was not
    assert [(m["receipt"], m["field"], m["flagged"]) for m in mistakes] == [
        ("b.jpg", "total", True),
        ("c.jpg", "vendor", False),
    ]
    out = receipts.write_mistakes(tmp_path / "m.csv", mistakes)
    assert out.read_text(encoding="utf-8-sig").splitlines()[0].startswith("receipt,field,truth")


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_fetch_downloads_only_what_it_needs(tmp_path):
    origin = tmp_path / "origin"
    _dataset(origin, ROWS)
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], cwd=origin, check=True)
    subprocess.run([*git, "add", "."], cwd=origin, check=True)
    subprocess.run([*git, "commit", "-qm", "data"], cwd=origin, check=True)

    cache = tmp_path / "cache"
    items = receipts.fetch_sroie(cache, 2, repo=origin.as_uri())
    assert [p.name for p, _ in items] == ["a.jpg", "b.jpg"]
    assert not (cache / "data" / "img" / "c.jpg").exists()  # not downloaded
    again = receipts.fetch_sroie(cache, 3, repo=origin.as_uri())  # reuses the clone
    assert len(again) == 3
