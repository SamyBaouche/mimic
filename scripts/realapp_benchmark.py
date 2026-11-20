"""Benchmark on a real app Mimic was not built for: a stock Django admin.

    uv run python scripts/realapp_benchmark.py --rows 30

Starts the admin, records one demonstration (scripted, like a person), replays N random rows
from an Excel file, and checks every saved record in the app's own database.
"""

import argparse
import os
import random
import sqlite3
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import test_realapp_django as t  # noqa: E402

from mimic.run import run_batch  # noqa: E402

CATEGORIES = {"Travel": "travel", "Meals": "meals", "Office": "office", "Software": "software"}
PAYMENTS = {"Company card": "card", "Cash": "cash", "Bank transfer": "transfer"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=30)
    args = parser.parse_args()
    rng = random.Random(7)
    tmp = Path(tempfile.mkdtemp())

    class Factory:  # minimal stand-in for pytest's tmp_path_factory
        def mktemp(self, name):
            path = tmp / name
            path.mkdir()
            return path

    gen = t.django_admin.__wrapped__(Factory())
    base, data = next(gen)
    receipts = t.receipts.__wrapped__(Factory())
    skill = t.skill.__wrapped__((base, data), receipts, Factory())

    records = []
    for i in range(args.rows):
        records.append(
            {
                "Title": f"Expense {i:03d} {rng.choice(['taxi', 'hotel', 'lunch', 'licence'])}",
                "Date": (date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))).isoformat(),
                "Amount": f"{rng.uniform(5, 900):.2f}",
                "Category": rng.choice(list(CATEGORIES)),
                "Payment": rng.choice(list(PAYMENTS)),
                "Reimbursable": rng.choice(["yes", "no"]),
                "Receipt": str(receipts / rng.choice(["taxi.pdf", "hotel.pdf", "lunch.png"])),
            }
        )
    os.environ["MIMIC_SECRET_PASSWORD"] = t.PASSWORD
    report = run_batch(skill, records, base + t.ADD, "v1", log_to_db=False, out_dir=None)

    with sqlite3.connect(data / "db.sqlite3") as db:
        saved = db.execute(
            "SELECT title, spent_on, amount, category, payment, reimbursable, receipt "
            "FROM expenses_expense ORDER BY id"
        ).fetchall()[1:]
    by_title = {row[0]: row for row in saved}
    correct = 0
    for r in records:
        row = by_title.get(r["Title"])
        correct += bool(
            row
            and row[1] == r["Date"]
            and abs(float(row[2]) - float(r["Amount"])) < 0.005
            and row[3] == CATEGORIES[r["Category"]]
            and row[4] == PAYMENTS[r["Payment"]]
            and bool(row[5]) == (r["Reimbursable"] == "yes")
            and row[6].startswith("receipts/")
        )
    print(f"\nRows: {len(records)}")
    print(f"Agent + confirmation check: {report['task_success_rate']:.0%}")
    print(
        f"Verified in the app's database: {correct}/{len(records)} ({correct / len(records):.0%})"
    )
    print(f"Average time per row: {report['avg_task_ms']} ms")
    print(f"Strategies: {report['strategies']}")
    try:
        next(gen)
    except StopIteration:
        pass


if __name__ == "__main__":
    main()
