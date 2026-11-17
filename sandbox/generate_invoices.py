"""Generate synthetic invoices (JSON ground truth) for demos and evaluation.

Sprint 3 will render these as PDF/images so the vision model has to read them.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

VENDORS = {
    "Office supplies": ["Bureau Express", "PaperPro Inc.", "Staples Business"],
    "Software": ["CloudNine SaaS", "DevTools Ltd.", "Atlas Analytics"],
    "Travel": ["Air Nordique", "Hotel Montclair", "RideCo"],
    "Utilities": ["Hydro Services", "TeleLink", "AquaNet"],
}


def generate(n: int = 50, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    start = date(2026, 1, 1)
    invoices = []
    for i in range(n):
        category = rng.choice(list(VENDORS))
        invoices.append(
            {
                "vendor": rng.choice(VENDORS[category]),
                "invoice_number": f"INV-{2026}{i:04d}",
                "date": (start + timedelta(days=rng.randint(0, 300))).isoformat(),
                "total": round(rng.uniform(20, 5000), 2),
                "category": category,
            }
        )
    return invoices


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "invoices.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(generate(), indent=2))
    print(f"Wrote {out}")
