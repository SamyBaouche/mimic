"""Render the synthetic invoices as realistic PDFs, to test reading documents.

Three layouts with different wording and formats, so extraction cannot rely on one template:
  classic   "Invoice No.", "Date: March 14, 2026", subtotal + tax + total
  modern    "Ref #", "Issued 14/03/2026", "Amount due 1 234,56 $"
  receipt   monospace, "No. INV-...", ISO date, "TOTAL $1,234.56"
The category never appears on the document: it must be inferred from the line items.
Ground truth stays in sandbox/data/invoices.json.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from mimic.sandbox.generate_invoices import generate

ITEMS = {
    "Office supplies": ["Printer paper (case)", "Toner cartridge", "Desk organizers", "Pens (box)"],
    "Software": ["Annual license", "Cloud subscription", "Support plan", "User seats"],
    "Travel": ["Flight YUL-YYZ", "Hotel night", "Airport shuttle", "Baggage fee"],
    "Utilities": ["Electricity - monthly", "Internet service", "Water service", "Phone line"],
}
ADDRESSES = ["1200 Rue Peel, Montreal, QC", "88 King St W, Toronto, ON", "45 Main St, Ottawa, ON"]
LAYOUTS = ["classic", "modern", "receipt"]


def _money_us(x: float) -> str:
    return f"${x:,.2f}"


def _money_fr(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ").replace(".", ",") + " $"


def _lines(total: float, category: str, rng: random.Random) -> list[tuple[str, int, float]]:
    """Split a total into 1-3 line items that add up exactly."""
    n = rng.randint(1, 3)
    names = rng.sample(ITEMS[category], n)
    remaining = round(total, 2)
    out = []
    for i, name in enumerate(names):
        if i == n - 1:
            amount = remaining
        else:
            amount = round(remaining * rng.uniform(0.2, 0.6), 2)
        remaining = round(remaining - amount, 2)
        out.append((name, 1, amount))
    return out


def render(inv: dict, layout: str, path: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    d = date.fromisoformat(inv["date"])
    total = float(inv["total"])
    lines = _lines(total, inv["category"], rng)
    c = canvas.Canvas(str(path), pagesize=LETTER)
    w, h = LETTER

    if layout == "classic":
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, h - 70, inv["vendor"])
        c.setFont("Helvetica", 10)
        c.drawString(50, h - 88, rng.choice(ADDRESSES))
        c.setFont("Helvetica-Bold", 16)
        c.drawRightString(w - 50, h - 70, "INVOICE")
        c.setFont("Helvetica", 11)
        c.drawRightString(w - 50, h - 92, f"Invoice No.: {inv['invoice_number']}")
        c.drawRightString(w - 50, h - 108, f"Date: {d.strftime('%B %d, %Y')}")
        c.drawString(50, h - 150, "Bill to: Acme Corp, Accounts Payable")
        y = h - 200
        c.setFont("Helvetica-Bold", 11)
        c.drawString(50, y, "Description")
        c.drawRightString(w - 50, y, "Amount")
        c.setFont("Helvetica", 11)
        subtotal = round(total / 1.15, 2)
        for name, _, amount in lines:
            y -= 20
            c.drawString(50, y, name)
            c.drawRightString(w - 50, y, _money_us(round(amount / 1.15, 2)))
        y -= 35
        c.drawRightString(w - 150, y, "Subtotal")
        c.drawRightString(w - 50, y, _money_us(subtotal))
        y -= 18
        c.drawRightString(w - 150, y, "Tax (15%)")
        c.drawRightString(w - 50, y, _money_us(round(total - subtotal, 2)))
        y -= 22
        c.setFont("Helvetica-Bold", 12)
        c.drawRightString(w - 150, y, "Total")
        c.drawRightString(w - 50, y, _money_us(total))

    elif layout == "modern":
        c.setFillColorRGB(0.12, 0.23, 0.37)
        c.rect(0, h - 110, w, 110, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1)
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, h - 60, "Bill")
        c.setFont("Helvetica", 11)
        c.drawString(50, h - 85, f"from {inv['vendor']}")
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica", 11)
        c.drawString(50, h - 150, f"Ref # {inv['invoice_number']}")
        c.drawString(50, h - 168, f"Issued {d.strftime('%d/%m/%Y')}")
        c.drawString(320, h - 150, "Payment terms: Net 30")
        y = h - 220
        for name, qty, amount in lines:
            c.drawString(50, y, f"{qty} x {name}")
            c.drawRightString(w - 50, y, _money_fr(amount))
            y -= 20
        y -= 30
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, "Amount due")
        c.drawRightString(w - 50, y, _money_fr(total))

    else:  # receipt
        c.setFont("Courier-Bold", 14)
        c.drawCentredString(w / 2, h - 60, inv["vendor"].upper())
        c.setFont("Courier", 11)
        c.drawCentredString(w / 2, h - 78, rng.choice(ADDRESSES))
        c.drawString(150, h - 115, f"No. {inv['invoice_number']}")
        c.drawString(
            150, h - 131, f"{d.isoformat()}  {rng.randint(8, 18):02d}:{rng.randint(0, 59):02d}"
        )
        y = h - 165
        for name, _, amount in lines:
            c.drawString(150, y, name[:26])
            c.drawRightString(w - 150, y, f"{amount:,.2f}")
            y -= 16
        c.drawString(150, y - 6, "-" * 38)
        c.setFont("Courier-Bold", 12)
        c.drawString(150, y - 24, "TOTAL")
        c.drawRightString(w - 150, y - 24, _money_us(total))
        c.setFont("Courier", 10)
        c.drawCentredString(w / 2, y - 60, "Thank you for your business!")
    c.showPage()
    c.save()


def generate_documents(out_dir: Path, n: int = 50, seed: int = 42) -> list[dict]:
    """Write one PDF per invoice and a truth.json mapping file name -> expected values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    invoices = generate(n, seed=seed)
    truth = {}
    for i, inv in enumerate(invoices):
        layout = LAYOUTS[i % len(LAYOUTS)]
        name = f"{inv['invoice_number']}.pdf"
        render(inv, layout, out_dir / name, seed=seed + i)
        truth[name] = {**inv, "layout": layout}
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    return invoices


DEFAULT_FOLDER = Path("sandbox/data/documents")


def main(folder: Path = DEFAULT_FOLDER) -> Path:
    generate_documents(folder)
    print(f"Wrote 50 invoice PDFs (3 layouts) + truth.json to {folder}")
    return folder


if __name__ == "__main__":
    main()
