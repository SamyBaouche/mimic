"""Supplier emails with an invoice PDF attached, as they land in an accounts-payable inbox.

Each message is a standard .eml file (what Outlook, Gmail or Thunderbird export), so the same
workflow runs on a real exported mailbox. truth.json keeps what each invoice really says.
"""

from __future__ import annotations

import json
import random
import tempfile
from datetime import UTC, datetime
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from pathlib import Path

from mimic.sandbox.generate_documents import LAYOUTS, render
from mimic.sandbox.generate_invoices import generate

DEFAULT_FOLDER = Path("sandbox/data/mail")
INBOX = "ap@ledgerlite.test"
GREETINGS = ["Hello,", "Hi team,", "Bonjour,", "Good morning,"]
BODIES = [
    "Please find attached invoice {n} for our recent services.",
    "Attached is invoice {n}. Payment terms: 30 days.",
    "Veuillez trouver ci-joint la facture {n}.",
    "Here is our invoice {n}. Let us know if anything is missing.",
]


def _domain(vendor: str) -> str:
    return "".join(c for c in vendor.lower() if c.isalnum())[:16] + ".test"


def build_message(inv: dict, pdf: Path, index: int, rng: random.Random) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = f"{inv['vendor']} Billing <billing@{_domain(inv['vendor'])}>"
    msg["To"] = INBOX
    msg["Subject"] = f"Invoice {inv['invoice_number']} from {inv['vendor']}"
    msg["Date"] = format_datetime(
        datetime(2026, 11, 2, 9, tzinfo=UTC).replace(hour=8 + index % 9, minute=rng.randint(0, 59))
    )
    msg["Message-ID"] = make_msgid(domain=_domain(inv["vendor"]))
    body = rng.choice(BODIES).format(n=inv["invoice_number"])
    msg.set_content(f"{rng.choice(GREETINGS)}\n\n{body}\n\nBest regards,\n{inv['vendor']}\n")
    msg.add_attachment(pdf.read_bytes(), maintype="application", subtype="pdf", filename=pdf.name)
    return msg


def generate_mail(out_dir: Path, n: int = 12, seed: int = 11) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    invoices = generate(n, seed=seed)
    truth = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, inv in enumerate(invoices, 1):
            pdf = Path(tmp) / f"{inv['invoice_number']}.pdf"
            render(inv, LAYOUTS[i % len(LAYOUTS)], pdf, seed=seed + i)
            message_id = f"mail-{i:04d}"
            (out_dir / f"{message_id}.eml").write_bytes(bytes(build_message(inv, pdf, i, rng)))
            truth[message_id] = inv
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main(folder: Path = DEFAULT_FOLDER) -> Path:
    generate_mail(folder)
    print(f"Wrote 12 supplier emails with invoice PDFs to {folder}")
    return folder


if __name__ == "__main__":
    main()
