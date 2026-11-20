"""Synthetic UI redesigns to train the element matcher.

Each example: an element the user used in a demo (old label) and the page after a redesign:
the same concept under a new label, plus decoys of the same kind. Labels come from banks of
real-world synonyms, so the model learns that "Save invoice" and "Submit bill" do the same job
while "Export CSV" does not.

Honest evaluation: every label that appears in the LedgerLite v2 redesign is HELD OUT of
training, so the sandbox test measures generalisation, not memorisation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# concept -> (kind, input type, synonyms). kind: click | fill | select
CONCEPTS: dict[str, tuple[str, str | None, list[str]]] = {
    # ---- buttons
    "save": (
        "click",
        "submit",
        [
            "Save",
            "Save invoice",
            "Submit",
            "Confirm",
            "Add bill",
            "Create",
            "Post",
            "Save changes",
            "Submit invoice",
            "Add",
            "Done",
            "Apply",
            "Record",
            "Save entry",
            "Register bill",
        ],
    ),
    "cancel": ("click", "button", ["Cancel", "Discard", "Back", "Close", "Go back", "Dismiss"]),
    "export": (
        "click",
        "button",
        ["Export", "Download", "Export CSV", "Download PDF", "Print", "Export to Excel"],
    ),
    "clear": ("click", "reset", ["Clear", "Reset", "Clear form", "Reset fields", "Start over"]),
    "delete": ("click", "button", ["Delete", "Remove", "Delete invoice", "Trash"]),
    "search": ("click", "button", ["Search", "Find", "Filter", "Look up"]),
    "next": ("click", "button", ["Next", "Continue", "Next step", "Proceed"]),
    "help": ("click", "button", ["Help", "Support", "?", "Get help"]),
    # ---- fields
    "vendor": (
        "fill",
        "text",
        [
            "Vendor",
            "Supplier",
            "Supplier name",
            "Payee",
            "Company",
            "Merchant",
            "Billed by",
            "Vendor name",
            "Seller",
        ],
    ),
    "invoice_number": (
        "fill",
        "text",
        [
            "Invoice number",
            "Invoice #",
            "Reference #",
            "Ref",
            "Bill no.",
            "Document number",
            "Invoice ID",
            "Bill number",
        ],
    ),
    "date": (
        "fill",
        "date",
        [
            "Invoice date",
            "Date",
            "Billing date",
            "Issued on",
            "Bill date",
            "Issue date",
            "Document date",
        ],
    ),
    "due_date": ("fill", "date", ["Due date", "Payment due", "Pay by", "Due on"]),
    "total": (
        "fill",
        "number",
        [
            "Total amount",
            "Total",
            "Amount",
            "Amount (CAD)",
            "Amount due",
            "Grand total",
            "Total due",
            "Invoice total",
        ],
    ),
    "tax": ("fill", "number", ["Tax", "Sales tax", "GST/HST", "VAT", "Tax amount"]),
    "email": ("fill", "email", ["Email", "E-mail address", "Contact email", "Email address"]),
    "phone": ("fill", "tel", ["Phone", "Phone number", "Telephone", "Mobile", "Contact number"]),
    "first_name": ("fill", "text", ["First name", "Given name", "First"]),
    "last_name": ("fill", "text", ["Last name", "Surname", "Family name", "Last"]),
    "address": ("fill", "text", ["Address", "Street address", "Mailing address", "Street"]),
    "city": ("fill", "text", ["City", "Town", "Municipality"]),
    "postal_code": ("fill", "text", ["Postal code", "ZIP code", "Zip", "Postcode"]),
    "notes": ("fill", "text", ["Notes", "Description", "Comments", "Memo", "Remarks"]),
    "quantity": ("fill", "number", ["Quantity", "Qty", "Units", "Number of items"]),
    "unit_price": ("fill", "number", ["Unit price", "Price", "Rate", "Price per unit"]),
    # ---- dropdowns
    "category": (
        "select",
        None,
        [
            "Category",
            "Expense type",
            "Account",
            "GL account",
            "Type",
            "Expense category",
            "Cost center",
        ],
    ),
    "currency": ("select", None, ["Currency", "Currency code", "Paid in"]),
    "status": ("select", None, ["Status", "State", "Payment status"]),
    "country": ("select", None, ["Country", "Country/Region", "Nation"]),
}

# Labels used by the LedgerLite v2 redesign: never seen in training.
HELD_OUT = {
    "Record",
    "Export CSV",
    "Clear",
    "Supplier name",
    "Reference #",
    "Billing date",
    "Amount (CAD)",
    "Expense type",
}


@dataclass
class Element:
    concept: str
    kind: str
    type: str | None
    label: str

    def text(self) -> str:
        """Same wording the agent uses at runtime (see dom.describe_text)."""
        noun = {"click": "button", "fill": "input field", "select": "dropdown"}[self.kind]
        extra = f" ({self.type})" if self.type and self.type not in ("text", "submit") else ""
        return f"{noun} '{self.label}'{extra}"


@dataclass
class Example:
    target: Element  # what the user used in the demo
    candidates: list[Element]  # the page after the redesign
    answer: int  # index of the equivalent candidate


def _labels(concept: str, split: str) -> list[str]:
    labels = CONCEPTS[concept][2]
    if split == "train":
        return [x for x in labels if x not in HELD_OUT]
    return labels


def make_example(rng: random.Random, split: str = "train") -> Example:
    concept = rng.choice(list(CONCEPTS))
    kind, typ, _ = CONCEPTS[concept]
    labels = _labels(concept, split)
    old, new = rng.sample(labels, 2) if len(labels) > 1 else (labels[0], labels[0])

    same_kind = [c for c, (k, _, _) in CONCEPTS.items() if k == kind and c != concept]
    decoys = rng.sample(same_kind, k=min(len(same_kind), rng.randint(2, 6)))
    candidates = [Element(concept, kind, typ, new)]
    for d in decoys:
        _, dtype, _ = CONCEPTS[d]
        dl = _labels(d, split)
        candidates.append(Element(d, kind, dtype, rng.choice(dl)))
    rng.shuffle(candidates)
    answer = next(i for i, c in enumerate(candidates) if c.concept == concept)
    return Example(Element(concept, kind, typ, old), candidates, answer)


def make_examples(n: int, seed: int = 0, split: str = "train") -> list[Example]:
    rng = random.Random(seed)
    return [make_example(rng, split) for _ in range(n)]


def to_pairs(examples: list[Example]) -> list[tuple[str, str, float]]:
    """(demo element, candidate, 1.0 if same purpose else 0.0)."""
    return [
        (ex.target.text(), c.text(), 1.0 if i == ex.answer else 0.0)
        for ex in examples
        for i, c in enumerate(ex.candidates)
    ]


def sandbox_v2_examples() -> list[Example]:
    """The real LedgerLite redesign (v1 demo -> v2 page): the held-out acid test."""
    buttons = [
        Element("export", "click", "button", "Export CSV"),
        Element("clear", "click", "reset", "Clear"),
        Element("save", "click", "submit", "Record"),
    ]
    fields = [
        Element("total", "fill", "number", "Amount (CAD)"),
        Element("vendor", "fill", "text", "Supplier name"),
        Element("invoice_number", "fill", "text", "Reference #"),
        Element("date", "fill", "date", "Billing date"),
    ]
    demo = [
        (Element("save", "click", "submit", "Save invoice"), buttons),
        (Element("vendor", "fill", "text", "Vendor"), fields),
        (Element("invoice_number", "fill", "text", "Invoice number"), fields),
        (Element("date", "fill", "date", "Invoice date"), fields),
        (Element("total", "fill", "number", "Total amount"), fields),
    ]
    return [
        Example(t, cands, next(i for i, c in enumerate(cands) if c.concept == t.concept))
        for t, cands in demo
    ]
