import pytest

from mimic.mapping import apply_mapping, map_deterministic, map_inputs

FIELDS = ["vendor", "invoice_number", "date", "total", "category"]


@pytest.mark.parametrize(
    "params, expected",
    [
        (FIELDS, {f: f for f in FIELDS}),
        (
            ["vendor_name", "invoice_number", "invoice_date", "total_amount", "category"],
            {
                "vendor_name": "vendor",
                "invoice_number": "invoice_number",
                "invoice_date": "date",
                "total_amount": "total",
                "category": "category",
            },
        ),
        (
            ["supplier", "reference", "billing_date", "amount", "expense_type"],
            {
                "supplier": "vendor",
                "reference": "invoice_number",
                "billing_date": "date",
                "amount": "total",
                "expense_type": "category",
            },
        ),
    ],
)
def test_deterministic_mapping(params, expected):
    assert map_deterministic(params, FIELDS) == expected


class FakeClient:
    def __init__(self, answer):
        self.answer = answer

    def chat_json(self, *args):
        return self.answer


def test_llm_resolves_what_tokens_cannot():
    client = FakeClient({"mapping": [{"parameter": "payee", "field": "vendor"}]})
    mapping = map_inputs(["payee", "total"], ["vendor", "total"], client, "m")
    assert mapping == {"payee": "vendor", "total": "total"}


def test_llm_cannot_invent_fields():
    client = FakeClient({"mapping": [{"parameter": "payee", "field": "made_up"}]})
    with pytest.raises(KeyError, match="payee"):
        map_inputs(["payee"], ["vendor"], client, "m")


def test_unmappable_parameter_raises_without_llm():
    with pytest.raises(KeyError):
        map_inputs(["payee"], ["vendor"])


def test_apply_mapping():
    assert apply_mapping({"vendor": "A", "x": 1}, {"supplier": "vendor"}) == {"supplier": "A"}
