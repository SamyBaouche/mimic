"""Reading documents: PDF rendering, skill-driven extraction, normalisation, evaluation."""

import json
from pathlib import Path

import pytest

from mimic.extract import (
    DocumentExtractor,
    FieldSpec,
    closest_option,
    evaluate,
    field_specs,
    load_pages,
    normalize,
    normalize_date,
    normalize_number,
    schema_for,
)
from mimic.models import Recording
from mimic.skills import draft_skill
from sandbox.generate_documents import LAYOUTS, generate_documents

FIXTURE = Path(__file__).parent / "fixtures" / "enter_invoice.json"
CATEGORIES = ["Office supplies", "Software", "Travel", "Utilities"]


@pytest.fixture
def skill():
    return draft_skill(Recording.model_validate_json(FIXTURE.read_text()))


@pytest.fixture(scope="module")
def docs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("docs")
    generate_documents(folder, n=6)
    return folder


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("$1,239.56", "1239.56"),
        ("3 389,96 $", "3389.96"),
        ("1.234,50 €", "1234.5"),
        ("1,234", "1234"),
        ("12,5", "12.5"),
        ("TOTAL: 178.28", "178.28"),
        ("200", "200"),
        ("n/a", None),
    ],
)
def test_amounts_in_any_format(raw, expected):
    assert normalize_number(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("May 21, 2026", "2026-05-21"),
        ("22/02/2026", "2026-02-22"),
        ("2026-08-05 08:14", "2026-08-05"),
        ("5 Aug 2026", "2026-08-05"),
        ("12/04/2026", "2026-04-12"),  # day-first by default (Canada, Europe)
        ("02/22/2026", "2026-02-22"),  # only month-first is possible
        ("tomorrow", None),
    ],
)
def test_dates_in_any_format(raw, expected):
    assert normalize_date(raw) == expected


def test_options_snap_to_the_allowed_list():
    assert closest_option("software", CATEGORIES)[0] == "Software"
    assert closest_option("Office Supply", CATEGORIES)[0] == "Office supplies"
    assert closest_option("Groceries", CATEGORIES)[0] is None


def test_specs_come_from_the_demonstration(skill):
    specs = {s.parameter: s for s in field_specs(skill, {"category": CATEGORIES})}
    assert specs["total"].kind == "number"
    assert specs["date"].kind == "date"
    assert specs["category"].kind == "option" and specs["category"].options == CATEGORIES
    assert specs["vendor"].label == "Vendor" and specs["vendor"].example == "Bureau Express"


def test_schema_restricts_dropdown_values(skill):
    schema = schema_for(field_specs(skill, {"category": CATEGORIES}))
    enum = schema["properties"]["category"]["properties"]["value"]["enum"]
    assert set(CATEGORIES) <= set(enum)


def test_generated_documents_cover_three_layouts(docs):
    truth = json.loads((docs / "truth.json").read_text())
    assert len(list(docs.glob("*.pdf"))) == 6
    assert {t["layout"] for t in truth.values()} == set(LAYOUTS)


def test_pdf_pages_render_to_png(docs, tmp_path):
    pages = load_pages(next(docs.glob("*.pdf")))
    assert len(pages) == 1 and pages[0][:8] == b"\x89PNG\r\n\x1a\n"
    img = tmp_path / "photo.png"
    img.write_bytes(pages[0])
    assert load_pages(img)[0][:4] == b"\x89PNG"
    with pytest.raises(ValueError):
        load_pages(tmp_path / "notes.docx")


class FakeVLM:
    """Answers like a vision model would: raw text as printed on the document."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def chat_json(self, model, system, user, schema, images=None):
        self.calls.append({"prompt": user, "images": images, "schema": schema})
        return self.answer


def test_extraction_normalizes_messy_values(skill, docs):
    vlm = FakeVLM(
        {
            "vendor": {"value": "Atlas Analytics", "confidence": 0.95},
            "invoice_number": {"value": "INV-20260001", "confidence": 0.9},
            "date": {"value": "22/02/2026", "confidence": 0.9},
            "total": {"value": "3 389,96 $", "confidence": 0.85},
            "category": {"value": "software", "confidence": 0.4},
        }
    )
    ex = DocumentExtractor(vlm, "qwen").extract(
        docs / "INV-20260001.pdf", field_specs(skill, {"category": CATEGORIES})
    )
    assert ex.inputs() == {
        "vendor": "Atlas Analytics",
        "invoice_number": "INV-20260001",
        "date": "2026-02-22",
        "total": "3389.96",
        "category": "Software",
    }
    assert ex.uncertain() == ["category"]  # inferred, low confidence: flagged for review
    call = vlm.calls[0]
    assert len(call["images"]) == 1 and "Total amount" in call["prompt"]
    assert "never reformat" in call["prompt"]  # the model copies, the code converts


def _receipt(docs):
    truth = json.loads((docs / "truth.json").read_text())
    name = next(n for n, t in truth.items() if t["layout"] == "receipt")
    return docs / name, truth[name]


def test_misread_text_is_flagged_by_the_pdf_text_layer(skill, docs):
    """Real failure seen with Qwen2.5-VL: 'RIDECO' read as 'RIDEKO' on a receipt font."""
    path, t = _receipt(docs)
    answer = {
        "vendor": {"value": t["vendor"].upper()[:-1] + "X", "confidence": 0.9},  # one bad letter
        "invoice_number": {"value": t["invoice_number"], "confidence": 0.9},
        "date": {"value": t["date"], "confidence": 0.9},
        "total": {"value": f"${t['total']:,.2f}", "confidence": 0.9},
        "category": {"value": t["category"], "confidence": 0.9},
    }
    ex = DocumentExtractor(FakeVLM(answer), "qwen").extract(path, field_specs(skill))
    assert ex.uncertain() == ["vendor"]
    assert ex.fields["vendor"].note == "not found in the document text"


def test_wrong_amount_is_flagged_but_right_one_is_not(skill, docs):
    path, t = _receipt(docs)
    specs = [s for s in field_specs(skill) if s.parameter == "total"]
    right = {"total": {"value": f"${t['total']:,.2f}", "confidence": 0.9}}
    wrong = {"total": {"value": "$9,999.99", "confidence": 0.9}}
    assert DocumentExtractor(FakeVLM(right), "q").extract(path, specs).uncertain() == []
    assert DocumentExtractor(FakeVLM(wrong), "q").extract(path, specs).uncertain() == ["total"]


def test_ambiguous_dates_are_flagged_for_review():
    spec = FieldSpec("date", "Invoice date", "date")
    clear = normalize(spec, "22/02/2026", 0.95)
    unsure = normalize(spec, "12/04/2026", 0.95)
    assert clear.value == "2026-02-22" and clear.confidence == 0.95
    assert unsure.value == "2026-04-12" and unsure.confidence <= 0.5
    assert "ambiguous" in unsure.note


def test_images_have_no_text_layer_so_no_grounding(docs, tmp_path):
    from mimic.extract import load_text

    assert load_text(docs / "INV-20260001.pdf").count("Atlas Analytics") == 1
    img = tmp_path / "scan.png"
    img.write_bytes(load_pages(docs / "INV-20260001.pdf")[0])
    assert load_text(img) is None


def test_missing_and_invalid_values_are_flagged():
    spec = FieldSpec("total", "Total", "number")
    assert normalize(spec, None, 0.9).value is None
    bad = normalize(spec, "see attached", 0.9)
    assert bad.value is None and bad.note == "bad number"


class TruthExtractor:
    """Reads truth.json like a perfect model, except it misreads one total, unsure."""

    model = "oracle"

    def __init__(self, folder):
        self.truth = json.loads((folder / "truth.json").read_text())

    def extract(self, path, specs):
        from mimic.extract import Extracted, Extraction

        t = self.truth[path.name]
        fields = {s.parameter: Extracted(str(t[s.parameter]), 0.95) for s in specs}
        if path.name.endswith("0000.pdf"):
            fields["total"] = Extracted("1.00", 0.3)  # wrong, but flagged
        return Extraction(path.name, fields, 0.01)


def test_evaluation_reports_accuracy_and_flagged_errors(skill, docs):
    specs = field_specs(skill, {"category": CATEGORIES})
    mapping = {p: p for p in skill.parameters}
    report = evaluate(TruthExtractor(docs), specs, docs, mapping)
    assert report["documents"] == 6
    assert report["document_accuracy"] == pytest.approx(5 / 6, abs=1e-3)
    assert report["field_accuracy"]["vendor"] == 1.0
    assert report["errors_flagged_for_review"] == 1.0  # the only error was flagged


# ---------------------------------------------------------------- end to end

pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

from mimic.approval import ConsoleApprover  # noqa: E402
from mimic.recorder import launch_browser  # noqa: E402
from mimic.run import collect_options, records_from_documents, run_batch  # noqa: E402


@pytest.fixture(scope="module")
def browser_ok():
    with sync_playwright() as p:
        try:
            launch_browser(p, headless=True).close()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")


def test_dropdown_options_are_read_from_the_live_page(skill, sandbox_server, browser_ok):
    opts = collect_options(skill, sandbox_server)
    assert set(opts["category"]) == set(CATEGORIES)


def test_documents_to_application_end_to_end(skill, docs, sandbox_server, browser_ok):
    records, titles, flags, truth = records_from_documents(
        skill, docs, TruthExtractor(docs), {"category": CATEGORIES}
    )
    assert flags[0] == {"total": "low confidence 0.30"}
    shown = []
    ask = iter(["e", "Total amount=" + str(truth[0]["total"]), "", "a", "t"])
    approver = ConsoleApprover(input_fn=lambda _: next(ask), print_fn=shown.append)
    report = run_batch(
        skill,
        records,
        sandbox_server,
        "v1",
        log_to_db=False,
        out_dir=None,
        approver=approver,
        truth=truth,
        titles=titles,
        flags=flags,
    )
    assert "check: low confidence 0.30" in shown[0]  # the reviewer sees the doubtful field
    assert report["graded_by"] == "application"
    assert report["task_success_rate"] == 1.0  # the human fixed the misread total
