"""Read documents (PDF, photos) and extract exactly the values a skill needs.

The skill knows its parameters (Vendor, Invoice number, Total amount...), the label each one
had on screen, an example value from the demo, and for dropdowns the allowed options.
Qwen2.5-VL reads the document with that list and answers JSON with a confidence per field.
Values are then normalised (dates -> YYYY-MM-DD, amounts -> numbers, options -> closest
allowed one). Works for any skill: invoices, business cards, forms...

    uv run python -m mimic.extract eval --docs sandbox/data/documents
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from mimic.skills import Skill

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOC_TYPES = IMAGE_TYPES | {".pdf"}
LOW_CONFIDENCE = 0.6

SYSTEM = (
    "You read business documents (invoices, receipts, forms, cards) and extract the requested "
    "fields precisely. Copy values exactly as written. Answer only JSON."
)


# ---------------------------------------------------------------- documents → images


def load_pages(path: Path, max_pages: int = 2, scale: float = 2.0) -> list[bytes]:
    """PNG bytes for each page (PDF) or the image itself."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(path))
        pages = []
        for i in range(min(len(pdf), max_pages)):
            img = pdf[i].render(scale=scale).to_pil().convert("RGB")
            pages.append(_png(img))
        return pages
    if suffix in IMAGE_TYPES:
        return [_png(Image.open(path).convert("RGB"))]
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _png(img: Image.Image, max_side: int = 1600) -> bytes:
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def load_text(path: Path) -> str | None:
    """The text layer of a PDF (None for images or scanned PDFs without text)."""
    if path.suffix.lower() != ".pdf":
        return None
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    text = "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    return text if text.strip() else None


def list_documents(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in DOC_TYPES)


# ---------------------------------------------------------------- what to extract


@dataclass
class FieldSpec:
    parameter: str
    label: str
    kind: str  # text | number | date | option
    example: str | None = None
    options: list[str] = field(default_factory=list)

    def describe(self) -> str:
        # The model only COPIES what is printed; deterministic code converts formats.
        hint = {
            "number": "copy the amount exactly as printed, with its symbols and separators",
            "date": "copy the date exactly as printed, do not convert it",
            "option": f"exactly one of: {', '.join(self.options)}",
            "text": "copy the text exactly as printed",
        }[self.kind]
        example = f" (example: {self.example})" if self.example and self.kind != "option" else ""
        return f"- {self.parameter}: the '{self.label}'{example}; {hint}"


def field_specs(skill: Skill, options: dict[str, list[str]] | None = None) -> list[FieldSpec]:
    """What the skill needs, derived from the demonstration."""
    options = options or {}
    specs = []
    for step in skill.steps:
        if not step.parameter:
            continue
        el = step.element
        if step.action == "select":
            kind = "option"
        elif el.type == "number":
            kind = "number"
        elif el.type == "date":
            kind = "date"
        else:
            kind = "text"
        specs.append(
            FieldSpec(
                parameter=step.parameter,
                label=el.label or el.name or step.parameter,
                kind=kind,
                example=step.value,
                options=options.get(step.parameter, []),
            )
        )
    return specs


def schema_for(specs: list[FieldSpec]) -> dict:
    props = {}
    for s in specs:
        value = {"type": ["string", "null"]}
        if s.kind == "option" and s.options:
            value = {"type": ["string", "null"], "enum": [*s.options, None]}
        props[s.parameter] = {
            "type": "object",
            "properties": {"value": value, "confidence": {"type": "number"}},
            "required": ["value", "confidence"],
        }
    return {"type": "object", "properties": props, "required": [s.parameter for s in specs]}


# ---------------------------------------------------------------- normalisation

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%B %d %Y",
]


_NUMERIC_DATE = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$")


def date_is_ambiguous(text: str) -> bool:
    """'04/12/2026' could be April 12 or December 4: a human should confirm."""
    m = _NUMERIC_DATE.match(text.strip())
    return bool(m) and int(m[1]) <= 12 and int(m[2]) <= 12 and m[1] != m[2]


def normalize_date(text: str, day_first: bool = True) -> str | None:
    """Day-first by default for numeric dates (Canada, Europe); ambiguity is flagged apart."""
    text = re.sub(r"\s+\d{1,2}:\d{2}(:\d{2})?$", "", text.strip())  # drop a trailing time
    m = _NUMERIC_DATE.match(text)
    if m:
        a, b, year = int(m[1]), int(m[2]), int(m[3])
        if a > 12:  # 22/02/2026: only day-first makes sense
            day, month = a, b
        elif b > 12:  # 02/22/2026: only month-first makes sense
            day, month = b, a
        else:  # both possible: follow the convention, the ambiguity is flagged elsewhere
            day, month = (a, b) if day_first else (b, a)
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_number(text: str) -> str | None:
    """'$1,234.56', '1 234,56 $', '1.234,56 €', '1234.5' -> '1234.56'-style string."""
    s = re.sub(r"[^\d,.\-]", "", text.replace(" ", " ").replace(" ", ""))
    if not re.search(r"\d", s):
        return None
    if "," in s and "." in s:
        # the last separator is the decimal one
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = s.replace(",", ".") if len(tail) in (1, 2) else s.replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return f"{value:.2f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))


def closest_option(text: str, options: list[str]) -> tuple[str | None, float]:
    if not options:
        return text, 1.0
    scored = [(SequenceMatcher(None, text.lower(), o.lower()).ratio(), o) for o in options]
    score, best = max(scored)
    return (best, score) if score >= 0.6 else (None, score)


@dataclass
class Extracted:
    value: str | None
    confidence: float
    raw: str | None = None
    note: str = ""


def normalize(spec: FieldSpec, raw: str | None, confidence: float) -> Extracted:
    if raw is None or str(raw).strip() == "":
        return Extracted(None, 0.0, raw, "not found")
    raw = str(raw).strip()
    if spec.kind == "date":
        value = normalize_date(raw)
        if value is None:
            return Extracted(None, 0.0, raw, "bad date")
        if date_is_ambiguous(raw):
            return Extracted(value, min(confidence, 0.5), raw, f"ambiguous date '{raw}'")
        return Extracted(value, confidence, raw)
    if spec.kind == "number":
        value = normalize_number(raw)
        return Extracted(value, confidence if value else 0.0, raw, "" if value else "bad number")
    if spec.kind == "option":
        value, score = closest_option(raw, spec.options)
        if value is None:
            return Extracted(None, 0.0, raw, "not an allowed option")
        return Extracted(value, min(confidence, score), raw, "" if value == raw else "mapped")
    return Extracted(raw, confidence, raw)


# ---------------------------------------------------------------- extraction


_NUMBER_TOKEN = re.compile(r"\d[\d\s.,]*\d|\d")


def grounded(spec: FieldSpec, ex: Extracted, text: str) -> bool:
    """Does what the model read actually appear in the document's own text?"""
    if ex.value is None or spec.kind == "option":
        return True  # nothing to check / categories are often inferred
    flat = " ".join(text.lower().split())
    raw = " ".join(str(ex.raw).lower().split())
    if spec.kind == "number":
        numbers = {
            normalize_number(t) for line in text.splitlines() for t in _NUMBER_TOKEN.findall(line)
        }
        return ex.value in numbers
    return raw in flat


@dataclass
class Extraction:
    document: str
    fields: dict[str, Extracted]
    seconds: float

    def inputs(self) -> dict:
        return {k: v.value for k, v in self.fields.items()}

    def uncertain(self) -> list[str]:
        return [
            k for k, v in self.fields.items() if v.value is None or v.confidence < LOW_CONFIDENCE
        ]


class DocumentExtractor:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def extract(self, path: Path, specs: list[FieldSpec]) -> Extraction:
        t0 = time.perf_counter()
        pages = load_pages(path)
        text = load_text(path)
        fields = "\n".join(s.describe() for s in specs)
        prompt = (
            "Extract these fields from the document. Copy values exactly as printed, never "
            "reformat dates or amounts. If a field is not written on the "
            "document but can be clearly inferred (for example a category from the line "
            "items), infer it and lower the confidence. If it is absent, use null.\n"
            f"{fields}\n"
            "Give each field a confidence from 0 to 1."
        )
        answer = self.client.chat_json(self.model, SYSTEM, prompt, schema_for(specs), images=pages)
        out = {}
        for s in specs:
            item = answer.get(s.parameter) or {}
            raw = item.get("value") if isinstance(item, dict) else item
            try:
                conf = float(item.get("confidence", 0.5)) if isinstance(item, dict) else 0.5
            except (TypeError, ValueError):
                conf = 0.5
            got = normalize(s, raw, max(0.0, min(conf, 1.0)))
            if text is not None and not grounded(s, got, text):
                # Vision read something the document's text does not contain: flag it.
                got = Extracted(
                    got.value, min(got.confidence, 0.3), got.raw, "not found in the document text"
                )
            out[s.parameter] = got
        return Extraction(path.name, out, time.perf_counter() - t0)


# ---------------------------------------------------------------- evaluation


def _same(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 0.005
    except (TypeError, ValueError):
        return " ".join(str(a).lower().split()) == " ".join(str(b).lower().split())


def evaluate(extractor, specs, folder: Path, mapping: dict[str, str], limit=None) -> dict:
    """Field-level accuracy against truth.json. mapping: skill parameter -> truth field."""
    truth = json.loads((folder / "truth.json").read_text())
    docs = [d for d in list_documents(folder) if d.name in truth][:limit]
    per_field = {s.parameter: [0, 0] for s in specs}
    per_layout: dict[str, list[int]] = {}
    seconds, perfect = [], 0
    flagged_wrong = flagged = wrong = 0
    for doc in docs:
        ex = extractor.extract(doc, specs)
        seconds.append(ex.seconds)
        expected = truth[doc.name]
        all_ok = True
        for param, got in ex.fields.items():
            ok = _same(got.value, expected[mapping[param]])
            per_field[param][0] += ok
            per_field[param][1] += 1
            low = got.value is None or got.confidence < LOW_CONFIDENCE
            flagged += low
            wrong += not ok
            flagged_wrong += (not ok) and low
            all_ok &= ok
        perfect += all_ok
        layout = expected.get("layout", "?")
        per_layout.setdefault(layout, [0, 0])
        per_layout[layout][0] += all_ok
        per_layout[layout][1] += 1
        mark = "ok " if all_ok else "ERR"
        print(f"  {mark} {doc.name:<20} {ex.seconds:5.1f}s  {ex.inputs()}")
    n = max(len(docs), 1)
    return {
        "documents": len(docs),
        "document_accuracy": round(perfect / n, 4),
        "field_accuracy": {k: round(a / max(b, 1), 4) for k, (a, b) in per_field.items()},
        "by_layout": {k: round(a / max(b, 1), 4) for k, (a, b) in per_layout.items()},
        "errors_flagged_for_review": round(flagged_wrong / wrong, 4) if wrong else None,
        "fields_flagged": flagged,
        "avg_seconds_per_document": round(sum(seconds) / n, 2),
    }


def main() -> None:
    from mimic.config import get_settings
    from mimic.llm import OllamaClient
    from mimic.mapping import map_inputs
    from mimic.store import SkillStore

    parser = argparse.ArgumentParser(description="Extract skill inputs from documents")
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="Measure extraction accuracy against truth.json")
    e.add_argument("--skill", default="enter_invoice")
    e.add_argument("--docs", type=Path, default=Path("sandbox/data/documents"))
    e.add_argument("--limit", type=int, default=None)
    r = sub.add_parser("read", help="Extract one document and print the values")
    r.add_argument("document", type=Path)
    r.add_argument("--skill", default="enter_invoice")
    args = parser.parse_args()

    settings = get_settings()
    skill = SkillStore().get(args.skill)
    if skill is None:
        raise SystemExit(f"Unknown skill '{args.skill}'.")
    options = default_options(skill)
    if any(s.action == "select" and s.parameter not in options for s in skill.steps):
        try:  # read dropdown choices from the live app (sandbox must be running)
            from mimic.run import collect_options

            options = collect_options(skill, skill.start_url or settings.sandbox_url) | options
        except Exception as exc:  # noqa: BLE001
            print(f"  (dropdown options not read: {type(exc).__name__}; values kept as read)")
    specs = field_specs(skill, options)
    extractor = DocumentExtractor(OllamaClient(), settings.vision_model)

    if args.cmd == "read":
        ex = extractor.extract(args.document, specs)
        for k, v in ex.fields.items():
            flag = "  <- check" if k in ex.uncertain() else ""
            print(f"{k:<18} {v.value!s:<28} conf {v.confidence:.2f}  (read: {v.raw!r}){flag}")
        print(f"{ex.seconds:.1f} s")
        return

    truth = json.loads((args.docs / "truth.json").read_text())
    fields = [k for k in next(iter(truth.values())) if k != "layout"]
    mapping = map_inputs(skill.parameters, fields)
    report = evaluate(extractor, specs, args.docs, mapping, args.limit)
    print(json.dumps(report, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/extraction_evaluation.json").write_text(json.dumps(report, indent=2))


def default_options(skill: Skill) -> dict[str, list[str]]:
    """Dropdown options read from the live page (see run.collect_options), cached in the skill."""
    return {
        s.parameter: s.element.options for s in skill.steps if s.parameter and s.element.options
    }


if __name__ == "__main__":
    main()
