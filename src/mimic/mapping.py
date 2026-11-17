"""Map the fields of a task record (e.g. one invoice) to the parameters of a skill.

The planner may name a parameter `vendor_name` while the data says `vendor`.
1. Deterministic: token overlap + a few synonyms, assigned greedily by best score.
2. LLM fallback (Ollama) for whatever is left. Computed once per batch, not per record.
"""

from __future__ import annotations

import re

SYNONYMS = {
    "supplier": "vendor",
    "seller": "vendor",
    "amount": "total",
    "price": "total",
    "cost": "total",
    "ref": "number",
    "reference": "number",
    "no": "number",
    "num": "number",
    "type": "category",
    "expense": "category",
    "billing": "date",
    "bill": "invoice",
}
MIN_SCORE = 0.34


def _tokens(text: str) -> set[str]:
    raw = {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t}
    return {SYNONYMS.get(t, t) for t in raw}


def _score(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def map_deterministic(parameters: list[str], fields: list[str]) -> dict[str, str]:
    pairs = sorted(
        ((_score(p, f), p, f) for p in parameters for f in fields),
        key=lambda x: x[0],
        reverse=True,
    )
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for score, param, field in pairs:
        if score < MIN_SCORE or param in mapping or field in used:
            continue
        mapping[param] = field
        used.add(field)
    return mapping


MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "mapping": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"parameter": {"type": "string"}, "field": {"type": "string"}},
                "required": ["parameter", "field"],
            },
        }
    },
    "required": ["mapping"],
}


def map_inputs(
    parameters: list[str], fields: list[str], client=None, model: str | None = None
) -> dict[str, str]:
    """Return {skill_parameter: record_field}. Raises KeyError if a parameter stays unmapped."""
    mapping = map_deterministic(parameters, fields)
    missing = [p for p in parameters if p not in mapping]
    free = [f for f in fields if f not in mapping.values()]

    if missing and free and client is not None and model:
        try:
            answer = client.chat_json(
                model,
                "Match each skill parameter to the data field that holds its value. "
                "Use only the given names. Answer in JSON.",
                f"Parameters: {missing}\nData fields: {free}",
                MAPPING_SCHEMA,
            )
            for item in answer.get("mapping", []):
                p, f = item.get("parameter"), item.get("field")
                if p in missing and f in free and f not in mapping.values():
                    mapping[p] = f
        except Exception as exc:  # noqa: BLE001
            print(f"  ! LLM mapping skipped ({type(exc).__name__}).")

    still_missing = [p for p in parameters if p not in mapping]
    if still_missing:
        raise KeyError(f"No data field found for parameters {still_missing} (fields: {fields})")
    return mapping


def apply_mapping(record: dict, mapping: dict[str, str]) -> dict:
    return {param: record[field] for param, field in mapping.items()}
