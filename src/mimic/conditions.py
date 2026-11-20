"""Conditions in workflows: "only request an approval if the total is over 2 500".

You can write a rule (`invoice.total > 2500`) or plain words ("the total is over 2 500 $").
Plain words are translated once by the local LLM into a rule, which is checked and shown to
you before anything runs. Rules are evaluated safely: comparisons, and / or / not, `in`,
numbers, text and fields. Nothing else (no function calls, no code).
"""

from __future__ import annotations

import ast
import operator
import re

ALLOWED = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare,
    ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn,
    ast.Constant, ast.Name, ast.Attribute, ast.Load,
)  # fmt: skip
COMPARE = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


class ConditionError(ValueError):
    pass


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def parse_rule(text: str, known: set[str] | None = None) -> ast.Expression:
    """Parse a rule; ConditionError if it is not one (or uses an unknown field)."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ConditionError(f"not a rule: {text!r}") from exc
    inner = set()  # Name / Attribute nodes that are part of a longer field name
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED):
            raise ConditionError(f"not allowed in a rule: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            inner.add(id(node.value))
            if node.attr.startswith("_"):
                raise ConditionError("field names cannot start with _")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ConditionError("field names cannot start with _")
    fields = [
        _dotted(n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Name, ast.Attribute)) and id(n) not in inner
    ]
    if any(f is None for f in fields):
        raise ConditionError("fields look like invoice.total")
    if known is not None:
        unknown = sorted({f for f in fields if f not in known})
        if unknown:
            raise ConditionError(f"unknown field(s): {', '.join(unknown)}")
    return tree


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = re.sub(r"[\s$€£]", "", str(value)).replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _resolve(name: str, context: dict):
    value = context
    for part in name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConditionError(f"no value for {name}")
        value = value[part]
    return value


def _value(node, context):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_value(node.operand, context)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _truth(node.operand, context)
    if isinstance(node, (ast.Name, ast.Attribute)):
        return _resolve(_dotted(node), context)
    return _truth(node, context)


def _compare(op, left, right) -> bool:
    if isinstance(op, (ast.In, ast.NotIn)):
        found = str(left).lower() in str(right).lower()
        return found if isinstance(op, ast.In) else not found
    a, b = _number(left), _number(right)
    if a is not None and b is not None:
        return COMPARE[type(op)](a, b)
    if isinstance(op, (ast.Eq, ast.NotEq)):
        same = " ".join(str(left).lower().split()) == " ".join(str(right).lower().split())
        return same if isinstance(op, ast.Eq) else not same
    raise ConditionError(f"cannot compare {left!r} and {right!r}")


def _truth(node, context) -> bool:
    if isinstance(node, ast.Expression):
        return _truth(node.body, context)
    if isinstance(node, ast.BoolOp):
        values = (_truth(v, context) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.Compare):
        left = _value(node.left, context)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _value(comparator, context)
            if not _compare(op, left, right):
                return False
            left = right
        return True
    return bool(_value(node, context))


def evaluate(tree: ast.Expression, context: dict) -> bool:
    return _truth(tree, context)


# ---------------------------------------------------------------- plain words

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"rule": {"type": "string"}},
    "required": ["rule"],
}
SYSTEM = """You turn a condition written in plain words into a rule.
A rule uses only: field names from the list, numbers, 'text in quotes', the operators
> >= < <= == != in, and the words and, or, not. Write numbers without spaces or currency.
Examples: "the total is over 2 500 $" -> invoice.total > 2500
"the supplier is Atlas or the subject mentions urgent" ->
invoice.vendor == 'Atlas' or 'urgent' in email.subject
Answer in JSON."""


def translate(words: str, known: set[str], client, model: str) -> str:
    """Plain words -> a rule, checked before it is returned."""
    answer = client.chat_json(
        model,
        SYSTEM,
        f"Fields: {', '.join(sorted(known))}\nCondition: {words}",
        TRANSLATE_SCHEMA,
    )
    rule = str(answer.get("rule", "")).strip()
    parse_rule(rule, known)  # raises if the model invented a field or wrote code
    return rule


def compile_condition(text: str, known: set[str], client=None, model: str | None = None):
    """(rule text, parsed rule, translated?) for a condition written either way."""
    try:
        return text, parse_rule(text, known), False
    except ConditionError:
        if client is None or not model:
            raise ConditionError(
                f"'{text}' is not a rule, and no model is available to translate it. "
                f"Write it like: invoice.total > 2500"
            ) from None
    rule = translate(text, known, client, model)
    return rule, parse_rule(rule, known), True
