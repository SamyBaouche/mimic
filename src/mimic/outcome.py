"""Success signal: how a task looks when it worked, learned from where the demonstration ended.

On the sandbox the agent is graded against the app's own records. On a real site there is no
such API, so Mimic checks what a person would check: did we land on the same kind of page, and
does it show the same kind of confirmation message as at the end of the demo?

    demo ended on   /expenses/EXP-0001   "Expense report EXP-0001 submitted for Alice Martin."
    signal          /expenses/<an id>    "expense report <id> submitted for <anyone>."

Ids (anything with a digit) and the demo's own values become wildcards, the rest must match.
"""

from __future__ import annotations

import re
import time
from urllib.parse import parse_qs, urlsplit

from pydantic import BaseModel, Field

from mimic.models import Outcome

# Words that make a message look like a confirmation (English and French).
CONFIRM_WORDS = re.compile(
    r"success|saved|submitted|created|added|updated|recorded|sent|thank|confirm|complete|done|"
    r"enregistr|ajout|cr[ée]{2}|envoy|soumis|merci|termin|valid",
    re.IGNORECASE,
)


class SuccessSignal(BaseModel):
    path: str  # regex on the final URL path
    query_keys: list[str] = Field(default_factory=list)  # keys the final URL must carry
    text: str | None = None  # regex the final page must contain (whitespace-insensitive)
    example: str | None = None  # the demo's message, for humans

    def describe(self) -> str:
        return f'"{self.example}"' if self.example else f"a page like {self.path}"


def _wild_path(path: str, values: list[str]) -> str:
    parts = []
    for seg in path.split("/"):
        if seg in values:
            parts.append("[^/]+")
        elif any(c.isdigit() for c in seg):
            parts.append(r"[^/]*\d[^/]*")  # an id: another id, not any word (e.g. "review")
        else:
            parts.append(re.escape(seg))
    return "/".join(parts) or "/"


def _wild_text(text: str, values: list[str]) -> str:
    spans = []  # demo values first: they become ".+?"
    lower = text.lower()
    for v in sorted({v for v in values if len(v) >= 2}, key=len, reverse=True):
        start = lower.find(v.lower())
        while start != -1:
            if not any(a < start + len(v) and start < b for a, b in spans):
                spans.append((start, start + len(v)))
            start = lower.find(v.lower(), start + 1)
    out, pos = [], 0
    for a, b in sorted(spans):
        out.append(_escape_tokens(text[pos:a]))
        out.append(".+?")
        pos = b
    out.append(_escape_tokens(text[pos:]))
    return "".join(out)


def _escape_tokens(chunk: str) -> str:
    """Escape literal text; tokens with digits (ids, counts, dates) become wildcards."""
    pieces = []
    for token in re.split(r"(\s+)", chunk):
        if not token:
            continue
        if token.isspace():
            pieces.append(r"\s+")
        elif any(c.isdigit() for c in token) or "@" in token or "://" in token:
            # ids, amounts, dates, email addresses, links: they change from task to task
            trail = re.match(r"^(.*?)([.,;:!?)\"'»”]*)$", token)
            pieces.append(r"\S+?" + re.escape(trail.group(2)) if trail.group(2) else r"\S+")
        else:
            pieces.append(re.escape(token))
    return "".join(pieces)


def learn_signal(
    outcome: Outcome | None, start_url: str, values: list[str]
) -> SuccessSignal | None:
    if outcome is None:
        return None
    end = urlsplit(outcome.url)
    start = urlsplit(start_url)
    values = [str(v) for v in values if v not in (None, "")]
    message = next((t for t in outcome.texts if CONFIRM_WORDS.search(t)), None)
    keys = sorted(set(parse_qs(end.query)) - set(parse_qs(start.query)))
    if message is None and end.path == start.path and not keys:
        return None  # nothing distinguishes the end of the task from its start
    return SuccessSignal(
        path=_wild_path(end.path or "/", values),
        query_keys=keys,
        text=_wild_text(message, values) if message else None,
        example=message,
    )


def check_signal(page, signal: SuccessSignal, wait_s: float = 2.0) -> str | None:
    """None if the page shows success, else what is missing. Waits briefly for late toasts."""
    deadline = time.monotonic() + wait_s
    while True:
        problem = _problem(page, signal)
        if problem is None or time.monotonic() >= deadline:
            return problem
        page.wait_for_timeout(150)


def _problem(page, signal: SuccessSignal) -> str | None:
    url = urlsplit(page.url)
    if not re.fullmatch(signal.path, url.path or "/"):
        return f"ended on {url.path}, not on the page the demo ended on"
    missing = [k for k in signal.query_keys if k not in parse_qs(url.query)]
    if missing:
        return f"ended on {url.path} without {', '.join(missing)}"
    if signal.text:
        try:
            body = " ".join(page.inner_text("body").split())
        except Exception:  # noqa: BLE001
            body = ""
        if not re.search(signal.text, body, re.IGNORECASE):
            return f"no confirmation like {signal.describe()}"
    return None
