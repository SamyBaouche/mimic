"""Data models shared across the recorder, planner and agent."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class ElementInfo(BaseModel):
    """Everything we know about the element the user interacted with.

    Several redundant descriptors are kept on purpose: when the UI changes,
    the agent can fall back from selector -> label -> visual position.
    """

    tag: str
    selector: str
    id: str | None = None
    name: str | None = None
    type: str | None = None
    label: str | None = None
    text: str | None = None
    placeholder: str | None = None
    options: list[str] = Field(default_factory=list)  # dropdown choices, when known
    bbox: BBox


class RecordedEvent(BaseModel):
    index: int
    kind: Literal["click", "fill", "select", "check", "choose", "upload", "press"]
    element: ElementInfo
    value: str | None = None
    secret: bool = False  # password-like field: the value is never recorded
    url: str
    timestamp_ms: int
    screenshot: str | None = None


class Outcome(BaseModel):
    """Where the demonstration ended: what success looks like (see skills.SuccessSignal)."""

    url: str
    texts: list[str] = Field(default_factory=list)  # confirmation-like messages on screen


class Recording(BaseModel):
    name: str
    start_url: str
    viewport: dict = Field(default_factory=lambda: {"width": 1280, "height": 800})
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    events: list[RecordedEvent] = Field(default_factory=list)
    outcome: Outcome | None = None


TRUE_WORDS = {"true", "yes", "y", "1", "x", "on", "oui", "checked", "vrai"}


def as_bool(value) -> bool:
    """Checkbox values from a sheet: true / yes / x / oui... (agent and review)."""
    return str(value).strip().lower() in TRUE_WORDS
