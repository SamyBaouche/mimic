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
    bbox: BBox


class RecordedEvent(BaseModel):
    index: int
    kind: Literal["click", "fill", "select"]
    element: ElementInfo
    value: str | None = None
    url: str
    timestamp_ms: int
    screenshot: str | None = None


class Recording(BaseModel):
    name: str
    start_url: str
    viewport: dict = Field(default_factory=lambda: {"width": 1280, "height": 800})
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    events: list[RecordedEvent] = Field(default_factory=list)
