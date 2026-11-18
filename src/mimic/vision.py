"""Vision grounding with a local VLM (Qwen2.5-VL via Ollama), using Set-of-Mark prompting.

Asking a 7B model for raw pixel coordinates is fragile. Instead:
  1. every candidate element on the current screen gets a numbered box drawn on the screenshot
  2. the model also sees the demo screenshot, with the element the user used boxed in red
  3. it answers with a NUMBER, which maps back to a real DOM element

"Vision proposes, DOM verifies": the agent still checks the effect of the action afterwards.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from mimic.dom import describe_text
from mimic.models import ElementInfo

SYSTEM = (
    "You are a UI grounding model. You compare screenshots of an application before and "
    "after a redesign and identify equivalent interface elements. Answer in JSON."
)

# Actions whose effect the agent cannot verify need a confident pick. A submit button is
# verified by the navigation it must trigger, so a best guess is safe to try there.
MIN_CONFIDENCE_UNVERIFIED = 0.6


def schema_for(n: int) -> dict:
    """Force a choice among the numbered candidates (no 'none' escape hatch)."""
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "choice": {"type": "integer", "enum": list(range(n))},
            "confidence": {"type": "number"},
        },
        "required": ["reason", "choice", "confidence"],
    }


COLORS = ["#0057ff", "#e67e00", "#00a65a", "#b000c8", "#d10000", "#008b8b", "#5a5a00"]


@dataclass
class VisionChoice:
    index: int
    reason: str = ""
    raw: dict | None = None
    confidence: float = 0.0


def _png(img: Image.Image) -> bytes:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _font(size: int = 18):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


def annotate_reference(screenshot: Path, el: ElementInfo) -> bytes:
    """Draw a red box around the element the user interacted with during the demo."""
    img = Image.open(screenshot).convert("RGB")
    b = el.bbox
    pad = 4
    ImageDraw.Draw(img).rectangle(
        [b.x - pad, b.y - pad, b.x + b.width + pad, b.y + b.height + pad], outline="red", width=4
    )
    return _png(img)


def annotate_candidates(current_png: bytes, candidates: list[ElementInfo]) -> bytes:
    """Set-of-Mark: a numbered box on every candidate element."""
    img = Image.open(io.BytesIO(current_png)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font()
    for i, c in enumerate(candidates):
        color = COLORS[i % len(COLORS)]
        b = c.bbox
        draw.rectangle(
            [b.x - 3, b.y - 3, b.x + b.width + 3, b.y + b.height + 3], outline=color, width=3
        )
        tag = str(i)
        tx, ty = b.x - 3, max(b.y - 26, 0)
        draw.rectangle([tx, ty, tx + 14 + 11 * len(tag), ty + 23], fill=color)
        draw.text((tx + 5, ty + 1), tag, fill="white", font=font)
    return _png(img)


class VisionLocator:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def choose(
        self,
        current_png: bytes,
        target: ElementInfo,
        action: str,
        candidates: list[ElementInfo],
        reference: Path | None = None,
    ) -> VisionChoice | None:
        if not candidates:
            return None
        what = describe_text(target, action)
        marked = annotate_candidates(current_png, candidates)
        listing = "\n".join(f"  {i}: {describe_text(c, action)}" for i, c in enumerate(candidates))

        if reference and reference.exists():
            images = [annotate_reference(reference, target), marked]
            intro = (
                f"Image 1 is the app BEFORE a redesign. The red box is the {what} the user used.\n"
                "Image 2 is the app NOW. Each candidate element has a numbered colored box.\n"
            )
        else:
            images = [marked]
            intro = (
                f"Before a redesign, the user used a {what}. The image is the app NOW; each "
                "candidate element has a numbered colored box.\n"
            )
        prompt = (
            intro + f"Candidates:\n{listing}\n"
            "The redesign changed labels, colors and positions ON PURPOSE: the right element "
            "will NOT have the same text or color. Pick the candidate that performs the same "
            "ACTION (for example: the button that submits the form, the field that holds the "
            "same information). You must pick one number. Then rate your confidence from "
            "0 (pure guess) to 1 (certain)."
        )
        answer = self.client.chat_json(
            self.model, SYSTEM, prompt, schema_for(len(candidates)), images=images
        )
        try:
            index = int(answer.get("choice", -1))
            confidence = max(0.0, min(1.0, float(answer.get("confidence", 0.0))))
        except (TypeError, ValueError):
            return VisionChoice(-1, "unparseable answer", answer)
        if not 0 <= index < len(candidates):
            return VisionChoice(-1, answer.get("reason", ""), answer)
        return VisionChoice(index, answer.get("reason", ""), answer, confidence)
