"""LedgerLite: a fake accounting app that Mimic learns to operate.

Two layouts simulate a UI redesign, to test self-healing:
  /?layout=v1  original layout
  /?layout=v2  fields reordered, button moved and relabelled, element ids changed

Chaos mode reproduces everyday web flakiness, at random, to train the failure predictor:
  /?chaos=0.3  each perturbation happens with probability ~0.3 on every page view
    overlay       a cookie banner covers the page until dismissed
    slow_submit   the submit button stays disabled while the page "loads"
    slow_field    one field is locked for a moment
    relabel       labels get a suffix (" *"), breaking exact label matching
    server_error  the save request fails with HTTP 500 (rarer)
"""

import random
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="LedgerLite sandbox")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

ENTRIES: list[dict] = []
FIELDS = ["vendor", "invoice_number", "date", "total", "category"]
_rng = random.Random()


def draw_chaos(level: float, rng: random.Random = _rng) -> dict:
    """Decide which perturbations happen on this page view."""
    level = max(0.0, min(level, 1.0))
    if level == 0:
        return {}
    return {
        "overlay": rng.random() < level,
        "overlay_escape": rng.random() < 0.5,  # some banners close with Escape, some do not
        "slow_submit_ms": rng.randint(900, 2500) if rng.random() < level else 0,
        "slow_field": rng.choice(FIELDS) if rng.random() < level / 2 else None,
        "slow_field_ms": rng.randint(900, 2500),
        "relabel": rng.random() < level,
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request, layout: str = "v1", chaos: float = 0.0, seed: int | None = None):
    layout = layout if layout in {"v1", "v2"} else "v1"
    rng = random.Random(seed) if seed is not None else _rng
    return templates.TemplateResponse(
        request,
        f"invoice_{layout}.html",
        {
            "entries": ENTRIES,
            "layout": layout,
            "chaos": draw_chaos(chaos, rng),
            "level": chaos,
            "seed": "" if seed is None else seed,
        },
    )


@app.post("/invoices")
def create_invoice(
    vendor: str = Form(...),
    invoice_number: str = Form(...),
    date: str = Form(...),
    total: float = Form(...),
    category: str = Form(...),
    layout: str = Form("v1"),
    chaos: float = Form(0.0),
    seed: str = Form(""),
):
    rng = random.Random(int(seed) * 7919) if seed.strip().lstrip("-").isdigit() else _rng
    if chaos and rng.random() < chaos / 4:
        return HTMLResponse("<h1>500 Internal Server Error</h1>", status_code=500)
    ENTRIES.append(
        {
            "vendor": vendor,
            "invoice_number": invoice_number,
            "date": date,
            "total": total,
            "category": category,
        }
    )
    return RedirectResponse(f"/?layout={layout}&saved=1", status_code=303)


@app.get("/api/entries")
def list_entries() -> list[dict]:
    """Ground truth used by the evaluation harness to check the agent's work."""
    return ENTRIES


@app.delete("/api/entries")
def reset_entries() -> dict:
    ENTRIES.clear()
    return {"cleared": True}
