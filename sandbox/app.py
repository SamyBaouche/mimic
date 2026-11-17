"""LedgerLite: a fake accounting app that Mimic learns to operate.

Two layouts simulate a UI redesign, to test self-healing:
  /?layout=v1  original layout
  /?layout=v2  fields reordered, button moved and relabelled, element ids changed
"""

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="LedgerLite sandbox")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

ENTRIES: list[dict] = []


@app.get("/", response_class=HTMLResponse)
def index(request: Request, layout: str = "v1"):
    layout = layout if layout in {"v1", "v2"} else "v1"
    return templates.TemplateResponse(
        request, f"invoice_{layout}.html", {"entries": ENTRIES, "layout": layout}
    )


@app.post("/invoices")
def create_invoice(
    vendor: str = Form(...),
    invoice_number: str = Form(...),
    date: str = Form(...),
    total: float = Form(...),
    category: str = Form(...),
    layout: str = Form("v1"),
):
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
