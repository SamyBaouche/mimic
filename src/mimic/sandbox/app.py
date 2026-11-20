"""LedgerLite: a fake accounting app that Mimic learns to operate.

WorkSuite adds a mailbox (supplier emails with invoice PDFs, replies) and an approvals app,
so one workflow can go from an email to the ledger to an approval and back to the sender.

Invoices (one page) and expense reports (two pages: form, review, confirmation) exercise
every action Mimic supports: text, dates, numbers, dropdown, radio, checkbox, file upload,
Enter in a search box, and a multi-page flow ending on a confirmation message.

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

import os
import random
import uuid
from email import policy
from email.parser import BytesParser
from email.utils import parseaddr
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
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


# Demo account for recording a sign-in (the password field is never recorded by Mimic).
DEMO_USERS = {"demo": "ledger-demo-2026"}
LOGINS: list[str] = []


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: int = 0):
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if DEMO_USERS.get(username) != password:
        return RedirectResponse("/login?error=1", status_code=303)
    LOGINS.append(username)
    return RedirectResponse("/?signed_in=1", status_code=303)


@app.get("/api/logins")
def list_logins() -> list[str]:
    return LOGINS


@app.get("/api/entries")
def list_entries() -> list[dict]:
    """Ground truth used by the evaluation harness to check the agent's work."""
    return ENTRIES


@app.delete("/api/entries")
def reset_entries() -> dict:
    ENTRIES.clear()
    return {"cleared": True}


# ---------------------------------------------------------------- expense reports

EXPENSE_TYPES = ["Travel", "Meals", "Office", "Software"]
EXPENSES: list[dict] = []
DRAFTS: dict[str, dict] = {}


@app.get("/expenses/new", response_class=HTMLResponse)
def expense_form(request: Request):
    return templates.TemplateResponse(request, "expense_new.html", {"types": EXPENSE_TYPES})


@app.post("/expenses/review", response_class=HTMLResponse)
async def expense_review(
    request: Request,
    employee: str = Form(...),
    spent_on: str = Form(...),
    amount: float = Form(...),
    kind: str = Form(...),
    billable: str = Form(""),
    receipt: UploadFile | None = File(None),  # noqa: B008
):
    name = receipt.filename if receipt is not None and receipt.filename else ""
    size = len(await receipt.read()) if name else 0
    draft = uuid.uuid4().hex
    DRAFTS[draft] = {
        "employee": employee,
        "spent_on": spent_on,
        "amount": amount,
        "kind": kind,
        "billable": billable == "yes",
        "receipt": name,
        "receipt_bytes": size,
    }
    return templates.TemplateResponse(
        request, "expense_review.html", {"d": DRAFTS[draft], "draft": draft}
    )


@app.post("/expenses/submit")
def expense_submit(draft: str = Form(...)):
    data = DRAFTS.pop(draft, None)
    if data is None:
        return HTMLResponse("<h1>This draft has expired</h1>", status_code=410)
    expense = {"id": f"EXP-{len(EXPENSES) + 1:04d}", **data}
    EXPENSES.append(expense)
    return RedirectResponse(f"/expenses/{expense['id']}", status_code=303)


@app.get("/expenses", response_class=HTMLResponse)
def expense_list(request: Request, q: str = ""):
    rows = [
        e for e in EXPENSES if q.lower() in e["employee"].lower() or q.lower() in e["id"].lower()
    ]
    return templates.TemplateResponse(request, "expense_list.html", {"rows": rows, "q": q})


@app.get("/expenses/{expense_id}", response_class=HTMLResponse)
def expense_done(request: Request, expense_id: str):
    match = next((e for e in EXPENSES if e["id"] == expense_id), None)
    if match is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return templates.TemplateResponse(request, "expense_done.html", {"e": match})


@app.get("/api/expenses")
def list_expenses() -> list[dict]:
    """Ground truth for tests only (Mimic grades expense runs with the confirmation message)."""
    return EXPENSES


@app.delete("/api/expenses")
def reset_expenses() -> dict:
    EXPENSES.clear()
    return {"cleared": True}


# ---------------------------------------------------------------- WorkSuite: mailbox

MAIL_DIR = Path(os.environ.get("MIMIC_MAIL_DIR", "sandbox/data/mail"))
SENT: list[dict] = []


def set_mail_dir(path: Path) -> None:
    global MAIL_DIR
    MAIL_DIR = Path(path)


def read_message(path: Path) -> dict:
    msg = BytesParser(policy=policy.default).parse(path.open("rb"))
    name, address = parseaddr(str(msg["From"]))
    body = msg.get_body(preferencelist=("plain",))
    return {
        "id": path.stem,
        "sender": str(msg["From"]),
        "sender_name": name or address,
        "sender_address": address,
        "to": str(msg["To"]),
        "subject": str(msg["Subject"]),
        "date": str(msg["Date"]),
        "body": body.get_content() if body else "",
        "attachments": [a.get_filename() for a in msg.iter_attachments()],
        "_msg": msg,
    }


def _messages() -> list[dict]:
    return [read_message(p) for p in sorted(MAIL_DIR.glob("*.eml"))]


def _message(message_id: str) -> dict | None:
    path = MAIL_DIR / f"{message_id}.eml"
    return read_message(path) if path.is_file() and path.parent == MAIL_DIR else None


@app.get("/mail", response_class=HTMLResponse)
def inbox(request: Request):
    return templates.TemplateResponse(
        request, "mail_inbox.html", {"messages": _messages(), "app_title": "Mail"}
    )


@app.get("/mail/{message_id}", response_class=HTMLResponse)
def open_message(request: Request, message_id: str):
    m = _message(message_id)
    if m is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return templates.TemplateResponse(request, "mail_message.html", {"m": m, "app_title": "Mail"})


@app.get("/mail/{message_id}/attachments/{name}")
def attachment(message_id: str, name: str):
    m = _message(message_id)
    for part in m["_msg"].iter_attachments() if m else []:
        if part.get_filename() == name:
            return Response(part.get_content(), media_type=part.get_content_type())
    return Response(status_code=404)


@app.get("/mail/{message_id}/reply", response_class=HTMLResponse)
def reply_form(request: Request, message_id: str):
    m = _message(message_id)
    if m is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return templates.TemplateResponse(request, "mail_reply.html", {"m": m, "app_title": "Mail"})


@app.post("/mail/{message_id}/reply")
def send_reply(
    message_id: str, to: str = Form(...), subject: str = Form(...), message: str = Form("")
):
    SENT.append({"in_reply_to": message_id, "to": to, "subject": subject, "message": message})
    return RedirectResponse(f"/mail/{message_id}/sent", status_code=303)


@app.get("/mail/{message_id}/sent", response_class=HTMLResponse)
def reply_sent(request: Request, message_id: str):
    sent = next((s for s in reversed(SENT) if s["in_reply_to"] == message_id), None)
    if sent is None:
        return HTMLResponse("<h1>Nothing sent</h1>", status_code=404)
    return templates.TemplateResponse(request, "mail_sent.html", {"s": sent, "app_title": "Mail"})


@app.get("/api/mail/sent")
def list_sent() -> list[dict]:
    return SENT


@app.delete("/api/mail/sent")
def reset_sent() -> dict:
    SENT.clear()
    return {"cleared": True}


# ---------------------------------------------------------------- WorkSuite: approvals

APPROVERS = ["Finance manager", "Controller", "CFO"]
APPROVALS: list[dict] = []


@app.get("/approvals/new", response_class=HTMLResponse)
def approval_form(request: Request):
    return templates.TemplateResponse(
        request, "approval_new.html", {"approvers": APPROVERS, "app_title": "Approvals"}
    )


@app.post("/approvals")
def request_approval(
    reference: str = Form(...),
    amount: float = Form(...),
    approver: str = Form(...),
    reason: str = Form(""),
):
    item = {
        "id": f"APR-{len(APPROVALS) + 1:04d}",
        "reference": reference,
        "amount": amount,
        "approver": approver,
        "reason": reason,
    }
    APPROVALS.append(item)
    return RedirectResponse(f"/approvals/{item['id']}", status_code=303)


@app.get("/approvals/{approval_id}", response_class=HTMLResponse)
def approval_done(request: Request, approval_id: str):
    item = next((a for a in APPROVALS if a["id"] == approval_id), None)
    if item is None:
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    return templates.TemplateResponse(
        request, "approval_done.html", {"a": item, "app_title": "Approvals"}
    )


@app.get("/api/approvals")
def list_approvals() -> list[dict]:
    return APPROVALS


@app.delete("/api/approvals")
def reset_approvals() -> dict:
    APPROVALS.clear()
    return {"cleared": True}
