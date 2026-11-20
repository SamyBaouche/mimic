"""WorkSuite: an email with an invoice PDF -> ledger -> approval if needed -> reply.
Three skills, each learned from one demo in its own app, chained by a workflow."""

import json
import textwrap

import httpx
import pytest
from playwright.sync_api import sync_playwright

from mimic.conditions import ConditionError, compile_condition, evaluate, parse_rule
from mimic.extract import Extracted, Extraction
from mimic.recorder import Recorder, launch_browser
from mimic.sandbox import app as sandbox_app
from mimic.sandbox.generate_mail import generate_mail
from mimic.skills import draft_skill
from mimic.workflow import ConsoleReview, fill, load_workflow, run_workflow

KNOWN = {"invoice.total", "invoice.vendor", "email.subject", "email.id"}


# ---------------------------------------------------------------- rules


@pytest.mark.parametrize(
    "rule, context, expected",
    [
        ("invoice.total > 2500", {"invoice": {"total": "3389.96"}}, True),
        ("invoice.total > 2500", {"invoice": {"total": "1 530,94 $"}}, False),
        ("invoice.vendor == 'atlas analytics'", {"invoice": {"vendor": "Atlas  Analytics"}}, True),
        ("'urgent' in email.subject", {"email": {"subject": "URGENT: invoice"}}, True),
        (
            "invoice.total >= 100 and not 'test' in email.subject",
            {"invoice": {"total": "100"}, "email": {"subject": "Invoice"}},
            True,
        ),
    ],
)
def test_rules(rule, context, expected):
    assert evaluate(parse_rule(rule, KNOWN), context) is expected


@pytest.mark.parametrize(
    "text",
    ["__import__('os').system('x')", "open('f')", "invoice.total.__class__", "[1, 2]", "x"],
)
def test_rules_never_run_code_or_invent_fields(text):
    with pytest.raises(ConditionError):
        parse_rule(text, KNOWN)


class FakeLLM:
    def __init__(self, rule):
        self.rule, self.prompts = rule, []

    def chat_json(self, model, system, user, schema, images=None):
        self.prompts.append(user)
        return {"rule": self.rule}


def test_plain_words_are_translated_and_checked():
    llm = FakeLLM("invoice.total > 2500")
    rule, tree, translated = compile_condition("the total is over 2 500 $", KNOWN, llm, "m")
    assert (rule, translated) == ("invoice.total > 2500", True)
    assert "invoice.total" in llm.prompts[0]
    # A translation that invents a field (or writes code) is refused, not run.
    with pytest.raises(ConditionError):
        compile_condition("the tax is high", KNOWN, FakeLLM("invoice.tax > 10"), "m")
    with pytest.raises(ConditionError, match="Write it like"):
        compile_condition("the total is big", KNOWN)  # no model to translate


def test_templates():
    ctx = {"invoice": {"invoice_number": "INV-1", "total": "12.50"}, "email": {"id": "mail-1"}}
    assert fill("Invoice {invoice.invoice_number} ({invoice.total} $)", ctx) == (
        "Invoice INV-1 (12.50 $)"
    )
    with pytest.raises(KeyError):
        fill("{invoice.tax}", ctx)


# ---------------------------------------------------------------- end to end


class TruthReader:
    """Stands in for the vision model: returns what each attached invoice really says,
    and marks the date of one of them as uncertain (like a blurry scan)."""

    model = "truth"

    def __init__(self, truth: dict):
        self.by_file = {f"{v['invoice_number']}.pdf": v for v in truth.values()}

    def extract(self, path, specs):
        inv = self.by_file[path.name]
        fields = {
            s.parameter: Extracted(str(inv[s.parameter]), 0.95 if s.parameter != "date" else 0.4)
            for s in specs
        }
        return Extraction(document=path.name, fields=fields, seconds=0.0)


def _record(url, out, name, demo):
    recorder = Recorder(name, url, out)
    with sync_playwright() as p:
        try:
            browser = launch_browser(p, headless=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"No Chromium available: {exc}")
        context = browser.new_context()
        recorder.attach(context)
        page = context.new_page()
        page.goto(url)
        demo(page)
        page.wait_for_timeout(250)
        recorder.capture_outcome(page)
        browser.close()
    return draft_skill(recorder.recording)


@pytest.fixture(scope="module")
def suite(sandbox_server, tmp_path_factory):
    mail = tmp_path_factory.mktemp("mail")
    truth = generate_mail(mail, n=6, seed=11)
    sandbox_app.set_mail_dir(mail)
    rec = tmp_path_factory.mktemp("rec")

    def ledger(page):
        page.fill("#vendor", "Demo Vendor")
        page.fill("#invoice_number", "INV-DEMO")
        page.fill("#date", "2026-11-01")
        page.fill("#total", "10.00")
        page.select_option("#category", "Software")
        page.click("#save-btn")
        page.wait_for_url("**saved=1")

    def approval(page):
        page.fill("#reference", "INV-DEMO")
        page.fill("#amount", "9000")
        page.select_option("#approver", "CFO")
        page.fill("#reason", "Over the limit")
        page.click("#request")
        page.wait_for_url("**/approvals/APR-*")

    def reply(page):
        page.fill("#message", "Hello, we received your invoice.")
        page.click("#send")
        page.wait_for_url("**/sent")

    skills = {
        "enter_invoice": _record(sandbox_server, rec / "a", "enter_invoice", ledger),
        "request_approval": _record(
            sandbox_server + "approvals/new", rec / "b", "request_approval", approval
        ),
        "reply_to_supplier": _record(
            sandbox_server + "mail/mail-0001/reply", rec / "c", "reply_to_supplier", reply
        ),
    }
    for path in ("api/entries", "api/approvals", "api/mail/sent"):
        httpx.delete(sandbox_server + path)
    return mail, truth, skills


WORKFLOW = """
name: process_invoice_emails
for_each:
  emails: {mail}
steps:
  - read: attachment
    fields_of: enter_invoice
    as: invoice
  - run: enter_invoice
  - run: request_approval
    if: the total is over 2 500 $
    with:
      reference: "{{invoice.invoice_number}}"
      amount: "{{invoice.total}}"
      reason: "Invoice from {{invoice.vendor}} is over the 2 500 $ limit"
  - run: reply_to_supplier
    url: /mail/{{email.id}}/reply
    with:
      message: "Hello, we received invoice {{invoice.invoice_number}} ({{invoice.total}} $)."
"""


def test_email_to_ledger_to_approval_to_reply(suite, sandbox_server, tmp_path):
    mail, truth, skills = suite
    path = tmp_path / "wf.yml"
    path.write_text(textwrap.dedent(WORKFLOW).format(mail=mail))
    wf = load_workflow(path)
    shown = []
    review = ConsoleReview(print_fn=shown.append, auto=True)

    report = run_workflow(
        wf, skills, TruthReader(truth), review, client=FakeLLM("invoice.total > 2500"), model="m"
    )

    big = [v for v in truth.values() if v["total"] > 2500]
    assert report["done"] == report["processed"] == 6, json.dumps(report["results"], default=str)
    assert report["steps_skipped_by_rule"] == 6 - len(big)
    assert 0 < len(big) < 6  # the rule really splits the batch

    entries = httpx.get(sandbox_server + "api/entries").json()
    assert {e["invoice_number"] for e in entries} == {v["invoice_number"] for v in truth.values()}
    approvals = httpx.get(sandbox_server + "api/approvals").json()
    assert sorted(a["reference"] for a in approvals) == sorted(v["invoice_number"] for v in big)
    assert all(a["approver"] == "CFO" for a in approvals)  # not in the workflow: from the demo
    sent = httpx.get(sandbox_server + "api/mail/sent").json()
    assert len(sent) == 6
    for s in sent:
        inv = truth[s["in_reply_to"]]
        assert s["to"].startswith("billing@")
        assert inv["invoice_number"] in s["message"] and inv["invoice_number"] in s["subject"]

    plans = "\n".join(shown)
    assert "skipped: invoice.total > 2500 is false" in plans
    assert "approver from the demo" in plans
    assert "check: invoice.date" in plans  # the uncertain field is shown before acting


def test_a_failed_step_stops_that_email_only(suite, sandbox_server, tmp_path):
    mail, truth, skills = suite
    broken = dict(skills)
    reply = skills["reply_to_supplier"]
    broken["reply_to_supplier"] = reply.model_copy(
        update={
            "steps": [
                s.model_copy(
                    update={
                        "element": s.element.model_copy(
                            update={
                                "selector": "#nope",
                                "label": "Nope",
                                "name": "nope",
                                "id": None,
                            }
                        )
                    }
                )
                if s.action == "fill"
                else s
                for s in reply.steps
            ]
        }
    )
    path = tmp_path / "wf.yml"
    path.write_text(textwrap.dedent(WORKFLOW).format(mail=mail))
    report = run_workflow(
        load_workflow(path),
        broken,
        TruthReader(truth),
        limit=2,
        client=FakeLLM("invoice.total > 2500"),
        model="m",
        log=lambda *_: None,
    )
    assert report["done"] == 0 and report["processed"] == 2
    first = report["results"][0]
    assert first["steps"][0] == {**first["steps"][0], "skill": "enter_invoice", "status": "done"}
    assert first["error"].startswith("reply_to_supplier:")
