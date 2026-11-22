"""Workflows: chain learned skills across apps, for every email (or every row), with conditions.

    name: process_invoice_emails
    for_each:
      emails: sandbox/data/mail              # a folder of .eml files (or rows: invoices.xlsx)
    steps:
      - read: attachment                     # the PDF attached to the email...
        fields_of: enter_invoice             # ...read for the values this skill needs
        as: invoice
      - run: enter_invoice                   # inputs found by name in {invoice}
      - run: request_approval
        if: the total is over 2 500 $        # plain words, or a rule: invoice.total > 2500
        with:
          reference: "{invoice.invoice_number}"
          amount: "{invoice.total}"
      - run: reply_to_supplier
        url: /mail/{email.id}/reply          # where this step starts, for this email
        with:
          message: "Hello, we received invoice {invoice.invoice_number}."

Each skill was learned from one demonstration in its own app. The workflow only says in which
order to run them, with which values, and when. Every email gets one plan you can approve;
every step is checked against the confirmation its demo ended on.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mimic.conditions import ConditionError, compile_condition, evaluate
from mimic.skills import Skill

TEMPLATE = re.compile(r"\{([A-Za-z_][\w.]*)\}")
EMAIL_FIELDS = ["id", "sender", "address", "name", "subject", "date", "body", "attachment"]


# ---------------------------------------------------------------- the file


class ReadStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    read: str = "attachment"  # a field holding a file path (email.attachment, row.file...)
    fields_of: str
    as_: str = Field("document", alias="as")


class RunStep(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    run: str
    with_: dict[str, str] = Field(default_factory=dict, alias="with")
    if_: str | None = Field(None, alias="if")
    url: str | None = None


class Workflow(BaseModel):
    name: str
    description: str = ""
    for_each: dict[str, str]
    steps: list[ReadStep | RunStep]

    @model_validator(mode="after")
    def _check(self):
        if len(self.for_each) != 1 or next(iter(self.for_each)) not in ("emails", "rows"):
            raise ValueError("for_each needs one source: emails: FOLDER or rows: FILE")
        if not any(isinstance(s, RunStep) for s in self.steps):
            raise ValueError("a workflow needs at least one 'run:' step")
        return self

    @property
    def source(self) -> tuple[str, Path]:
        kind, path = next(iter(self.for_each.items()))
        return kind, Path(path)


def load_workflow(path: Path) -> Workflow:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    steps = []
    for raw in data.get("steps", []):
        steps.append(ReadStep.model_validate(raw) if "read" in raw else RunStep.model_validate(raw))
    return Workflow.model_validate({**data, "steps": steps})


# ---------------------------------------------------------------- values


def fill(template: str, context: dict) -> str:
    """ "Invoice {invoice.invoice_number}" -> "Invoice INV-2026..." (KeyError if unknown)."""

    def one(match):
        value = context
        for part in match.group(1).split("."):
            if not isinstance(value, dict) or part not in value:
                raise KeyError(match.group(1))
            value = value[part]
        return "" if value is None else str(value)

    return TEMPLATE.sub(one, template)


def known_fields(wf: Workflow, skills: dict[str, Skill], row_fields: list[str]) -> set[str]:
    kind, _ = wf.source
    known = {f"email.{f}" for f in EMAIL_FIELDS} if kind == "emails" else set()
    known |= {f"row.{f}" for f in row_fields}
    for step in wf.steps:
        if isinstance(step, ReadStep):
            known |= {f"{step.as_}.{p}" for p in skills[step.fields_of].parameters}
    return known


def lint_workflow(wf: Workflow, skills: dict[str, Skill]) -> list[str]:
    """What would make every item fail, found before anything runs: an input a skill does not
    have, or a {field} nothing provides. (row.* fields are only known once the file is read.)"""
    import difflib

    known = known_fields(wf, skills, [])
    problems = []
    for step in wf.steps:
        if not isinstance(step, RunStep):
            continue
        params = skills[step.run].parameters
        for key in step.with_:
            if key in params:
                continue
            close = difflib.get_close_matches(key, params, n=1, cutoff=0.0)
            fix = (
                f" Rename it: mimic skills edit {step.run} --rename {close[0]}={key}"
                if close
                else ""
            )
            problems.append(
                f"{step.run} has no input '{key}' (its inputs: {', '.join(params) or 'none'}).{fix}"
            )
        for template in [*step.with_.values(), step.url or ""]:
            for ref in TEMPLATE.findall(template):
                if ref.startswith("row.") or ref in known:
                    continue
                prefix = ref.split(".", 1)[0]
                same = sorted(k for k in known if k.startswith(prefix + "."))
                close = difflib.get_close_matches(ref, same, n=1, cutoff=0.0)
                hint = f" Did you mean {{{close[0]}}}?" if close else ""
                problems.append(f"{step.run}: nothing provides {{{ref}}}.{hint}")
    return problems


def find_inputs(skill: Skill, step: RunStep, context: dict) -> tuple[dict, list[str], list[str]]:
    """Inputs for one skill: `with:` first, then the same name in what was read, then the row,
    then the demo's own value. Returns (inputs, taken_from_demo, missing)."""
    inputs, from_demo, missing = {}, [], []
    # Not the email itself: a skill's "date" must not silently become the email's date.
    sources = [v for k, v in context.items() if k != "email" and isinstance(v, dict)]
    demo = {
        s.parameter: s.value
        for s in skill.steps
        if s.parameter and s.value is not None and not s.secret and s.action != "upload"
    }
    for p in skill.parameters:
        if p in skill.secrets:
            continue
        if p in step.with_:
            inputs[p] = fill(step.with_[p], context)
            continue
        found = next((src[p] for src in sources if src.get(p) not in (None, "")), None)
        if found is not None:
            inputs[p] = found
        elif p in demo:
            inputs[p] = demo[p]
            from_demo.append(p)
        else:
            missing.append(p)
    return inputs, from_demo, missing


# ---------------------------------------------------------------- planning


@dataclass
class PlannedStep:
    skill: str
    runs: bool
    why: str = ""  # why it is skipped / where values come from
    actions: list[dict] = field(default_factory=list)
    url: str | None = None


@dataclass
class Item:
    id: str
    title: str
    context: dict
    flags: dict = field(default_factory=dict)
    plan: list[PlannedStep] = field(default_factory=list)
    error: str | None = None


def plan_item(item: Item, wf: Workflow, skills: dict[str, Skill], rules: dict) -> None:
    for i, step in enumerate(wf.steps):
        if not isinstance(step, RunStep):
            continue
        skill = skills[step.run]
        if step.if_:
            text, tree = rules[i]
            try:
                ok = evaluate(tree, item.context)
            except ConditionError as exc:
                item.error = f"{step.run}: {exc}"
                return
            if not ok:
                item.plan.append(PlannedStep(step.run, False, f"skipped: {text} is false"))
                continue
        try:
            inputs, from_demo, missing = find_inputs(skill, step, item.context)
        except KeyError as exc:
            item.error = f"{step.run}: no value for {{{exc.args[0]}}}"
            return
        if missing:
            item.error = f"{step.run}: no value for {', '.join(missing)}"
            return
        base = start_url(skill)
        url = urljoin(base, fill(step.url, item.context)) if step.url else base
        why = f"{', '.join(from_demo)} from the demo" if from_demo else ""
        if step.if_:
            why = f"because {rules[i][0]}" + (f"; {why}" if why else "")
        item.plan.append(PlannedStep(step.run, True, why, skill.render(inputs), url))


def format_item(index: int, total: int, item: Item) -> str:
    from mimic.approval import display_value

    lines = [f"\n{index}/{total}  {item.title}"]
    for key, note in item.flags.items():
        lines.append(f"  check: {key} ({note})")
    if item.error:
        lines.append(f"  cannot run: {item.error}")
    for n, step in enumerate(item.plan, 1):
        if not step.runs:
            lines.append(f"  {n}. {step.skill:<20} {step.why}")
            continue
        lines.append(f"  {n}. {step.skill}" + (f"   ({step.why})" if step.why else ""))
        for a in step.actions:
            el = a["element"]
            label = el.label or el.text or el.name or el.selector
            if a["action"] == "click":
                lines.append(f"       then click '{label}'")
            elif a["action"] == "press":
                lines.append(f"       then press {a['value']} in '{label}'")
            else:
                lines.append(f"       {label:<18} <- {display_value(a)}")
    return "\n".join(lines)


class ConsoleReview:
    """Approve each email's plan: [A]pprove, [S]kip, [T] approve all, [Q]uit."""

    def __init__(self, input_fn=input, print_fn=print, auto: bool = False):
        self.input, self.print, self.auto = input_fn, print_fn, auto

    def __call__(self, text: str, **_) -> str:
        self.print(text)
        if self.auto:
            return "approve"
        while True:
            answer = self.input("[A]pprove  [S]kip  [T] approve all  [Q]uit > ").strip().lower()
            if answer in ("", "a"):
                return "approve"
            if answer == "t":
                self.auto = True
                return "approve"
            if answer == "s":
                return "skip"
            if answer == "q":
                return "quit"


# ---------------------------------------------------------------- running


def load_items(wf: Workflow, attachments_dir: Path, limit: int | None) -> list[Item]:
    kind, path = wf.source
    if kind == "emails":
        from mimic.mailbox import read_mailbox

        emails = read_mailbox(path, attachments_dir)[:limit]
        return [Item(e.id, e.subject, {"email": e.fields()}) for e in emails]
    from mimic.tables import load_records

    rows = load_records(path)[:limit]
    return [Item(str(i), str(next(iter(r.values()), i)), {"row": r}) for i, r in enumerate(rows, 1)]


def compile_rules(wf: Workflow, known: set[str], client=None, model=None, log=print) -> dict:
    rules = {}
    for i, step in enumerate(wf.steps):
        if isinstance(step, RunStep) and step.if_:
            text, tree, translated = compile_condition(step.if_, known, client, model)
            if translated:
                log(f'  if "{step.if_}"  ->  {text}')
            rules[i] = (text, tree)
    return rules


def start_url(skill: Skill) -> str:
    """Where a skill starts. Skills learned before 0.3 did not store it: the sandbox, as in
    `mimic run`."""
    from mimic.config import get_settings

    return skill.start_url or get_settings().sandbox_url


def check_apps_answer(skills: dict[str, Skill]) -> None:
    """Before reading any email: every app the skills start in must answer, or nothing can run."""
    from urllib.parse import urlsplit

    import httpx

    for base in sorted({f"{urlsplit(u).scheme}://{urlsplit(u).netloc}/"
                        for u in map(start_url, skills.values())}):  # fmt: skip
        try:
            httpx.get(base, timeout=5, follow_redirects=True)
        except httpx.HTTPError:
            raise SystemExit(
                f"Nothing answers at {base}. Start the app first and keep it running in another "
                "terminal (for the demo apps: mimic sandbox)."
            ) from None


def dropdown_choices(wf: Workflow, skills: dict[str, Skill], log=print) -> dict[str, dict]:
    """The allowed values of each dropdown the read steps fill, so a document's text is matched
    to a real choice. Kept in the skill since 0.3; for older skills, read from the live app."""
    from mimic.extract import default_options

    out = {}
    for step in wf.steps:
        if not isinstance(step, ReadStep) or step.fields_of in out:
            continue
        skill = skills[step.fields_of]
        options = default_options(skill)
        missing = [s.parameter for s in skill.steps if s.action == "select" and s.parameter
                   and s.parameter not in options]  # fmt: skip
        if missing:
            try:
                from mimic.run import collect_options

                options = collect_options(skill, start_url(skill)) | options
            except Exception as exc:  # noqa: BLE001 - app not running
                log(f"  ! Could not read the choices of {', '.join(missing)} from the app "
                    f"({type(exc).__name__}): they will be read as free text.")  # fmt: skip
        out[step.fields_of] = options
    return out


def read_documents(item: Item, wf: Workflow, skills: dict[str, Skill], extractor,
                   choices: dict | None = None) -> None:  # fmt: skip
    from mimic.extract import LOW_CONFIDENCE, default_options, field_specs

    if item.error is not None:
        return
    for step in wf.steps:
        if not isinstance(step, ReadStep):
            continue
        source = step.read if "." in step.read else f"email.{step.read}"
        try:
            path = Path(fill("{" + source + "}", item.context))
        except KeyError:
            item.error = f"nothing to read at {source}"
            return
        if not path.is_file():
            item.error = f"no file to read ({source})"
            return
        skill = skills[step.fields_of]
        options = (choices or {}).get(step.fields_of) or default_options(skill)
        ex = extractor.extract(path, field_specs(skill, options))
        item.context[step.as_] = {**ex.inputs(), "file": str(path)}
        for name, got in ex.fields.items():
            if got.value is None:
                item.flags[f"{step.as_}.{name}"] = got.note or "not found"
            elif got.confidence < LOW_CONFIDENCE:
                item.flags[f"{step.as_}.{name}"] = got.note or f"confidence {got.confidence:.2f}"


def run_workflow(
    wf: Workflow,
    skills: dict[str, Skill],
    extractor=None,
    review=None,
    healer=None,
    client=None,
    model: str | None = None,
    show: bool = False,
    limit: int | None = None,
    log=print,
    emit=None,
) -> dict:
    from playwright.sync_api import sync_playwright

    from mimic.agent import ReplayAgent
    from mimic.outcome import check_signal
    from mimic.recorder import launch_browser
    from mimic.run import check_secrets

    review = review or ConsoleReview(auto=True)
    emit = emit or (lambda event: None)
    t0 = time.monotonic()
    check_apps_answer({st.run: skills[st.run] for st in wf.steps if isinstance(st, RunStep)})
    with tempfile.TemporaryDirectory() as tmp:
        items = load_items(wf, Path(tmp), limit)
        row_fields = list(items[0].context.get("row", {})) if items else []
        rules = compile_rules(wf, known_fields(wf, skills, row_fields), client, model, log)
        for step in wf.steps:
            if isinstance(step, RunStep):
                check_secrets(skills[step.run], {})
        if any(isinstance(s, ReadStep) for s in wf.steps) and extractor is None:
            raise ValueError("this workflow reads documents: an extractor (vision model) is needed")

        choices = dropdown_choices(wf, skills, log)
        results = []
        first = next(skills[st.run] for st in wf.steps if isinstance(st, RunStep))
        emit(
            {
                "type": "run_started",
                "run_id": wf.name,
                "skill": wf.name,
                "workflow": True,
                "total": len(items),
                "layout": "",
                "url": start_url(first),
            }
        )
        with sync_playwright() as p:
            browser = launch_browser(p, headless=not show)
            page = browser.new_context().new_page()
            live = None
            if getattr(emit, "wants_frames", False):
                from mimic.live import Screencast

                live = Screencast(page, emit).start()
            for n, item in enumerate(items, 1):
                checkpoint = getattr(review, "checkpoint", None)
                if checkpoint is not None and not checkpoint():
                    break
                read_documents(item, wf, skills, extractor, choices)
                if item.error is None:
                    plan_item(item, wf, skills, rules)
                decision = review(
                    format_item(n, len(items), item), item=item, index=n, total=len(items)
                )
                if decision == "quit":
                    break
                emit(
                    {
                        "type": "item_started",
                        "index": n,
                        "total": len(items),
                        "title": item.title,
                        "steps": plan_groups(item),
                    }
                )
                result = _execute(
                    item,
                    decision,
                    page,
                    skills,
                    healer,
                    ReplayAgent,
                    check_signal,
                    log,
                    n,
                    len(items),
                    emit,
                )
                results.append(result)
                done = sum(r["status"] == "done" for r in results)
                emit(
                    {
                        "type": "item_done",
                        "index": n,
                        "status": result["status"],
                        "error": result.get("error"),
                        "title": item.title,
                        "steps": result["steps"],
                        "done": len(results),
                        "succeeded": done,
                    }
                )
            if live is not None:
                page.wait_for_timeout(500)
                live.stop()
            browser.close()

    done = sum(r["status"] == "done" for r in results)
    report = {
        "run_id": f"{wf.name}-{uuid.uuid4().hex[:8]}",
        "workflow": wf.name,
        "items": len(items),
        "processed": len(results),
        "done": done,
        "success_rate": round(done / max(len(results), 1), 4),
        "steps_run": sum(s["status"] != "skipped" for r in results for s in r["steps"]),
        "steps_skipped_by_rule": sum(s["status"] == "skipped" for r in results for s in r["steps"]),
        "total_s": round(time.monotonic() - t0, 1),
        "results": results,
    }
    emit({"type": "run_done", "report": {k: v for k, v in report.items() if k != "results"}})
    return report


# Where each skill of the example workflow is demonstrated in the sandbox (mimic sandbox).
SANDBOX_DEMOS = {
    "enter_invoice": "http://localhost:8001",
    "request_approval": "http://localhost:8001/approvals/new",
    "reply_to_supplier": "http://localhost:8001/mail/mail-0001/reply",
}


def learn_hint(missing: list[str]) -> str:
    """The commands that create the missing skills: record the task once, then learn it."""
    lines = ["Record each one once, then learn it:"]
    for name in missing:
        url = SANDBOX_DEMOS.get(name, "<the app's URL>")
        lines.append(f"  mimic record {name} --url {url}")
        lines.append(f"  mimic learn recordings/{name}/recording.json")
    return "\n".join(lines)


def plan_groups(item: Item) -> list[dict]:
    """The plan of one email for the dashboard: each skill, whether it runs, and its actions."""
    from mimic.approval import plan_rows

    flags = {k.split(".", 1)[-1]: v for k, v in item.flags.items()}
    return [
        {
            "skill": st.skill,
            "runs": st.runs,
            "why": st.why,
            "plan": plan_rows(st.actions, flags) if st.runs else [],
        }
        for st in item.plan
    ]


def _execute(item, decision, page, skills, healer, agent_cls, check_signal, log, n, total,
             emit=None):  # fmt: skip
    emit = emit or (lambda event: None)
    out = {
        "id": item.id,
        "title": item.title,
        "flags": item.flags,
        "steps": [],
        "values": {k: v for k, v in item.context.items() if k not in ("email",)},
    }
    if item.error:
        log(f"  {n:>3}/{total}  cannot run: {item.error}")
        return {**out, "status": "failed", "error": item.error}
    if decision == "skip":
        log(f"  {n:>3}/{total}  skipped by you")
        return {**out, "status": "skipped", "error": "skipped by you"}
    from mimic.approval import plan_label

    for part, step in enumerate(item.plan):
        if not step.runs:
            out["steps"].append({"skill": step.skill, "status": "skipped", "why": step.why})
            continue
        t = time.monotonic()
        page.goto(step.url)

        def on_step(r, step=step, part=part):
            action = next((a for a in step.actions if a["index"] == r.index), None)
            emit(
                {
                    "type": "step",
                    "task": n,
                    "part": part,
                    "index": r.index,
                    "action": r.action,
                    "label": plan_label(action) if action else "",
                    "success": r.success,
                    "strategy": r.strategy,
                    "healed": r.healed,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
            )

        agent = agent_cls(page, on_step=on_step, healer=healer, skill_name=step.skill)
        result = agent.run(step.actions)
        error = result.error
        if result.success and skills[step.skill].success is not None:
            error = check_signal(page, skills[step.skill].success)
        ok = result.success and error is None
        out["steps"].append(
            {
                "skill": step.skill,
                "status": "done" if ok else "failed",
                "ms": int((time.monotonic() - t) * 1000),
                "error": error,
            }
        )
        if not ok:  # later steps depend on this one: stop this item here
            log(f"  {n:>3}/{total}  {step.skill} failed: {error}")
            return {**out, "status": "failed", "error": f"{step.skill}: {error}"}
    ran = " + ".join(s["skill"] for s in out["steps"] if s["status"] == "done")
    log(f"  {n:>3}/{total}  done  ({ran})")
    return {**out, "status": "done", "error": None}


# ---------------------------------------------------------------- CLI


def _skills(wf: Workflow) -> dict[str, Skill]:
    from mimic.store import SkillStore

    store = SkillStore()
    names = {s.run for s in wf.steps if isinstance(s, RunStep)}
    names |= {s.fields_of for s in wf.steps if isinstance(s, ReadStep)}
    skills = {name: store.get(name) for name in sorted(names)}
    missing = [n for n, s in skills.items() if s is None]
    if missing:
        raise SystemExit(f"Unknown skill(s): {', '.join(missing)}. " + learn_hint(missing))
    return skills


def main(argv: list[str] | None = None) -> int:
    from mimic.config import get_settings
    from mimic.llm import OllamaClient

    parser = argparse.ArgumentParser(description="Run a workflow of skills across apps")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="Validate a workflow and show its rules")
    c.add_argument("file", type=Path)
    r = sub.add_parser("run", help="Run a workflow on every email / row")
    r.add_argument("file", type=Path)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--yes", action="store_true", help="Do not ask before each email")
    r.add_argument("--show", action="store_true", help="Show the browser")
    r.add_argument("--no-heal", action="store_true")
    r.add_argument("--results", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    wf = load_workflow(args.file)
    skills = _skills(wf)
    client = OllamaClient()
    problems = lint_workflow(wf, skills)
    if problems:
        print(f"{args.file}: fix these first")
        for p in problems:
            print(f"  - {p}")
        return 1
    if args.cmd == "check":
        known = known_fields(wf, skills, [])
        compile_rules(wf, known, client, settings.planner_model)
        print(f"{wf.name}: {len(wf.steps)} steps, skills {', '.join(skills)}")
        print(f"Fields you can use: {', '.join(sorted(known))}")
        return 0

    extractor = None
    if any(isinstance(s, ReadStep) for s in wf.steps):
        from mimic.extract import DocumentExtractor

        extractor = DocumentExtractor(client, settings.vision_model)
    healer = None
    if not args.no_heal:
        from mimic.healing import Healer, HealMemory
        from mimic.run import _load_matcher
        from mimic.vision import VisionLocator

        healer = Healer(
            embedder=client.embed,
            vision=VisionLocator(client, settings.vision_model),
            memory=HealMemory(use_db=True),
            matcher=_load_matcher(),
        )
    report = run_workflow(
        wf,
        skills,
        extractor,
        ConsoleReview(auto=args.yes),
        healer,
        client,
        settings.planner_model,
        args.show,
        args.limit,
    )
    print(
        f"\n{report['done']}/{report['processed']} done · {report['steps_run']} steps run · "
        f"{report['steps_skipped_by_rule']} skipped by a rule · {report['total_s']} s"
    )
    out = Path("reports") / f"{report['run_id']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    path = write_workflow_results(args.results or out.with_suffix(".csv"), report)
    print(f"Results per item: {path}")
    return 0 if report["done"] == report["processed"] else 1


def write_workflow_results(path: Path, report: dict) -> Path:
    from mimic.tables import write_results

    records, rows = [], []
    for i, r in enumerate(report["results"], 1):
        flat = {"item": r["id"], "title": r["title"]}
        for name, values in r["values"].items():
            if isinstance(values, dict):
                flat |= {f"{name}.{k}": v for k, v in values.items() if k != "file"}
        flat["steps"] = "; ".join(f"{s['skill']}: {s['status']}" for s in r["steps"]) or "-"
        records.append(flat)
        rows.append(
            {
                "index": i,
                "status": r["status"],
                "error": r.get("error"),
                "ms": sum(s.get("ms", 0) for s in r["steps"]),
            }
        )
    return write_results(path, records, rows)


if __name__ == "__main__":
    raise SystemExit(main())
