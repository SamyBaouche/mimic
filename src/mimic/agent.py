"""Replay agent: executes a skill for one task with a LangGraph loop.

    locate ──► act ──► verify ──► next step / done
       ▲                  │
       └──── retry ◄──────┘  (next locator strategy, max MAX_ATTEMPTS)

Locator strategies, from most to least precise. Each one that the recorder captured is tried:
    selector  #vendor                       (exact, breaks when ids change)
    label     <label>Vendor</label>         (survives id changes)
    name      name="vendor"                 (survives relabelling)
    text      button "Save invoice"         (buttons and links)
    heal      when all of the above fail: memory → semantic → vision (see healing.py)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Locator, Page

from mimic.models import TRUE_WORDS, ElementInfo, as_bool  # noqa: F401 - re-exported

MAX_ATTEMPTS = 3
LOCATE_TIMEOUT_MS = 1500
GRACE_MS = 150
NAV_TIMEOUT_MS = 5000
ACT_TIMEOUT_MS = 1000  # fail fast, like a production agent; mitigations wait longer

STRATEGY_CONFIDENCE = {"selector": 1.0, "option": 1.0, "label": 0.8, "name": 0.7, "text": 0.6}


def _radio_value(el: ElementInfo, value: str) -> str:
    """Match the task's value to one of the options seen in the demo, ignoring case."""
    wanted = str(value).strip().lower()
    return next((o for o in el.options if o.lower() == wanted), str(value).strip())


@dataclass
class StepResult:
    index: int
    action: str
    strategy: str | None
    confidence: float
    attempts: int
    latency_ms: int
    success: bool
    error: str | None = None
    fallbacks: int = 0  # strategies tried before the one that worked
    healed: bool = False
    features: dict | None = None  # observed right before acting (risk model input)
    risk: float | None = None  # predicted failure probability
    mitigations: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    success: bool
    skipped: bool = False  # the user declined the submit
    steps: list[StepResult] = field(default_factory=list)
    duration_ms: int = 0
    error: str | None = None


class AgentState(TypedDict):
    actions: list[dict]
    idx: int
    attempt: int
    tried: list[str]
    locator_strategy: str | None
    confidence: float
    error: str | None
    step_started: float
    results: list[StepResult]
    status: Literal["running", "done", "failed", "skipped"]


def _candidates(page: Page, step: dict) -> list[tuple[str, Locator]]:
    el: ElementInfo = step["element"]
    action = step["action"]
    if action == "choose":
        # The demo clicked one radio; this task may need another option of the same group.
        value = _radio_value(el, step["value"])
        found = []
        if el.name:
            css = f'input[type=radio][name="{el.name}"][value="{value}"]'
            found.append(("option", page.locator(css)))
        found.append(("label", page.get_by_role("radio", name=value, exact=True)))
        return found
    found = []
    if el.selector:
        found.append(("selector", page.locator(el.selector)))
    if el.label and action != "click":
        found.append(("label", page.get_by_label(el.label, exact=True)))
    if el.name:
        found.append(("name", page.locator(f'[name="{el.name}"]')))
    if el.text and action == "click":
        found.append(("text", page.get_by_role("button", name=el.text, exact=True)))
    return found


class ReplayAgent:
    """Runs one rendered skill (list of concrete actions) on an open page."""

    def __init__(
        self,
        page: Page,
        on_step=None,
        healer=None,
        skill_name: str = "skill",
        before_submit=None,  # callable(action) -> bool: human confirmation before submitting
        risk_policy: str = "never",  # never | always | predicted
        risk_model=None,  # RiskModel, required for the "predicted" policy
    ):
        if risk_policy == "predicted" and risk_model is None:
            raise ValueError("risk_policy='predicted' needs a trained risk model")
        self.before_submit = before_submit
        self.risk_policy = risk_policy
        self.risk_model = risk_model
        self._observation: dict = {}
        self.page = page
        self.healer = healer
        self.skill_name = skill_name
        self.on_step = on_step  # callback(StepResult) for telemetry / live progress
        self._locator: Locator | None = None
        self._pending_heal = None
        self.graph = self._build()

    # ---------- nodes ----------

    def _skip_ahead(self, state: AgentState) -> int | None:
        """The app skipped a page of the demo (already signed in, no consent screen today...):
        jump to the first later step recorded on the page we are actually on."""
        actions, idx = state["actions"], state["idx"]
        recorded = actions[idx].get("page")
        here = urlsplit(self.page.url).path or "/"
        if not recorded or recorded == here or state["tried"]:
            return None
        return next(
            (j for j in range(idx + 1, len(actions)) if actions[j].get("page") == here), None
        )

    def locate(self, state: AgentState) -> dict:
        skipped: dict = {}
        jump = self._skip_ahead(state)
        if jump is not None:
            done = [
                StepResult(
                    index=k,
                    action=state["actions"][k]["action"],
                    strategy="not needed",
                    confidence=1.0,
                    attempts=0,
                    latency_ms=0,
                    success=True,
                )
                for k in range(state["idx"], jump)
            ]
            for r in done:
                if self.on_step:
                    self.on_step(r)
            skipped = {"idx": jump, "results": [*state["results"], *done]}
            state = {**state, **skipped}
        update = self._locate(state)
        return {**skipped, **update}

    def _locate(self, state: AgentState) -> dict:
        action = state["actions"][state["idx"]]
        candidates = [
            (name, loc)
            for name, loc in _candidates(self.page, action)
            if name not in state["tried"]
        ]

        def first_match():
            # Instant check of every strategy: no waiting on the ones that cannot match.
            for name, loc in candidates:
                try:
                    if loc.count() == 1 and loc.first.is_visible():
                        return name, loc.first
                except Exception:  # noqa: BLE001
                    continue
            return None

        match = first_match()
        if match is None and candidates:
            # Nothing yet: the page may still be rendering. Wait (long only if it is still
            # loading, a short grace period otherwise), then check again.
            try:
                loaded = self.page.evaluate("document.readyState") == "complete"
                self.page.wait_for_load_state()
                timeout = GRACE_MS if loaded else LOCATE_TIMEOUT_MS
                candidates[0][1].first.wait_for(state="visible", timeout=timeout)
            except Exception:  # noqa: BLE001
                pass
            match = first_match()

        if match is not None:
            name, self._locator = match
            return {
                "locator_strategy": name,
                "confidence": STRATEGY_CONFIDENCE[name],
                "tried": [*state["tried"], name],
            }

        tried = [*state["tried"], *(n for n, _ in candidates)]
        # Radio options are found by their value; healing would guess an option, so it stays off.
        if self.healer is not None and "heal" not in tried and action["action"] != "choose":
            tried.append("heal")
            healed = self.healer.heal(self.page, action, self.skill_name)
            self._pending_heal = healed
            if healed is not None:
                self._locator = self.page.locator(healed.selector).first
                return {
                    "locator_strategy": healed.strategy,
                    "confidence": healed.confidence,
                    "tried": tried,
                }
        self._locator = None
        return {
            "locator_strategy": None,
            "confidence": 0.0,
            "tried": tried,
            "error": "element not found",
        }

    def act(self, state: AgentState) -> dict:
        if state["locator_strategy"] is None:
            return {}
        action = state["actions"][state["idx"]]
        self._observe_and_mitigate(state, action)
        try:
            kind = action["action"]
            if kind == "fill":
                self._locator.fill(str(action["value"]), timeout=ACT_TIMEOUT_MS)
            elif kind == "select":
                self._locator.select_option(str(action["value"]), timeout=ACT_TIMEOUT_MS)
            elif kind == "check":
                self._locator.set_checked(as_bool(action["value"]), timeout=ACT_TIMEOUT_MS)
            elif kind == "choose":
                self._locator.check(timeout=ACT_TIMEOUT_MS)
            elif kind == "upload":
                path = Path(str(action["value"])).expanduser()
                if not path.is_file():
                    return {"error": f"file not found: {path}"}
                self._locator.set_input_files(str(path), timeout=ACT_TIMEOUT_MS)
            elif kind == "press":
                if action.get("navigates"):
                    if self.before_submit is not None and not self.before_submit(action):
                        return {"error": "declined by user", "status": "skipped"}
                    return self._then_new_page(
                        lambda: self._locator.press(str(action["value"]), timeout=ACT_TIMEOUT_MS)
                    )
                self._locator.press(str(action["value"]), timeout=ACT_TIMEOUT_MS)
            elif action["element"].type == "submit":
                if self.before_submit is not None and not self.before_submit(action):
                    return {"error": "declined by user", "status": "skipped"}
                # A submit in the demo navigated: if nothing happens, it was the wrong button.
                with self.page.expect_navigation(timeout=NAV_TIMEOUT_MS) as nav:
                    self._locator.click(timeout=ACT_TIMEOUT_MS)
                response = nav.value
                if response is not None and response.status >= 400:
                    return {"error": f"server error {response.status}"}
            elif action.get("navigates"):
                # A link or a "Next" button: the demo moved to another page (or route).
                return self._then_new_page(lambda: self._locator.click(timeout=ACT_TIMEOUT_MS))
            else:
                self._locator.click(timeout=ACT_TIMEOUT_MS)
                self.page.wait_for_load_state()
            return {"error": None}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}".splitlines()[0][:200]}

    def _then_new_page(self, do) -> dict:
        """Act, then wait until the URL changes: real navigations and single-page apps alike."""
        before = self.page.url
        do()
        try:
            self.page.wait_for_url(lambda url: url != before, timeout=NAV_TIMEOUT_MS)
        except Exception:  # noqa: BLE001
            return {"error": "the page did not change as in the demo"}
        self.page.wait_for_load_state()
        return {"error": None}

    def _observe_and_mitigate(self, state: AgentState, action: dict) -> None:
        """Probe the element, score the risk, and fix the page first if it is worth it."""
        from mimic.risk import mitigate, probe

        features = probe(
            self._locator,
            action,
            state["locator_strategy"],
            state["confidence"],
            max(len(state["tried"]) - 1, 0),
        )
        if self._observation:
            # Retry of the same step: keep what was observed at the first attempt (the decision
            # point the model learns), just add the new mitigations.
            first = self._observation
            features = first["features"]
        risk = self.risk_model.score(features) if self.risk_model is not None else None
        should = self.risk_policy == "always" or (
            self.risk_policy == "predicted"
            and risk is not None
            and risk >= self.risk_model.threshold
        )
        done = mitigate(self.page, self._locator) if should else []
        if self._observation:
            self._observation["mitigations"] += done
        else:
            self._observation = {"features": features, "risk": risk, "mitigations": done}

    def verify(self, state: AgentState) -> dict:
        if state["status"] == "skipped":
            self._pending_heal = None  # not verified either way: neither remember nor reject
            return {}
        action = state["actions"][state["idx"]]
        ok = state["locator_strategy"] is not None and state["error"] is None
        if ok and action["action"] in ("fill", "select", "check", "choose", "upload"):
            try:
                problem = self._check_value(action)
                if problem:
                    ok, state = False, {**state, "error": problem}
            except Exception as exc:  # noqa: BLE001
                ok, state = False, {**state, "error": type(exc).__name__}

        if self._pending_heal is not None:
            (self.healer.confirm if ok else self.healer.reject)(self._pending_heal)
            self._pending_heal = None

        attempts = state["attempt"] + 1
        if ok or attempts >= MAX_ATTEMPTS or not self._strategies_left(state):
            result = StepResult(
                index=state["idx"],
                action=action["action"],
                strategy=state["locator_strategy"] if ok else None,
                confidence=state["confidence"] if ok else 0.0,
                attempts=attempts,
                latency_ms=int((time.monotonic() - state["step_started"]) * 1000),
                success=ok,
                error=None if ok else state["error"],
                fallbacks=max(len(state["tried"]) - 1, 0),
                healed=ok
                and state["locator_strategy"] in ("memory", "matcher", "semantic", "vision"),
                features=self._observation.get("features"),
                risk=self._observation.get("risk"),
                mitigations=self._observation.get("mitigations", []),
            )
            self._observation = {}
            if self.on_step:
                self.on_step(result)
            last = state["idx"] + 1 >= len(state["actions"])
            status = "failed" if not ok else ("done" if last else "running")
            return {
                "results": [*state["results"], result],
                "idx": state["idx"] + (1 if ok else 0),
                "attempt": 0,
                "tried": [],
                "error": None if ok else state["error"],
                "step_started": time.monotonic(),
                "status": status,
            }
        return {"attempt": attempts, "error": state["error"]}  # retry with next strategy

    def _check_value(self, action: dict) -> str | None:
        """Read the element back: the action only counts if the page shows the right value."""
        kind = action["action"]
        if kind == "check":
            want = as_bool(action["value"])
            return None if self._locator.is_checked() == want else "checkbox did not change"
        if kind == "choose":
            return None if self._locator.is_checked() else "option not selected"
        if kind == "upload":
            name = self._locator.evaluate(
                "el => el.files && el.files.length ? el.files[0].name : ''"
            )
            return None if name == Path(str(action["value"])).name else "file not attached"
        if kind == "select":
            # A dropdown shows a label ("Travel") but stores a value ("travel"): either is fine.
            value, label = self._locator.evaluate(
                "el => [el.value, el.selectedOptions[0] ? el.selectedOptions[0].text : '']"
            )
            want = " ".join(str(action["value"]).lower().split())
            if want in (" ".join(value.lower().split()), " ".join(label.lower().split())):
                return None
            return f"value mismatch: {label or value!r}"
        actual = self._locator.input_value()
        if _same_value(actual, str(action["value"])):
            return None
        return f"value mismatch: {actual!r}"

    def _strategies_left(self, state: AgentState) -> bool:
        action = state["actions"][state["idx"]]
        names = [n for n, _ in _candidates(self.page, action)]
        if self.healer is not None and action["action"] != "choose":
            names.append("heal")
        return any(n not in state["tried"] for n in names)

    @staticmethod
    def route(state: AgentState) -> str:
        if state["status"] in ("done", "failed", "skipped"):
            return END
        return "locate"

    # ---------- graph ----------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("locate", self.locate)
        g.add_node("act", self.act)
        g.add_node("verify", self.verify)
        g.add_edge(START, "locate")
        g.add_edge("locate", "act")
        g.add_edge("act", "verify")
        g.add_conditional_edges("verify", self.route, ["locate", END])
        return g.compile()

    def run(self, actions: list[dict]) -> TaskResult:
        t0 = time.monotonic()
        final = self.graph.invoke(
            AgentState(
                actions=actions,
                idx=0,
                attempt=0,
                tried=[],
                locator_strategy=None,
                confidence=0.0,
                error=None,
                step_started=t0,
                results=[],
                status="running",
            ),
            config={"recursion_limit": 20 * len(actions) + 10},
        )
        return TaskResult(
            success=final["status"] == "done",
            skipped=final["status"] == "skipped",
            steps=final["results"],
            duration_ms=int((time.monotonic() - t0) * 1000),
            error=final["error"],
        )


def _same_value(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    try:
        return abs(float(actual) - float(expected)) < 1e-9
    except ValueError:
        return False
