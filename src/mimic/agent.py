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
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from playwright.sync_api import Locator, Page

from mimic.models import ElementInfo

MAX_ATTEMPTS = 3
LOCATE_TIMEOUT_MS = 1500
GRACE_MS = 150
NAV_TIMEOUT_MS = 5000

STRATEGY_CONFIDENCE = {"selector": 1.0, "label": 0.8, "name": 0.7, "text": 0.6}


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


def _candidates(page: Page, el: ElementInfo, action: str) -> list[tuple[str, Locator]]:
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
    ):
        self.before_submit = before_submit
        self.page = page
        self.healer = healer
        self.skill_name = skill_name
        self.on_step = on_step  # callback(StepResult) for telemetry / live progress
        self._locator: Locator | None = None
        self._pending_heal = None
        self.graph = self._build()

    # ---------- nodes ----------

    def locate(self, state: AgentState) -> dict:
        action = state["actions"][state["idx"]]
        candidates = [
            (name, loc)
            for name, loc in _candidates(self.page, action["element"], action["action"])
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
        if self.healer is not None and "heal" not in tried:
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
        try:
            if action["action"] == "fill":
                self._locator.fill(str(action["value"]))
            elif action["action"] == "select":
                self._locator.select_option(str(action["value"]))
            elif action["element"].type == "submit":
                if self.before_submit is not None and not self.before_submit(action):
                    return {"error": "declined by user", "status": "skipped"}
                # A submit in the demo navigated: if nothing happens, it was the wrong button.
                with self.page.expect_navigation(timeout=NAV_TIMEOUT_MS):
                    self._locator.click()
            else:
                self._locator.click()
                self.page.wait_for_load_state()
            return {"error": None}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}".splitlines()[0][:200]}

    def verify(self, state: AgentState) -> dict:
        if state["status"] == "skipped":
            self._pending_heal = None  # not verified either way: neither remember nor reject
            return {}
        action = state["actions"][state["idx"]]
        ok = state["locator_strategy"] is not None and state["error"] is None
        if ok and action["action"] in ("fill", "select"):
            try:
                actual = self._locator.input_value()
                ok = _same_value(actual, str(action["value"]))
                if not ok:
                    state = {**state, "error": f"value mismatch: {actual!r}"}
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
                healed=ok and state["locator_strategy"] in ("memory", "semantic", "vision"),
            )
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

    def _strategies_left(self, state: AgentState) -> bool:
        action = state["actions"][state["idx"]]
        names = [n for n, _ in _candidates(self.page, action["element"], action["action"])]
        if self.healer is not None:
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
