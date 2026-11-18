"""Self-healing: find an element again after the UI changed.

Called by the agent only when the recorded locators (id, label, name, text) all fail.
Strategies, cheapest first:
    memory    a fix found earlier for this step on this page (pgvector + in-run cache)
    semantic  compare the recorded element with every candidate on the page by meaning
              (Ollama embeddings) plus structural hints (same input/button type)
    vision    Qwen2.5-VL looks at the demo screenshot and the current screen;
              the proposed point is verified against the DOM before acting
Every successful heal is stored in memory, so only the first task pays for it.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import Page

from mimic.config import get_settings
from mimic.dom import describe_text, list_candidates
from mimic.models import ElementInfo
from mimic.vision import MIN_CONFIDENCE_UNVERIFIED

SEMANTIC_THRESHOLD = 0.80
SEMANTIC_MARGIN = 0.05
CONFIDENCE = {"memory": 0.9}
MAX_FAILURES_PER_STEP = 3  # stop paying for models on a step that keeps failing


@dataclass
class HealResult:
    strategy: str
    selector: str
    confidence: float
    element: ElementInfo
    context: tuple = ()  # (skill, step, page_path, embedding) used to remember the fix


@dataclass
class HealStats:
    attempts: int = 0
    healed: Counter = field(default_factory=Counter)
    failed: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "healed": dict(self.healed),
            "failed": self.failed,
            "seconds": round(self.seconds, 2),
        }


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class HealMemory:
    """Remembers healed selectors. In-run cache always; Postgres + pgvector when available."""

    def __init__(self, use_db: bool = True):
        self.cache: dict[tuple[str, int, str], str] = {}
        self.use_db = use_db
        self.dim = get_settings().embedding_dim

    def _fits(self, embedding: list[float] | None) -> bool:
        # The column is vector(dim): a different embedding model must not break the memory.
        return embedding is not None and len(embedding) == self.dim

    def lookup(self, skill: str, step: int, path: str, embedding: list[float] | None) -> list[str]:
        found = [self.cache[(skill, step, path)]] if (skill, step, path) in self.cache else []
        if not self.use_db:
            return found
        try:
            from mimic.db import connect

            with connect() as conn:
                if self._fits(embedding):
                    # Same step first, then the most similar element healed on this page
                    # (lets a fix learned by one skill help another skill on the same screen).
                    rows = conn.execute(
                        """SELECT selector FROM healed_elements WHERE page_path = %s
                           ORDER BY (skill_name = %s AND step_index = %s) DESC,
                                    embedding <=> %s::vector LIMIT 3""",
                        (path, skill, step, _vec(embedding)),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT selector FROM healed_elements
                           WHERE page_path = %s AND skill_name = %s AND step_index = %s
                           ORDER BY id DESC LIMIT 3""",
                        (path, skill, step),
                    ).fetchall()
            found += [r[0] for r in rows if r[0] not in found]
        except Exception:  # noqa: BLE001 - memory is an optimisation, never a hard dependency
            pass
        return found

    def remember(
        self,
        skill: str,
        step: int,
        path: str,
        selector: str,
        strategy: str,
        embedding: list[float] | None,
    ) -> None:
        self.cache[(skill, step, path)] = selector
        if not self.use_db:
            return
        try:
            from mimic.db import connect

            with connect() as conn:
                conn.execute(
                    """INSERT INTO healed_elements
                       (skill_name, step_index, page_path, selector, strategy, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s::vector)""",
                    (
                        skill,
                        step,
                        path,
                        selector,
                        strategy,
                        _vec(embedding) if self._fits(embedding) else None,
                    ),
                )
                conn.commit()
        except Exception:  # noqa: BLE001
            pass


class Healer:
    def __init__(
        self, embedder=None, vision=None, memory: HealMemory | None = None, debug: bool = False
    ):
        self.embedder = embedder  # callable(list[str]) -> list[list[float]]
        self.vision = vision  # VisionLocator
        self.memory = memory or HealMemory(use_db=False)
        self.debug = debug
        self.stats = HealStats()
        self._emb_cache: dict[str, list[float]] = {}
        self._failures: Counter = Counter()

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"    [heal] {msg}")

    # ---------- embeddings ----------

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if self.embedder is None:
            return None
        todo = [t for t in texts if t not in self._emb_cache]
        try:
            if todo:
                for t, v in zip(todo, self.embedder(todo), strict=True):
                    self._emb_cache[t] = v
        except Exception:  # noqa: BLE001
            self.embedder = None  # embeddings unavailable: stop trying for this run
            return None
        return [self._emb_cache[t] for t in texts]

    # ---------- strategies ----------

    def _try_memory(self, page, skill, step, path, emb) -> HealResult | None:
        for selector in self.memory.lookup(skill, step, path, emb):
            loc = page.locator(selector)
            try:
                if loc.count() == 1 and loc.first.is_visible():
                    el = ElementInfo(
                        tag="?", selector=selector, bbox={"x": 0, "y": 0, "width": 0, "height": 0}
                    )
                    return HealResult("memory", selector, CONFIDENCE["memory"], el)
            except Exception:  # noqa: BLE001
                continue
        return None

    def semantic_scores(
        self, target: ElementInfo, candidates: list[ElementInfo], action: str
    ) -> list[float]:
        texts = [describe_text(target, action)] + [describe_text(c, action) for c in candidates]
        vectors = self._embed(texts)
        scores = []
        for i, cand in enumerate(candidates):
            if vectors is not None:
                sim = _cosine(vectors[0], vectors[i + 1])
            else:
                a = (target.label or target.text or "").lower()
                b = (cand.label or cand.text or "").lower()
                sim = SequenceMatcher(None, a, b).ratio()
            if target.type and cand.type == target.type:
                sim += 0.1  # same input/button type (e.g. both submit buttons)
            scores.append(min(sim, 1.0))
        return scores

    def _try_semantic(self, page, target, action) -> HealResult | None:
        candidates = list_candidates(page, action)
        if not candidates:
            return None
        scores = self.semantic_scores(target, candidates, action)
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
        best_score, best = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        self._log(
            f"semantic for {describe_text(target, action)}: "
            + ", ".join(f"{describe_text(c, action)}={sc:.3f}" for sc, c in ranked[:4])
        )
        if best_score >= SEMANTIC_THRESHOLD and best_score - second >= SEMANTIC_MARGIN:
            return HealResult("semantic", best.selector, round(best_score, 3), best)
        return None

    def _try_vision(self, page, target, action, reference) -> HealResult | None:
        if self.vision is None:
            return None
        candidates = list_candidates(page, action)
        try:
            pick = self.vision.choose(page.screenshot(), target, action, candidates, reference)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! vision unavailable ({type(exc).__name__}: {exc})")
            self.vision = None
            return None
        self._log(f"vision answer: {pick.raw if pick else None}")
        if pick is None or pick.index < 0:
            return None
        verifiable = action == "click" and target.type == "submit"
        if not verifiable and pick.confidence < MIN_CONFIDENCE_UNVERIFIED:
            # A wrong field would be filled silently: only act on a confident pick.
            self._log(f"vision pick too uncertain ({pick.confidence:.2f}) for an unverifiable step")
            return None
        el = candidates[pick.index]  # a real DOM element, never a guessed coordinate
        return HealResult("vision", el.selector, round(pick.confidence, 3), el)

    # ---------- entry point ----------

    def heal(self, page: Page, action: dict, skill: str) -> HealResult | None:
        t0 = time.monotonic()
        self.stats.attempts += 1
        target: ElementInfo = action["element"]
        kind = action["action"]
        step = action.get("index", -1)
        path = urlsplit(page.url).path or "/"
        ref = Path(action["screenshot"]) if action.get("screenshot") else None
        if self._failures[(skill, step, path)] >= MAX_FAILURES_PER_STEP:
            self.stats.failed += 1
            return None

        emb = self._embed([describe_text(target, kind)])
        emb = emb[0] if emb else None

        result = (
            self._try_memory(page, skill, step, path, emb)
            or self._try_semantic(page, target, kind)
            or self._try_vision(page, target, kind, ref)
        )
        self.stats.seconds += time.monotonic() - t0
        if result is None:
            self.stats.failed += 1
            self._failures[(skill, step, path)] += 1
            self._log("no element found")
            return None
        result.context = (skill, step, path, emb)
        self._log(f"proposed {result.selector} via {result.strategy}")
        return result

    def confirm(self, result: HealResult) -> None:
        """Called by the agent once the action on the healed element was verified."""
        self.stats.healed[result.strategy] += 1
        self._failures.pop(result.context[:3], None)
        if result.strategy != "memory":
            skill, step, path, emb = result.context
            self.memory.remember(skill, step, path, result.selector, result.strategy, emb)

    def reject(self, result: HealResult) -> None:
        """The healed element did not work: never remember it."""
        self.stats.failed += 1
        key = result.context[:3]
        self._failures[key] += 1
        self._log(f"rejected {result.selector} ({result.strategy}): action had no effect")
        if self.memory.cache.get(key) == result.selector:
            del self.memory.cache[key]
