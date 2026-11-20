"""Event bus: the agent (in a worker thread) publishes, the dashboard (asyncio) listens.

Events are plain dicts with a "type": run_started, reading, task_started, plan, step,
task_done, run_done, frame, error... A short history is kept so a page opened mid-run
catches up immediately.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque


class EventBus:
    def __init__(self, history: int = 300):
        self._lock = threading.Lock()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._history: deque = deque(maxlen=history)
        self._last_frame: dict | None = None

    def publish(self, event: dict) -> None:
        """Thread-safe: callable from the agent thread."""
        event = {"ts": round(time.time(), 3), **event}
        with self._lock:
            if event["type"] == "frame":
                self._last_frame = event  # frames are not replayed, only the latest one
            else:
                self._history.append(event)
            subscribers = list(self._subscribers)
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(_put_latest, queue, event)
            except RuntimeError:  # loop closed: the page went away
                pass

    def subscribe(self) -> asyncio.Queue:
        """Called from the asyncio side (a WebSocket). Replays history first."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        with self._lock:
            for event in self._history:
                queue.put_nowait(event)
            if self._last_frame:
                queue.put_nowait(self._last_frame)
            self._subscribers.append((loop, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(lp, q) for lp, q in self._subscribers if q is not queue]

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
            self._last_frame = None

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)


def _put_latest(queue: asyncio.Queue, event: dict) -> None:
    """Never block the agent: when a slow page falls behind, drop the oldest frames."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    queue.put_nowait(event)
