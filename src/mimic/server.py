"""Mimic dashboard: watch the agent work live and approve its plans from the browser.

    uv run python -m mimic.server          # opens http://localhost:8765

FastAPI serves one page and a WebSocket. The agent runs in a worker thread with the same
run_batch() as the CLI; it publishes events (and screencast frames) on an EventBus, and a
WebApprover turns the page's buttons into the same decisions the console prompt makes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import traceback
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from mimic.approval import WebApprover
from mimic.config import get_settings
from mimic.events import EventBus

STATIC = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    skill: str = "enter_invoice"
    source: str = "data"  # data | docs
    path: str | None = None
    layout: str = "v1"
    limit: int = 10
    review: bool = True
    confirm_submit: bool = False
    heal: bool = True


class RunManager:
    """One run at a time, executed in a worker thread."""

    def __init__(self, bus: EventBus, runner=None):
        self.bus = bus
        self.approver: WebApprover | None = None
        self.thread: threading.Thread | None = None
        self.runner = runner or default_runner

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, req: RunRequest) -> None:
        if self.running:
            raise RuntimeError("A run is already in progress")
        self.bus.clear()
        self.approver = WebApprover(self.bus.publish, confirm_submit=req.confirm_submit)
        emit = LiveEmitter(self.bus.publish)

        def target():
            try:
                self.runner(req, self.approver if req.review else None, emit)
            except Exception as exc:  # noqa: BLE001 - surfaced in the page
                traceback.print_exc()
                self.bus.publish({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

        self.thread = threading.Thread(target=target, daemon=True, name="mimic-run")
        self.thread.start()

    def decide(self, message: dict) -> None:
        if self.approver is not None:
            self.approver.decide(message)


class LiveEmitter:
    """The publish function handed to run_batch; asks for screencast frames."""

    wants_frames = True

    def __init__(self, publish):
        self.publish = publish

    def __call__(self, event: dict) -> None:
        self.publish(event)


def default_runner(req: RunRequest, approver, emit) -> None:
    """Build everything the CLI builds, then call the same run_batch()."""
    from mimic.healing import Healer, HealMemory
    from mimic.llm import OllamaClient
    from mimic.run import _load_matcher, collect_options, records_from_documents, run_batch
    from mimic.store import SkillStore
    from mimic.vision import VisionLocator

    settings = get_settings()
    skill = SkillStore().get(req.skill)
    if skill is None:
        raise ValueError(f"Unknown skill '{req.skill}'")
    url = skill.start_url or settings.sandbox_url
    client = OllamaClient()

    titles = flags = truth = None
    if req.source == "docs":
        from mimic.extract import DocumentExtractor

        folder = Path(req.path or "sandbox/data/documents")
        emit({"type": "status", "message": f"Reading documents in {folder}..."})
        extractor = DocumentExtractor(client, settings.vision_model)
        records, titles, flags, truth = records_from_documents(
            skill, folder, extractor, collect_options(skill, url, req.layout), req.limit, emit
        )
    else:
        from mimic.tables import load_records, strip_results

        path = Path(req.path or "sandbox/data/invoices.json")
        records = strip_results(load_records(path))[: req.limit]

    healer = None
    if req.heal:
        healer = Healer(
            embedder=client.embed,
            vision=VisionLocator(client, settings.vision_model),
            memory=HealMemory(use_db=True),
            matcher=_load_matcher(),
        )
    run_batch(
        skill,
        records,
        url,
        req.layout,
        client=client,
        healer=healer,
        approver=approver,
        truth=truth,
        titles=titles,
        flags=flags,
        emit=emit,
    )


def create_app(manager: RunManager | None = None) -> FastAPI:
    bus = manager.bus if manager else EventBus()
    manager = manager or RunManager(bus)
    app = FastAPI(title="Mimic dashboard")
    app.state.manager = manager

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/skills")
    def skills():
        try:
            from mimic.store import SkillStore

            store = SkillStore()
            out = []
            for name in store.list_names():
                skill = store.get(name)
                out.append(
                    {"name": name, "description": skill.description, "parameters": skill.parameters}
                )
            return out
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"Skill library unavailable: {type(exc).__name__}") from exc

    @app.get("/api/state")
    def state():
        return {"running": manager.running}

    @app.post("/api/runs", status_code=202)
    def start(req: RunRequest):
        try:
            manager.start(req)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"started": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        queue = bus.subscribe()

        async def send():
            while True:
                await socket.send_text(json.dumps(await queue.get()))

        async def receive():
            while True:
                manager.decide(json.loads(await socket.receive_text()))

        sender = asyncio.create_task(send())
        try:
            await receive()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            bus.unsubscribe(queue)

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Mimic live dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = parser.parse_args(argv)

    import uvicorn

    url = f"http://localhost:{args.port}"
    print(f"Mimic dashboard on {url}  (Ctrl+C to stop)")
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
