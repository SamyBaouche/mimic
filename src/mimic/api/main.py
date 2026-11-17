"""Mimic API: control plane for skills and agent runs."""

import httpx
from fastapi import FastAPI

from mimic import __version__
from mimic.config import get_settings

app = FastAPI(title="Mimic", version=__version__)


def _check_db() -> str:
    try:
        from mimic.db import connect

        with connect() as conn:
            conn.execute("SELECT 1")
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"down ({type(exc).__name__})"


def _check_ollama() -> str:
    try:
        r = httpx.get(f"{get_settings().ollama_host}/api/tags", timeout=2)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        return f"ok ({len(models)} models)"
    except Exception as exc:  # noqa: BLE001
        return f"down ({type(exc).__name__})"


@app.get("/health")
def health() -> dict:
    checks = {"db": _check_db(), "ollama": _check_ollama()}
    status = "ok" if all(v.startswith("ok") for v in checks.values()) else "degraded"
    return {"status": status, "version": __version__, "checks": checks}
