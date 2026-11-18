"""Thin client for the local Ollama server (chat with JSON output + embeddings)."""

from __future__ import annotations

import base64
import json

import httpx

from mimic.config import get_settings


class OllamaClient:
    def __init__(self, host: str | None = None, timeout: float = 120.0):
        settings = get_settings()
        self.host = (host or settings.ollama_host).rstrip("/")
        self.timeout = timeout

    def chat_json(
        self, model: str, system: str, user: str, schema: dict, images: list[bytes] | None = None
    ) -> dict:
        """Ask the model for a JSON answer constrained by a JSON schema (optionally with images)."""
        user_msg: dict = {"role": "user", "content": user}
        if images:
            user_msg["images"] = [base64.b64encode(img).decode() for img in images]
        r = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": model,
                "stream": False,
                "format": schema,
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, user_msg],
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        r = httpx.post(
            f"{self.host}/api/embed",
            json={"model": model or get_settings().embed_model, "input": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["embeddings"]
