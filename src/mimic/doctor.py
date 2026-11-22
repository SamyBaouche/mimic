"""`mimic doctor`: check everything Mimic needs, and say exactly how to fix what is missing."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from mimic import __version__
from mimic.config import get_settings


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""
    required: bool = True


def check_python() -> Check:
    v = sys.version_info
    return Check(
        "Python",
        v >= (3, 11),
        f"{v.major}.{v.minor}.{v.micro} on {platform.system()}",
        "Install Python 3.11 or newer",
    )


def check_storage() -> Check:
    from mimic.storage import get_storage

    storage = get_storage()
    try:
        storage.init()
        n = storage.stats()["skills"]
        return Check("Storage", True, f"{storage.kind}, {storage.location} ({n} skills)")
    except Exception as exc:  # noqa: BLE001
        fix = (
            "Start Postgres (docker compose up -d), or remove DATABASE_URL to use SQLite"
            if storage.kind == "postgres"
            else f"Check that {Path(storage.location).parent} is writable"
        )
        return Check("Storage", False, f"{storage.kind}: {type(exc).__name__}: {exc}", fix)


def check_browser() -> Check:
    try:
        from playwright.sync_api import sync_playwright

        from mimic.recorder import launch_browser
    except ImportError:
        return Check("Browser", False, "Playwright is not installed", "pip install playwright")
    try:
        with sync_playwright() as p:
            browser = launch_browser(p, headless=True)
            version = browser.version
            browser.close()
        where = "MIMIC_CHROME_PATH" if os.environ.get("MIMIC_CHROME_PATH") else "Playwright"
        return Check("Browser", True, f"Chromium {version} ({where})")
    except Exception as exc:  # noqa: BLE001
        first_line = str(exc).strip().splitlines()[0][:90]
        return Check("Browser", False, first_line, "playwright install chromium")


def check_ollama() -> tuple[Check, list[str]]:
    host = get_settings().ollama_host
    try:
        r = httpx.get(f"{host}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        return Check("Ollama", True, f"running at {host}"), names
    except Exception:  # noqa: BLE001
        return (
            Check(
                "Ollama",
                False,
                f"not reachable at {host}",
                "Install it from https://ollama.com and open the app (or run: ollama serve)",
            ),
            [],
        )


def check_hardware() -> Check:
    from mimic.hardware import PROFILES, detect, recommend

    s = get_settings()
    hw = detect()
    advised, why = recommend(hw)
    used = f"{s.profile_used} profile ({s.profile_reason})"
    detail = f"{hw.describe()} -> {used}"
    heavy = "7b" in s.vision_model or "7b" in s.planner_model
    if advised == "light" and heavy:
        return Check(
            "Hardware",
            False,
            f"{detail}; 7B models will be slow with {why}",
            "set MIMIC_PROFILE=light (" + " + ".join(PROFILES["light"].values()) + ")",
            required=False,
        )
    return Check("Hardware", True, detail, required=False)


def pull_models(names: list[str], host: str) -> None:
    """Download missing models through Ollama's API, with a simple progress line."""
    import json

    for name in names:
        print(f"  pulling {name}...")
        last = -1
        with httpx.stream(
            "POST", f"{host}/api/pull", json={"model": name, "stream": True}, timeout=None
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if msg.get("error"):
                    raise RuntimeError(f"{name}: {msg['error']}")
                if msg.get("total") and msg.get("completed") is not None:
                    pct = int(100 * msg["completed"] / msg["total"])
                    if pct >= last + 10:
                        print(f"    {pct:3d}%  {msg['total'] / 1e9:.1f} GB")
                        last = pct
        print(f"  {name} ready")


def check_models(available: list[str] | None) -> list[Check]:
    s = get_settings()
    roles = [
        (s.planner_model, "plans skills, maps your data"),
        (s.vision_model, "reads screens and documents"),
        (s.embed_model, "skill search, semantic healing"),
    ]
    checks = []
    have = set(available or [])
    have |= {n.removesuffix(":latest") for n in have}
    for model, role in roles:
        if available is None:
            checks.append(
                Check(f"Model {model}", False, "unknown, Ollama is not running", "", True)
            )
            continue
        ok = model in have or model.removesuffix(":latest") in have
        checks.append(
            Check(
                f"Model {model}",
                ok,
                role if ok else f"not pulled ({role})",
                f"mimic doctor --pull  (or: ollama pull {model})",
            )
        )
    return checks


def check_optional() -> list[Check]:
    s = get_settings()
    checks = []
    try:
        r = httpx.get(s.sandbox_url, timeout=2)
        checks.append(
            Check("Demo app", r.is_success, f"running at {s.sandbox_url}", required=False)
        )
    except Exception:  # noqa: BLE001
        checks.append(
            Check("Demo app", False, f"not running at {s.sandbox_url}", "mimic sandbox", False)
        )
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False
    from mimic.matcher import find_model

    local = find_model(download=False)
    if not has_torch:
        detail = "PyTorch not installed; healing uses embeddings and vision instead"
        checks.append(
            Check("UI matcher", False, detail, 'pip install "mimic-agent[matcher]"', False)
        )
    elif local is not None:
        checks.append(Check("UI matcher", True, f"fine-tuned model in {local}", required=False))
    elif s.matcher_repo:
        detail = f"downloaded from huggingface.co/{s.matcher_repo} on first use"
        checks.append(Check("UI matcher", True, detail, required=False))
    else:
        checks.append(Check("UI matcher", False, "no model", "mimic matcher train", False))
    return checks


def run_checks() -> list[Check]:
    ollama, models = check_ollama()
    return [
        check_python(),
        check_hardware(),
        check_storage(),
        check_browser(),
        ollama,
        *check_models(models if ollama.ok else None),
        *check_optional(),
    ]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="mimic doctor", description="Check what Mimic needs")
    parser.add_argument("--pull", action="store_true", help="Download the missing local models")
    args = parser.parse_args(argv)
    print(f"Mimic {__version__}\n")
    if args.pull:
        ollama, available = check_ollama()
        if not ollama.ok:
            print(f"Ollama is {ollama.detail}. {ollama.fix}")
            return 1
        missing = [c.name.removeprefix("Model ") for c in check_models(available) if not c.ok]
        if missing:
            pull_models(missing, get_settings().ollama_host)
            print()
    checks = run_checks()
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        status = "ok" if c.ok else ("missing" if c.required else "optional")
        print(f"  {status:<9}{c.name:<{width}}{c.detail}")
        if not c.ok and c.fix:
            print(f"  {'':<9}{'':<{width}}fix: {c.fix}")
    failed = [c for c in checks if c.required and not c.ok]
    print()
    if failed:
        print(f"{len(failed)} required check(s) failed. Fix them, then run mimic doctor again.")
        return 1
    print("Everything Mimic needs is ready. Try: mimic sandbox, then mimic record enter_invoice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
