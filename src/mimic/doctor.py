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
                f"Model {model}", ok, role if ok else f"not pulled ({role})", f"ollama pull {model}"
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
    matcher = Path("models/element_matcher")
    try:
        import torch  # noqa: F401

        has_torch = True
    except ImportError:
        has_torch = False
    if matcher.exists() and has_torch:
        checks.append(Check("UI matcher", True, f"fine-tuned model in {matcher}", required=False))
    else:
        missing = "PyTorch not installed" if not has_torch else "no model trained yet"
        fix = 'pip install "mimic-agent[matcher]"' if not has_torch else "mimic matcher train"
        detail = f"{missing}; healing uses embeddings and vision instead"
        checks.append(Check("UI matcher", False, detail, fix, False))
    return checks


def run_checks() -> list[Check]:
    ollama, models = check_ollama()
    return [
        check_python(),
        check_storage(),
        check_browser(),
        ollama,
        *check_models(models if ollama.ok else None),
        *check_optional(),
    ]


def main(argv: list[str] | None = None) -> int:
    print(f"Mimic {__version__}\n")
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
