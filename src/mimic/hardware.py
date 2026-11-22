"""What this computer can run, and which local models to use on it.

Two profiles of Ollama models:

    standard   qwen2.5vl:7b + qwen2.5:7b   ~6 GB each, best accuracy (GPU or 16 GB+ of RAM)
    light      qwen2.5vl:3b + qwen2.5:3b   ~3 GB each, for laptops without a GPU

MIMIC_PROFILE=auto (default) picks one from the memory found here; VISION_MODEL / PLANNER_MODEL
still override a single model.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass

PROFILES = {
    "standard": {"vision_model": "qwen2.5vl:7b", "planner_model": "qwen2.5:7b"},
    "light": {"vision_model": "qwen2.5vl:3b", "planner_model": "qwen2.5:3b"},
}
STANDARD_MIN_RAM_GB = 16  # without a GPU, two 7B models need this much memory
STANDARD_MIN_VRAM_GB = 8


@dataclass
class Hardware:
    ram_gb: float | None
    gpu: str | None = None
    vram_gb: float | None = None
    unified_memory: bool = False  # Apple Silicon: the GPU uses the RAM

    def describe(self) -> str:
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM unknown"
        if self.unified_memory:
            return f"{ram}, Apple Silicon (shared GPU memory)"
        if self.gpu:
            vram = f", {self.vram_gb:.0f} GB" if self.vram_gb else ""
            return f"{ram}, GPU {self.gpu}{vram}"
        return f"{ram}, no GPU found"


def _nvidia() -> tuple[str | None, float | None]:
    if shutil.which("nvidia-smi") is None:
        return None, None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        name, mib = out.splitlines()[0].rsplit(",", 1)
        return name.strip(), float(mib) / 1024
    except Exception:  # noqa: BLE001
        return None, None


def detect() -> Hardware:
    try:
        import psutil

        ram = psutil.virtual_memory().total / 1024**3
    except Exception:  # noqa: BLE001
        ram = None
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    if apple:
        return Hardware(ram, "Apple Silicon", ram, unified_memory=True)
    gpu, vram = _nvidia()
    return Hardware(ram, gpu, vram)


def recommend(hw: Hardware) -> tuple[str, str]:
    """(profile, why) for this machine."""
    if hw.vram_gb and not hw.unified_memory and hw.vram_gb >= STANDARD_MIN_VRAM_GB:
        return "standard", f"{hw.vram_gb:.0f} GB of GPU memory"
    if hw.ram_gb and hw.ram_gb >= STANDARD_MIN_RAM_GB:
        return "standard", f"{hw.ram_gb:.0f} GB of memory"
    if hw.ram_gb is None:
        return "standard", "memory could not be measured"
    return "light", f"only {hw.ram_gb:.0f} GB of memory and no large GPU"
