"""
ARIA — Adaptive Resource Manager
==================================
Auto-detects the device's capabilities and configures ARIA to run
optimally on ANY hardware: 1 GB RAM Raspberry Pi → 64 GB workstation.

What it does:
  1. Measures available RAM, CPU cores, GPU presence
  2. Picks the best Ollama model that fits in memory
  3. Adjusts batch sizes, context lengths, embedding model
  4. Monitors runtime memory pressure and downgrades if needed
  5. Exposes a clean profile dict every other module can read

Profiles:
  NANO   — < 2 GB RAM   : phi3:mini / tiny Whisper / no parallelism
  MINI   — 2–4 GB RAM   : phi3:mini / base Whisper / 1 worker
  BASE   — 4–8 GB RAM   : llama3.2:3b / small Whisper / 2 workers
  PLUS   — 8–16 GB RAM  : llama3.1:8b / medium Whisper / 4 workers
  PRO    —  > 16 GB RAM : llama3.1:70b (if available) / large Whisper / 8 workers
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
import platform
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field
from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# HARDWARE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HardwareProfile:
    tier:              str    = "BASE"       # NANO / MINI / BASE / PLUS / PRO
    ram_gb_total:      float  = 4.0
    ram_gb_available:  float  = 2.0
    cpu_cores:         int    = 2
    has_gpu:           bool   = False
    gpu_vram_gb:       float  = 0.0
    platform:          str    = "unknown"
    arch:              str    = "unknown"

    # Model selections
    llm_model:         str    = "phi3:mini"
    fallback_model:    str    = "phi3:mini"
    embedding_model:   str    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    whisper_size:      str    = "base"

    # Tuning knobs
    max_tokens:        int    = 512
    context_length:    int    = 2048
    batch_size:        int    = 1
    worker_threads:    int    = 1
    chunk_size:        int    = 400
    top_k_retrieval:   int    = 5
    temperature:       float  = 0.3

    # Runtime flags
    use_gpu:           bool   = False
    low_memory_mode:   bool   = False
    stream_responses:  bool   = True

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveManager:
    """
    Singleton resource manager.
    Call AdaptiveManager.get() to get the current profile.
    Starts a background monitor that watches memory pressure.
    """

    _instance: Optional["AdaptiveManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "AdaptiveManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._profile: Optional[HardwareProfile] = None
        self._monitor_thread: Optional[threading.Thread] = None
        self._detect()
        self._start_monitor()

    # ── Detection ──────────────────────────────────────────────────────────────

    def _detect(self):
        """Probe hardware and build a profile."""
        ram_total, ram_avail = self._get_ram()
        cpu_cores            = self._get_cpu()
        has_gpu, vram        = self._get_gpu()
        plat                 = platform.system()
        arch                 = platform.machine()

        tier = self._classify_tier(ram_avail, has_gpu, vram)

        # Model selection table
        model_table = {
            "NANO": ("phi3:mini",       "phi3:mini",    "tiny",   512,  1024, 1),
            "MINI": ("phi3:mini",       "phi3:mini",    "base",   512,  2048, 1),
            "BASE": ("llama3.2:3b",     "phi3:mini",    "small",  1024, 4096, 2),
            "PLUS": ("llama3.1:8b",     "llama3.2:3b",  "medium", 2048, 8192, 4),
            "PRO":  ("llama3.1:70b",    "llama3.1:8b",  "large",  4096, 16384, 8),
        }
        llm, fallback, whisper, max_tok, ctx, workers = model_table[tier]

        # Embedding: use smaller model on very low RAM
        emb_model = (
            "sentence-transformers/all-MiniLM-L6-v2"
            if ram_avail < 3
            else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self._profile = HardwareProfile(
            tier=tier,
            ram_gb_total=round(ram_total, 1),
            ram_gb_available=round(ram_avail, 1),
            cpu_cores=cpu_cores,
            has_gpu=has_gpu,
            gpu_vram_gb=round(vram, 1),
            platform=plat,
            arch=arch,
            llm_model=llm,
            fallback_model=fallback,
            embedding_model=emb_model,
            whisper_size=whisper,
            max_tokens=max_tok,
            context_length=ctx,
            batch_size=min(4, max(1, cpu_cores // 2)),
            worker_threads=workers,
            chunk_size=300 if tier == "NANO" else 400,
            top_k_retrieval=3 if tier in ("NANO","MINI") else 5,
            temperature=0.3,
            use_gpu=has_gpu,
            low_memory_mode=tier in ("NANO", "MINI"),
            stream_responses=True,
        )

        console.print(
            f"  [green]Adaptive profile:[/] [bold]{tier}[/] — "
            f"{ram_avail:.1f} GB available RAM, "
            f"{cpu_cores} CPUs, "
            f"{'GPU ' + str(round(vram,1)) + ' GB' if has_gpu else 'CPU only'}"
        )
        console.print(
            f"  [dim]LLM: {llm} | Whisper: {whisper} | Workers: {workers}[/]"
        )

    @staticmethod
    def _classify_tier(ram_gb: float, has_gpu: bool, vram_gb: float) -> str:
        # If GPU available, bump tier by 1 level
        effective = ram_gb + (vram_gb * 0.5 if has_gpu else 0)
        if effective < 2:   return "NANO"
        if effective < 4:   return "MINI"
        if effective < 8:   return "BASE"
        if effective < 16:  return "PLUS"
        return "PRO"

    @staticmethod
    def _get_ram() -> tuple[float, float]:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return vm.total / (1024**3), vm.available / (1024**3)
        except ImportError:
            # Fallback via /proc/meminfo on Linux or systeminfo on Windows
            try:
                if platform.system() == "Linux":
                    with open("/proc/meminfo") as f:
                        lines = {l.split(":")[0]: int(l.split()[1])
                                 for l in f if ":" in l}
                    total = lines.get("MemTotal", 4_000_000) / (1024**2)
                    avail = lines.get("MemAvailable", total / 2) / (1024**2)
                    return total, avail
            except Exception:
                pass
            return 4.0, 2.0

    @staticmethod
    def _get_cpu() -> int:
        try:
            import psutil
            return psutil.cpu_count(logical=False) or os.cpu_count() or 1
        except Exception:
            return os.cpu_count() or 2

    @staticmethod
    def _get_gpu() -> tuple[bool, float]:
        # NVIDIA via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            h    = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            return True, mem.total / (1024**3)
        except Exception:
            pass
        # PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return True, vram
        except Exception:
            pass
        # No GPU detected
        return False, 0.0

    # ── Runtime monitor ────────────────────────────────────────────────────────

    def _start_monitor(self):
        """Background thread that checks memory pressure every 60 s."""
        def _monitor():
            while True:
                try:
                    time.sleep(60)
                    _, avail = self._get_ram()
                    p = self._profile
                    if p is None:
                        continue
                    p.ram_gb_available = round(avail, 1)

                    # If we're critically low, switch to fallback model
                    if avail < 0.8 and not p.low_memory_mode:
                        console.print(
                            f"  [yellow]Memory pressure![/] {avail:.1f} GB left — "
                            f"switching to fallback model: {p.fallback_model}"
                        )
                        p.llm_model      = p.fallback_model
                        p.low_memory_mode = True
                        p.max_tokens     = min(p.max_tokens, 512)
                    elif avail > 1.5 and p.low_memory_mode:
                        # Recovered — restore
                        p.low_memory_mode = False
                        console.print(
                            f"  [green]Memory recovered[/] ({avail:.1f} GB) — restoring full model"
                        )
                        self._detect()   # re-detect and reset profile
                except Exception:
                    pass

        self._monitor_thread = threading.Thread(
            target=_monitor, daemon=True, name="aria-resource-monitor"
        )
        self._monitor_thread.start()

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def profile(self) -> HardwareProfile:
        return self._profile

    @property
    def tier(self) -> str:
        return self._profile.tier if self._profile else "BASE"

    def get_model(self) -> str:
        """Return the recommended LLM model for current hardware."""
        return self._profile.llm_model if self._profile else "phi3:mini"

    def get_fallback_model(self) -> str:
        return self._profile.fallback_model if self._profile else "phi3:mini"

    def max_tokens(self) -> int:
        return self._profile.max_tokens if self._profile else 512

    def worker_threads(self) -> int:
        return self._profile.worker_threads if self._profile else 1

    def is_low_memory(self) -> bool:
        return self._profile.low_memory_mode if self._profile else False

    def whisper_size(self) -> str:
        return self._profile.whisper_size if self._profile else "base"

    def embedding_model(self) -> str:
        return self._profile.embedding_model if self._profile else (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def chunk_size(self) -> int:
        return self._profile.chunk_size if self._profile else 400

    def top_k(self) -> int:
        return self._profile.top_k_retrieval if self._profile else 5

    def summary(self) -> dict:
        return self._profile.to_dict() if self._profile else {}

    def force_tier(self, tier: str):
        """
        Manually override the detected tier.
        Useful for testing or for embedded deployments with known hardware.
        """
        valid = ("NANO", "MINI", "BASE", "PLUS", "PRO")
        if tier not in valid:
            raise ValueError(f"tier must be one of {valid}")
        self._profile.tier = tier
        console.print(f"  [yellow]Tier manually forced to:[/] {tier}")


# ── Module-level convenience ─────────────────────────────────────────────────

def get_profile() -> HardwareProfile:
    """Quick access to the current hardware profile."""
    return AdaptiveManager.get().profile


def get_model() -> str:
    """Quick access to the recommended LLM model."""
    return AdaptiveManager.get().get_model()
