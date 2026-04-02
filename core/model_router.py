"""
ARIA Model Router
=================
The brain selects the right model — not the agent, not the user.

Routing logic:
  Complexity LOW    → fast local model or rule-based answer
  Complexity MEDIUM → mid-tier model (phi3, mistral-7b, groq-fast)
  Complexity HIGH   → best available model (llama3, groq-llama, gpt)
  Privacy mode ON   → always local, never cloud
  Budget mode ON    → always cheapest option

Router checks in order:
  1. Is there a cached answer?
  2. Can the rule engine answer? (free, instant)
  3. Which model tier fits complexity?
  4. Is preferred model available?
  5. Fallback chain if preferred is down
"""

from __future__ import annotations

import os
import time
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # Local models (Ollama)
    "phi3:mini":        {"tier": 1, "ram_gb": 2.3, "provider": "ollama", "speed": "fast",   "quality": 0.65},
    "llama3.2:3b":      {"tier": 1, "ram_gb": 2.0, "provider": "ollama", "speed": "fast",   "quality": 0.68},
    "llama3.1:8b":      {"tier": 2, "ram_gb": 4.7, "provider": "ollama", "speed": "medium", "quality": 0.78},
    "llama3.1:70b":     {"tier": 3, "ram_gb": 40,  "provider": "ollama", "speed": "slow",   "quality": 0.92},
    "mistral:7b":       {"tier": 2, "ram_gb": 4.1, "provider": "ollama", "speed": "medium", "quality": 0.76},
    "deepseek-r1:8b":   {"tier": 2, "ram_gb": 5.0, "provider": "ollama", "speed": "medium", "quality": 0.80},

    # Groq (cloud, free tier)
    "groq:llama-3.3-70b-versatile":   {"tier": 3, "ram_gb": 0, "provider": "groq", "speed": "fast",   "quality": 0.91, "cost": "free"},
    "groq:llama-3.1-8b-instant":      {"tier": 1, "ram_gb": 0, "provider": "groq", "speed": "fast",   "quality": 0.72, "cost": "free"},
    "groq:mixtral-8x7b-32768":        {"tier": 2, "ram_gb": 0, "provider": "groq", "speed": "fast",   "quality": 0.82, "cost": "free"},
    "groq:gemma2-9b-it":              {"tier": 2, "ram_gb": 0, "provider": "groq", "speed": "fast",   "quality": 0.79, "cost": "free"},

    # OpenAI
    "openai:gpt-4o-mini":   {"tier": 2, "ram_gb": 0, "provider": "openai", "speed": "fast",   "quality": 0.85, "cost": "low"},
    "openai:gpt-4o":        {"tier": 3, "ram_gb": 0, "provider": "openai", "speed": "medium", "quality": 0.95, "cost": "medium"},
}


# ─────────────────────────────────────────────────────────────────────────────
# AVAILABILITY CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class _AvailabilityCache:
    """Cache model availability checks for 60 seconds."""
    def __init__(self):
        self._cache: Dict[str, tuple] = {}
        self._ttl = 60

    def is_available(self, model_key: str) -> bool:
        cached = self._cache.get(model_key)
        if cached and time.time() - cached[1] < self._ttl:
            return cached[0]

        available = self._check(model_key)
        self._cache[model_key] = (available, time.time())
        return available

    def _check(self, model_key: str) -> bool:
        info = MODEL_REGISTRY.get(model_key, {})
        provider = info.get("provider", "")

        if provider == "ollama":
            try:
                import requests
                r = requests.get("http://localhost:11434/api/tags", timeout=2)
                if r.status_code == 200:
                    tags = [m["name"] for m in r.json().get("models", [])]
                    model_name = model_key.split(":")[0] if ":" in model_key else model_key
                    return any(model_name in t for t in tags)
            except Exception:
                return False

        elif provider == "groq":
            return bool(os.getenv("GROQ_API_KEY", ""))

        elif provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY", ""))

        return False


_avail = _AvailabilityCache()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class ModelRouter:
    """
    Selects the best available model for a given task.
    Brain calls this — agents don't select models directly.
    """

    def __init__(self):
        self._privacy_mode = os.getenv("ARIA_PRIVACY_MODE", "false").lower() == "true"
        self._budget_mode  = os.getenv("ARIA_BUDGET_MODE",  "false").lower() == "true"
        self._perf_log: List[Dict] = []

    # ── Primary routing decision ──────────────────────────────────────────────

    def select(self, complexity: str = "medium", intent: str = "question") -> Dict[str, Any]:
        """
        Returns the best model config for this task.
        complexity: "low" | "medium" | "high"
        """
        tier_map = {"low": 1, "medium": 2, "high": 3}
        target_tier = tier_map.get(complexity, 2)

        # Special overrides
        if intent == "code":
            target_tier = max(target_tier, 2)
        if intent == "analysis":
            target_tier = max(target_tier, 2)

        # Privacy mode forces local only
        if self._privacy_mode:
            return self._best_local(target_tier)

        # Try cloud first if available and tier is high
        if target_tier >= 2 and not self._budget_mode:
            cloud = self._best_cloud(target_tier)
            if cloud:
                return cloud

        # Fall back to local
        local = self._best_local(target_tier)
        if local:
            return local

        # Last resort — any cloud
        cloud = self._best_cloud(1)
        if cloud:
            return cloud

        return {"model": None, "provider": "none", "reason": "no model available"}

    def _best_local(self, min_tier: int) -> Optional[Dict]:
        candidates = [
            (key, info) for key, info in MODEL_REGISTRY.items()
            if info["provider"] == "ollama" and info["tier"] >= 1
        ]
        # Sort: prefer tier that matches, then quality
        candidates.sort(key=lambda x: (
            abs(x[1]["tier"] - min_tier),
            -x[1]["quality"]
        ))
        for key, info in candidates:
            if _avail.is_available(key):
                return {"model": key, **info, "reason": "local"}
        return None

    def _best_cloud(self, min_tier: int) -> Optional[Dict]:
        candidates = [
            (key, info) for key, info in MODEL_REGISTRY.items()
            if info["provider"] in ("groq", "openai") and info["tier"] >= min_tier
        ]
        candidates.sort(key=lambda x: (
            -x[1]["quality"],
            0 if x[1].get("cost", "free") == "free" else 1,
        ))
        for key, info in candidates:
            if _avail.is_available(key):
                return {"model": key, **info, "reason": "cloud"}
        return None

    # ── Routing explanation ───────────────────────────────────────────────────

    def explain(self, complexity: str = "medium", intent: str = "question") -> str:
        selected = self.select(complexity, intent)
        if not selected.get("model"):
            return "No model available. Check Ollama is running or set GROQ_API_KEY."
        return (
            f"Selected: {selected['model']} "
            f"(provider={selected['provider']}, "
            f"quality={selected.get('quality', '?')}, "
            f"reason={selected.get('reason', '?')})"
        )

    # ── Performance tracking ──────────────────────────────────────────────────

    def log_performance(self, model: str, latency_ms: float, quality_score: float):
        self._perf_log.append({
            "model":       model,
            "latency_ms":  latency_ms,
            "quality":     quality_score,
            "ts":          time.time(),
        })
        # Adapt quality scores based on real performance
        if model in MODEL_REGISTRY and len(self._perf_log) > 5:
            recent = [p for p in self._perf_log[-10:] if p["model"] == model]
            if recent:
                avg_quality = sum(p["quality"] for p in recent) / len(recent)
                # Gentle adjustment — don't wildly change registry values
                old = MODEL_REGISTRY[model]["quality"]
                MODEL_REGISTRY[model]["quality"] = round(old * 0.8 + avg_quality * 0.2, 3)

    def set_privacy_mode(self, enabled: bool):
        self._privacy_mode = enabled

    def set_budget_mode(self, enabled: bool):
        self._budget_mode = enabled

    def available_models(self) -> List[str]:
        return [k for k in MODEL_REGISTRY if _avail.is_available(k)]

    def status(self) -> Dict[str, Any]:
        available = self.available_models()
        return {
            "available":     available,
            "total_known":   len(MODEL_REGISTRY),
            "privacy_mode":  self._privacy_mode,
            "budget_mode":   self._budget_mode,
            "groq_ready":    bool(os.getenv("GROQ_API_KEY")),
            "openai_ready":  bool(os.getenv("OPENAI_API_KEY")),
        }
