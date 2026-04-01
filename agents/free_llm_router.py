"""
ARIA Free LLM Router
=====================
Routes queries to the best available FREE AI model.
Zero cost. No credit card. Falls back gracefully.

Priority order (fastest/best free tier first):
  1. Local Ollama        — always available, fully offline
  2. Groq API            — llama3, mixtral, gemma (14,400 req/day free)
  3. Google Gemini Flash — 1M tokens/day free (Google AI Studio key)
  4. Cloudflare AI       — free Workers AI (llama, mistral)
  5. HuggingFace Inference API — thousands of free models
  6. OpenRouter          — many free models (deepseek, llama, mistral)
  7. Together AI         — free tier with llama models
  8. Cohere              — free tier (1000 API calls/month)

Features:
  - Auto-detects which services have API keys configured in .env
  - Tries each in priority order, falls back on failure/rate-limit
  - Model selection per task type: "fast", "reasoning", "vision", "code", "long"
  - Caches responses (5-min TTL) to avoid repeated API calls
  - Rate limit tracking per provider (auto-backoff)
  - Token counting and cost tracking (always $0 on free tier)
  - Extracts API keys from env: GROQ_API_KEY, GOOGLE_API_KEY, HF_TOKEN,
    OPENROUTER_KEY, TOGETHER_KEY, COHERE_KEY
"""

import os
import sys
import time
import json
import hashlib
import asyncio
import logging
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import requests

# ── Algo-core (graceful fallback) ─────────────────────────────────────────────
try:
    from agents.algo_core import PatternEngine, AdaptiveLearner, DecisionEngine
    _ALGO_CORE = True
except ImportError:
    _ALGO_CORE = False

# ---------------------------------------------------------------------------
# Optional: load .env from project root
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not required

logger = logging.getLogger("aria.free_llm_router")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    provider: str
    context_length: int
    is_free: bool
    best_for: List[str]  # e.g. ["fast", "reasoning", "code", "vision", "long"]

    def supports(self, task_type: str) -> bool:
        return task_type in self.best_for or "fast" in self.best_for


@dataclass
class RouterResult:
    content: str
    provider: str
    model: str
    latency_ms: float
    tokens_used: int
    from_cache: bool
    cost_usd: float = 0.0
    error: Optional[str] = None

    def __bool__(self):
        return bool(self.content and not self.error)


# ---------------------------------------------------------------------------
# Free model catalog
# ---------------------------------------------------------------------------

FREE_MODELS: List[ModelSpec] = [
    # Ollama (local — whatever is installed)
    ModelSpec("llama3.2",      "ollama", 128_000, True, ["fast", "reasoning", "code"]),
    ModelSpec("mistral",       "ollama", 32_768,  True, ["fast", "code"]),
    ModelSpec("phi3",          "ollama", 128_000, True, ["fast", "reasoning"]),
    ModelSpec("gemma2",        "ollama", 8_192,   True, ["fast"]),
    ModelSpec("llava",         "ollama", 4_096,   True, ["vision"]),

    # Groq
    ModelSpec("llama-3.3-70b-versatile",         "groq", 32_768,  True, ["reasoning", "code", "long"]),
    ModelSpec("mixtral-8x7b-32768",              "groq", 32_768,  True, ["fast", "reasoning", "code"]),
    ModelSpec("gemma2-9b-it",                    "groq", 8_192,   True, ["fast"]),
    ModelSpec("llama-3.2-11b-vision-preview",    "groq", 8_192,   True, ["vision"]),
    ModelSpec("llama-3.2-1b-preview",            "groq", 8_192,   True, ["fast"]),

    # Google Gemini
    ModelSpec("gemini-1.5-flash",                "gemini", 1_000_000, True, ["fast", "vision", "long", "reasoning"]),
    ModelSpec("gemini-1.5-flash-8b",             "gemini", 1_000_000, True, ["fast"]),

    # Cloudflare Workers AI (no key needed via public endpoint)
    ModelSpec("@cf/meta/llama-3.1-8b-instruct",        "cloudflare", 8_192,  True, ["fast"]),
    ModelSpec("@cf/mistral/mistral-7b-instruct-v0.1",  "cloudflare", 8_192,  True, ["fast", "code"]),

    # HuggingFace Inference API
    ModelSpec("meta-llama/Llama-3.2-1B-Instruct",   "huggingface", 8_192,  True, ["fast"]),
    ModelSpec("microsoft/DialoGPT-large",            "huggingface", 1_024,  True, ["fast"]),
    ModelSpec("HuggingFaceH4/zephyr-7b-beta",        "huggingface", 4_096,  True, ["fast", "reasoning"]),

    # OpenRouter (free models)
    ModelSpec("deepseek/deepseek-r1:free",                    "openrouter", 163_840, True, ["reasoning", "code"]),
    ModelSpec("meta-llama/llama-3.2-3b-instruct:free",        "openrouter", 131_072, True, ["fast"]),
    ModelSpec("mistralai/mistral-7b-instruct:free",           "openrouter", 32_768,  True, ["fast", "code"]),
    ModelSpec("google/gemma-3-4b-it:free",                    "openrouter", 8_192,   True, ["fast"]),

    # Together AI
    ModelSpec("meta-llama/Llama-3.2-3B-Instruct-Turbo",  "together", 131_072, True, ["fast"]),
    ModelSpec("mistralai/Mixtral-8x7B-Instruct-v0.1",    "together", 32_768,  True, ["reasoning", "code"]),

    # Cohere
    ModelSpec("command-r",    "cohere", 128_000, True, ["reasoning", "long"]),
    ModelSpec("command-light", "cohere", 4_096,  True, ["fast"]),
]

# Best model per task type, per provider (first match used)
TASK_MODEL_PREFERENCE: Dict[str, Dict[str, str]] = {
    "fast": {
        "ollama":      "llama3.2",
        "groq":        "gemma2-9b-it",
        "gemini":      "gemini-1.5-flash-8b",
        "cloudflare":  "@cf/meta/llama-3.1-8b-instruct",
        "huggingface": "meta-llama/Llama-3.2-1B-Instruct",
        "openrouter":  "meta-llama/llama-3.2-3b-instruct:free",
        "together":    "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "cohere":      "command-light",
    },
    "reasoning": {
        "ollama":      "llama3.2",
        "groq":        "llama-3.3-70b-versatile",
        "gemini":      "gemini-1.5-flash",
        "cloudflare":  "@cf/meta/llama-3.1-8b-instruct",
        "huggingface": "HuggingFaceH4/zephyr-7b-beta",
        "openrouter":  "deepseek/deepseek-r1:free",
        "together":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "cohere":      "command-r",
    },
    "code": {
        "ollama":      "mistral",
        "groq":        "llama-3.3-70b-versatile",
        "gemini":      "gemini-1.5-flash",
        "cloudflare":  "@cf/mistral/mistral-7b-instruct-v0.1",
        "huggingface": "HuggingFaceH4/zephyr-7b-beta",
        "openrouter":  "deepseek/deepseek-r1:free",
        "together":    "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "cohere":      "command-r",
    },
    "vision": {
        "ollama":   "llava",
        "groq":     "llama-3.2-11b-vision-preview",
        "gemini":   "gemini-1.5-flash",
    },
    "long": {
        "gemini":     "gemini-1.5-flash",
        "groq":       "llama-3.3-70b-versatile",
        "openrouter": "deepseek/deepseek-r1:free",
        "cohere":     "command-r",
        "ollama":     "llama3.2",
    },
}

# Provider priority for each task type
PROVIDER_PRIORITY: Dict[str, List[str]] = {
    "fast":      ["ollama", "groq", "gemini", "cloudflare", "huggingface", "openrouter", "together", "cohere"],
    "reasoning": ["ollama", "groq", "gemini", "openrouter", "together", "huggingface", "cloudflare", "cohere"],
    "code":      ["ollama", "groq", "openrouter", "gemini", "together", "huggingface", "cloudflare", "cohere"],
    "vision":    ["ollama", "groq", "gemini"],
    "long":      ["gemini", "openrouter", "groq", "cohere", "together", "ollama", "huggingface", "cloudflare"],
}

# ---------------------------------------------------------------------------
# Simple in-memory cache
# ---------------------------------------------------------------------------

class _ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self._store: Dict[str, tuple] = {}  # key -> (result, expire_time)
        self.ttl = ttl_seconds

    def _key(self, prompt: str, system: str, model: str) -> str:
        raw = f"{prompt}||{system}||{model}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, prompt: str, system: str, model: str) -> Optional[str]:
        k = self._key(prompt, system, model)
        entry = self._store.get(k)
        if entry and time.time() < entry[1]:
            return entry[0]
        if entry:
            del self._store[k]
        return None

    def set(self, prompt: str, system: str, model: str, value: str):
        k = self._key(prompt, system, model)
        self._store[k] = (value, time.time() + self.ttl)

    def clear_expired(self):
        now = time.time()
        self._store = {k: v for k, v in self._store.items() if now < v[1]}


# ---------------------------------------------------------------------------
# Rate limit tracker
# ---------------------------------------------------------------------------

class _RateLimitTracker:
    def __init__(self):
        self._backoff: Dict[str, float] = {}  # provider -> epoch when safe to retry

    def is_limited(self, provider: str) -> bool:
        until = self._backoff.get(provider, 0)
        return time.time() < until

    def mark_limited(self, provider: str, seconds: int = 60):
        self._backoff[provider] = time.time() + seconds
        logger.warning(f"[router] {provider} rate-limited, backing off {seconds}s")

    def reset(self, provider: str):
        self._backoff.pop(provider, None)


# ---------------------------------------------------------------------------
# Main router class
# ---------------------------------------------------------------------------

class FreeLLMRouter:
    """
    Routes prompts to the best available free LLM.
    Usage:
        router = FreeLLMRouter()
        result = router.generate("Explain quantum computing", task_type="reasoning")
        print(result.content, result.provider)
    """

    OLLAMA_URL      = "http://localhost:11434"
    GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
    GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    CLOUDFLARE_URL  = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    # Public Cloudflare playground endpoint (no account ID needed, rate-limited)
    CLOUDFLARE_PUBLIC_URL = "https://workers-ai-api.salieri.workers.dev/run/{model}"
    HF_URL          = "https://api-inference.huggingface.co/models/{model}"
    OPENROUTER_URL  = "https://openrouter.ai/api/v1/chat/completions"
    TOGETHER_URL    = "https://api.together.xyz/v1/chat/completions"
    COHERE_URL      = "https://api.cohere.ai/v1/generate"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.cache = _ResponseCache(ttl_seconds=300)
        self.rl = _RateLimitTracker()

        # Load API keys
        self.groq_key        = os.getenv("GROQ_API_KEY", "")
        self.google_key      = os.getenv("GOOGLE_API_KEY", "")
        self.hf_token        = os.getenv("HF_TOKEN", "")
        self.openrouter_key  = os.getenv("OPENROUTER_KEY", "")
        self.together_key    = os.getenv("TOGETHER_KEY", "")
        self.cohere_key      = os.getenv("COHERE_KEY", "")
        # Cloudflare (optional — falls back to public proxy)
        self.cf_account_id   = os.getenv("CF_ACCOUNT_ID", "")
        self.cf_api_token    = os.getenv("CF_API_TOKEN", "")

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "ARIA-Router/1.0"})

        # Total token counter (free tier — always $0)
        self.total_tokens = 0

        # ── Upgrade A: quality history per provider ────────────────────────────
        self._quality_history: Dict[str, List[float]] = {}
        # ── Upgrade B: latency history per provider ────────────────────────────
        self._latency_history: Dict[str, List[float]] = {}
        # Precompute EWMA routing score cache (invalidated on new data)
        self._routing_score_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ── Upgrade B: compute EWMA routing score ─────────────────────────────────

    def _compute_routing_score(self, provider: str) -> float:
        """
        Adaptive routing score: quality * 0.6 + speed * 0.4.
        Both quality and speed are EWMA-smoothed over history.
        Speed is normalised as 1 / (1 + ewma_latency_seconds).
        """
        if not _ALGO_CORE:
            return 0.5

        q_hist = self._quality_history.get(provider, [])
        l_hist = self._latency_history.get(provider, [])

        ewma_quality = AdaptiveLearner.exponential_smoothing(q_hist, alpha=0.3) if q_hist else 0.5
        ewma_latency = AdaptiveLearner.exponential_smoothing(l_hist, alpha=0.3) if l_hist else 1000.0
        # Normalise latency to [0,1]: 0ms = 1.0, 10000ms = ~0.5
        speed_score = 1.0 / (1.0 + ewma_latency / 2000.0)

        # ── Upgrade D: anomaly penalty for latency spikes ──────────────────────
        anomaly_penalty = 0.0
        if l_hist and len(l_hist) >= 3:
            latest_latency = l_hist[-1]
            anomaly = DecisionEngine.anomaly_score(latest_latency, l_hist[:-1])
            if anomaly >= 3.5:  # spike detected
                anomaly_penalty = 0.25  # lower routing weight
                logger.debug(f"[router] {provider} latency anomaly score={anomaly:.2f}, applying penalty")

        return max(0.0, min(1.0, ewma_quality * 0.6 + speed_score * 0.4 - anomaly_penalty))

    def _record_call(self, provider: str, content: str, latency_ms: float) -> None:
        """Record quality and latency for a completed provider call."""
        if not _ALGO_CORE:
            return
        # Quality: complexity + positive sentiment (proxy for richness)
        complexity = PatternEngine.complexity_score(content)
        sentiment  = PatternEngine.sentiment_score(content)
        # Normalise: complexity grade 0–20 → 0–1; sentiment already -1..1 → 0..1
        quality = min(1.0, complexity / 15.0) * 0.7 + (sentiment * 0.5 + 0.5) * 0.3

        self._quality_history.setdefault(provider, []).append(quality)
        self._latency_history.setdefault(provider, []).append(latency_ms)
        # Keep bounded history (last 20 entries)
        self._quality_history[provider] = self._quality_history[provider][-20:]
        self._latency_history[provider] = self._latency_history[provider][-20:]
        # Invalidate score cache
        self._routing_score_cache.pop(provider, None)

    def _adaptive_priority(self, task_type: str) -> List[str]:
        """
        Upgrade B: return providers sorted by EWMA routing score (best first).
        Falls back to static priority if no history.
        """
        base_priority = PROVIDER_PRIORITY.get(task_type, PROVIDER_PRIORITY["fast"])
        if not _ALGO_CORE:
            return base_priority
        scored = []
        for p in base_priority:
            score = self._routing_score_cache.get(p)
            if score is None:
                score = self._compute_routing_score(p)
                self._routing_score_cache[p] = score
            scored.append((p, score))
        # Sort by score descending, but keep providers with no history at their base position
        has_history = {p for p in base_priority if self._quality_history.get(p)}
        known  = sorted([(p, s) for p, s in scored if p in has_history], key=lambda x: -x[1])
        unknown = [(p, s) for p, s in scored if p not in has_history]
        ordered = [p for p, _ in known] + [p for p, _ in unknown]
        return ordered

    def generate(
        self,
        prompt: str,
        system: str = "You are ARIA, a helpful AI assistant.",
        task_type: str = "fast",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> RouterResult:
        """
        Generate a response. Tries providers in adaptive priority order.
        task_type: "fast" | "reasoning" | "code" | "vision" | "long"
        Upgrade B: routing order adapts via EWMA quality + speed scores.
        """
        task_type = task_type if task_type in PROVIDER_PRIORITY else "fast"
        priority = self._adaptive_priority(task_type)

        for provider in priority:
            if self.rl.is_limited(provider):
                logger.debug(f"[router] skipping {provider} (rate-limited)")
                continue

            model = TASK_MODEL_PREFERENCE.get(task_type, {}).get(provider)
            if not model:
                continue

            # Check cache
            cached = self.cache.get(prompt, system, model)
            if cached:
                return RouterResult(
                    content=cached, provider=provider, model=model,
                    latency_ms=0, tokens_used=0, from_cache=True
                )

            # Check key availability
            if not self._has_credentials(provider):
                logger.debug(f"[router] {provider} skipped — no API key")
                continue

            t0 = time.time()
            try:
                content = self._dispatch(provider, prompt, system, model, max_tokens, temperature)
                if content:
                    latency = (time.time() - t0) * 1000
                    tokens = self._estimate_tokens(prompt + content)
                    self.total_tokens += tokens
                    self.cache.set(prompt, system, model, content)
                    self.rl.reset(provider)
                    # ── Upgrade A: record quality/latency ──────────────────────
                    self._record_call(provider, content, latency)
                    return RouterResult(
                        content=content, provider=provider, model=model,
                        latency_ms=round(latency, 1), tokens_used=tokens,
                        from_cache=False
                    )
            except _RateLimitError as e:
                self.rl.mark_limited(provider, seconds=int(e.retry_after or 60))
            except Exception as exc:
                logger.warning(f"[router] {provider}/{model} failed: {exc}")

        # All providers exhausted
        return RouterResult(
            content="", provider="none", model="none",
            latency_ms=0, tokens_used=0, from_cache=False,
            error="All providers failed or unavailable."
        )

    # ── Upgrade C: ensemble generation ────────────────────────────────────────

    def generate_ensemble(
        self,
        prompt: str,
        system: str = "You are ARIA, a helpful AI assistant.",
        task_type: str = "reasoning",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        providers: int = 3,
    ) -> RouterResult:
        """
        Upgrade C: Call top-`providers` providers simultaneously, merge with
        DecisionEngine.ensemble_predict(). Returns merged response + source attribution.
        """
        task_type = task_type if task_type in PROVIDER_PRIORITY else "reasoning"
        priority = self._adaptive_priority(task_type)

        # Pick top N available providers
        selected: List[str] = []
        for p in priority:
            if self.rl.is_limited(p):
                continue
            if not self._has_credentials(p):
                continue
            model = TASK_MODEL_PREFERENCE.get(task_type, {}).get(p)
            if not model:
                continue
            selected.append(p)
            if len(selected) >= providers:
                break

        if not selected:
            return RouterResult(
                content="", provider="none", model="none",
                latency_ms=0, tokens_used=0, from_cache=False,
                error="No providers available for ensemble."
            )

        # Fire all selected providers in parallel
        def _call_provider(prov: str) -> Dict[str, Any]:
            model = TASK_MODEL_PREFERENCE.get(task_type, {}).get(prov, "")
            t0 = time.time()
            try:
                content = self._dispatch(prov, prompt, system, model, max_tokens, temperature)
                latency = (time.time() - t0) * 1000
                if content:
                    self._record_call(prov, content, latency)
                    conf = self._compute_routing_score(prov)
                    return {"content": content, "confidence": conf, "source": prov, "latency_ms": latency}
            except Exception as exc:
                logger.debug(f"[ensemble] {prov} failed: {exc}")
            return {"content": "", "confidence": 0.0, "source": prov, "latency_ms": 0}

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {executor.submit(_call_provider, p): p for p in selected}
            for fut in concurrent.futures.as_completed(futures, timeout=60):
                try:
                    results.append(fut.result())
                except Exception:
                    pass

        if not _ALGO_CORE:
            # Fallback: return first valid result
            for r in results:
                if r.get("content"):
                    return RouterResult(
                        content=r["content"], provider=r["source"], model="ensemble",
                        latency_ms=r.get("latency_ms", 0), tokens_used=0, from_cache=False
                    )
            return RouterResult(content="", provider="none", model="none",
                                latency_ms=0, tokens_used=0, from_cache=False,
                                error="Ensemble: no valid results.")

        merged = DecisionEngine.ensemble_predict(results)
        sources_str = ", ".join(merged.get("sources", []))
        content = merged.get("content", "")
        if content and sources_str:
            content += f"\n\n*Sources: {sources_str}*"

        avg_latency = sum(r.get("latency_ms", 0) for r in results if r.get("content")) / max(
            sum(1 for r in results if r.get("content")), 1
        )
        tokens = self._estimate_tokens(content)
        self.total_tokens += tokens
        return RouterResult(
            content=content,
            provider=f"ensemble:{sources_str}",
            model="ensemble",
            latency_ms=round(avg_latency, 1),
            tokens_used=tokens,
            from_cache=False,
        )

    def generate_vision(
        self,
        prompt: str,
        image_b64: str,
        task_type: str = "vision",
    ) -> RouterResult:
        """Generate a response for a vision (image+text) query."""
        priority = PROVIDER_PRIORITY.get("vision", ["ollama", "groq", "gemini"])

        for provider in priority:
            if self.rl.is_limited(provider):
                continue
            if not self._has_credentials(provider):
                continue

            t0 = time.time()
            try:
                content = self._dispatch_vision(provider, prompt, image_b64)
                if content:
                    latency = (time.time() - t0) * 1000
                    tokens = self._estimate_tokens(prompt + content)
                    self.total_tokens += tokens
                    return RouterResult(
                        content=content, provider=provider,
                        model=TASK_MODEL_PREFERENCE["vision"].get(provider, "unknown"),
                        latency_ms=round(latency, 1), tokens_used=tokens,
                        from_cache=False
                    )
            except _RateLimitError as e:
                self.rl.mark_limited(provider, seconds=int(e.retry_after or 60))
            except Exception as exc:
                logger.warning(f"[router] vision/{provider} failed: {exc}")

        return RouterResult(
            content="", provider="none", model="none",
            latency_ms=0, tokens_used=0, from_cache=False,
            error="No vision provider available."
        )

    def list_available(self) -> List[Dict[str, Any]]:
        """Return list of available providers and their status."""
        results = []
        for provider in ["ollama", "groq", "gemini", "cloudflare", "huggingface", "openrouter", "together", "cohere"]:
            has_creds = self._has_credentials(provider)
            is_limited = self.rl.is_limited(provider)
            models = [m.name for m in FREE_MODELS if m.provider == provider]
            results.append({
                "provider": provider,
                "available": has_creds and not is_limited,
                "has_credentials": has_creds,
                "rate_limited": is_limited,
                "models": models,
            })
        return results

    async def generate_async(
        self,
        prompt: str,
        system: str = "You are ARIA, a helpful AI assistant.",
        task_type: str = "fast",
    ) -> RouterResult:
        """Async wrapper — runs generate() in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate(prompt, system, task_type))

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _has_credentials(self, provider: str) -> bool:
        mapping = {
            "ollama":      True,  # always available locally
            "groq":        bool(self.groq_key),
            "gemini":      bool(self.google_key),
            "cloudflare":  True,  # public endpoint or keyed
            "huggingface": True,  # free tier without key, better with
            "openrouter":  bool(self.openrouter_key),
            "together":    bool(self.together_key),
            "cohere":      bool(self.cohere_key),
        }
        return mapping.get(provider, False)

    def _dispatch(self, provider: str, prompt: str, system: str, model: str,
                  max_tokens: int, temperature: float) -> str:
        if provider == "ollama":
            return self._try_ollama(prompt, system, model, max_tokens, temperature)
        elif provider == "groq":
            return self._try_groq(prompt, system, model, max_tokens, temperature)
        elif provider == "gemini":
            return self._try_gemini(prompt, system, model, max_tokens, temperature)
        elif provider == "cloudflare":
            return self._try_cloudflare(prompt, system, model, max_tokens)
        elif provider == "huggingface":
            return self._try_huggingface(prompt, model, max_tokens)
        elif provider == "openrouter":
            return self._try_openrouter(prompt, system, model, max_tokens, temperature)
        elif provider == "together":
            return self._try_together(prompt, system, model, max_tokens, temperature)
        elif provider == "cohere":
            return self._try_cohere(prompt, model, max_tokens, temperature)
        raise ValueError(f"Unknown provider: {provider}")

    def _dispatch_vision(self, provider: str, prompt: str, image_b64: str) -> str:
        if provider == "ollama":
            return self._try_ollama_vision(prompt, image_b64)
        elif provider == "groq":
            return self._try_groq_vision(prompt, image_b64)
        elif provider == "gemini":
            return self._try_gemini_vision(prompt, image_b64)
        raise ValueError(f"Vision not supported for provider: {provider}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _try_ollama(self, prompt: str, system: str, model: str,
                    max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call local Ollama server."""
        url = f"{self.OLLAMA_URL}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "stream": False,
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        if resp.status_code == 404:
            # Model not installed — try pulling or skip
            raise RuntimeError(f"Ollama model '{model}' not found (404)")
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "").strip()

    def _try_ollama_vision(self, prompt: str, image_b64: str) -> str:
        """Ollama multimodal (llava) call."""
        url = f"{self.OLLAMA_URL}/api/chat"
        payload = {
            "model": "llava",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": False,
        }
        resp = self._session.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()

    def _try_groq(self, prompt: str, system: str, model: str,
                  max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call Groq API (OpenAI-compatible)."""
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._session.post(self.GROQ_URL, headers=headers, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "groq")
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _try_groq_vision(self, prompt: str, image_b64: str) -> str:
        """Groq vision call using llama-3.2-11b-vision-preview."""
        model = "llama-3.2-11b-vision-preview"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
        }
        resp = self._session.post(self.GROQ_URL, headers=headers, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "groq")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _try_gemini(self, prompt: str, system: str, model: str,
                    max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call Google Gemini API."""
        url = self.GEMINI_URL.format(model=model)
        params = {"key": self.google_key}
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        resp = self._session.post(url, params=params, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "gemini")
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates")
        return candidates[0]["content"]["parts"][0]["text"].strip()

    def _try_gemini_vision(self, prompt: str, image_b64: str) -> str:
        """Gemini vision call."""
        model = "gemini-1.5-flash"
        url = self.GEMINI_URL.format(model=model)
        params = {"key": self.google_key}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": 1024},
        }
        resp = self._session.post(url, params=params, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "gemini")
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _try_cloudflare(self, prompt: str, system: str, model: str,
                        max_tokens: int = 512) -> str:
        """
        Call Cloudflare Workers AI.
        Uses keyed endpoint if CF_ACCOUNT_ID + CF_API_TOKEN are set,
        otherwise uses the public community proxy (rate-limited).
        """
        if self.cf_account_id and self.cf_api_token:
            url = self.CLOUDFLARE_URL.format(account_id=self.cf_account_id, model=model)
            headers = {"Authorization": f"Bearer {self.cf_api_token}"}
        else:
            # Public proxy — no auth needed, limited throughput
            url = self.CLOUDFLARE_PUBLIC_URL.format(model=model)
            headers = {}

        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": max_tokens,
        }
        resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "cloudflare")
        resp.raise_for_status()
        data = resp.json()
        # Cloudflare returns {"result": {"response": "..."}, "success": true}
        if data.get("success"):
            return data["result"].get("response", "").strip()
        raise RuntimeError(f"Cloudflare error: {data.get('errors', data)}")

    def _try_huggingface(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        """
        Call HuggingFace Inference API.
        Works without a token (lower rate limits) or with HF_TOKEN.
        """
        url = self.HF_URL.format(model=model)
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "return_full_text": False,
                "temperature": 0.7,
            },
            "options": {"wait_for_model": True},
        }
        resp = self._session.post(url, headers=headers, json=payload, timeout=60)
        self._check_rate_limit(resp, "huggingface")

        if resp.status_code == 503:
            # Model loading — could retry but skip for now
            raise RuntimeError("HuggingFace model is loading (503)")

        resp.raise_for_status()
        data = resp.json()

        # HF returns a list of dicts or a dict depending on model type
        if isinstance(data, list) and data:
            return str(data[0].get("generated_text", "")).strip()
        if isinstance(data, dict):
            return str(data.get("generated_text", data)).strip()
        raise RuntimeError(f"Unexpected HuggingFace response format: {data}")

    def _try_openrouter(self, prompt: str, system: str, model: str,
                        max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call OpenRouter (OpenAI-compatible) with free models."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aria-assistant",  # required by OpenRouter
            "X-Title": "ARIA Assistant",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._session.post(self.OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "openrouter")
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _try_together(self, prompt: str, system: str, model: str,
                      max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call Together AI (OpenAI-compatible)."""
        headers = {
            "Authorization": f"Bearer {self.together_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|eot_id|>", "<|im_end|>"],
        }
        resp = self._session.post(self.TOGETHER_URL, headers=headers, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "together")
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def _try_cohere(self, prompt: str, model: str = "command-light",
                    max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Call Cohere Generate API (v1)."""
        headers = {
            "Authorization": f"Bearer {self.cohere_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "k": 0,
            "stop_sequences": [],
            "return_likelihoods": "NONE",
        }
        resp = self._session.post(self.COHERE_URL, headers=headers, json=payload, timeout=self.timeout)
        self._check_rate_limit(resp, "cohere")
        resp.raise_for_status()
        data = resp.json()
        generations = data.get("generations", [])
        if generations:
            return generations[0].get("text", "").strip()
        raise RuntimeError(f"Cohere returned no generations: {data}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self, resp: requests.Response, provider: str):
        """Raise _RateLimitError if response is a rate limit (429)."""
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", 60)
            raise _RateLimitError(provider=provider, retry_after=retry_after)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def get_stats(self) -> Dict[str, Any]:
        """Return usage stats."""
        available = self.list_available()
        active_providers = [p["provider"] for p in available if p["available"]]
        routing_scores = {p: round(self._compute_routing_score(p), 3) for p in active_providers}
        quality_ewma = {}
        for p, hist in self._quality_history.items():
            if hist and _ALGO_CORE:
                quality_ewma[p] = round(AdaptiveLearner.exponential_smoothing(hist, 0.3), 3)
        return {
            "total_tokens_used": self.total_tokens,
            "estimated_cost_usd": 0.0,
            "active_providers": active_providers,
            "cache_entries": len(self.cache._store),
            "routing_scores": routing_scores,
            "quality_ewma": quality_ewma,
        }


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class _RateLimitError(Exception):
    def __init__(self, provider: str, retry_after: Any = 60):
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(f"Rate limited by {provider}, retry after {retry_after}s")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_router_instance: Optional[FreeLLMRouter] = None


def get_router(timeout: int = 30) -> FreeLLMRouter:
    """
    Returns the global FreeLLMRouter singleton.
    Creates it on first call.

    Example:
        router = get_router()
        result = router.generate("Hello!", task_type="fast")
        print(result.content)
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = FreeLLMRouter(timeout=timeout)
        logger.info("[router] FreeLLMRouter initialized")
    return _router_instance


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="ARIA Free LLM Router CLI")
    parser.add_argument("prompt", nargs="?", default="Say hello and introduce yourself briefly.")
    parser.add_argument("--task", default="fast", choices=["fast", "reasoning", "code", "vision", "long"])
    parser.add_argument("--system", default="You are ARIA, a helpful AI assistant.")
    parser.add_argument("--list", action="store_true", help="List available providers")
    parser.add_argument("--stats", action="store_true", help="Show router stats")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    router = get_router()

    if args.list:
        providers = router.list_available()
        print("\nAvailable Providers:")
        print(f"{'Provider':<15} {'Available':<10} {'Has Key':<10} {'Rate Limited'}")
        print("-" * 50)
        for p in providers:
            print(f"{p['provider']:<15} {str(p['available']):<10} {str(p['has_credentials']):<10} {p['rate_limited']}")
        return

    if args.stats:
        stats = router.get_stats()
        print(json.dumps(stats, indent=2))
        return

    print(f"\nPrompt: {args.prompt}")
    print(f"Task type: {args.task}\n")
    result = router.generate(args.prompt, system=args.system, task_type=args.task)

    if result:
        print(f"[{result.provider} / {result.model}] ({result.latency_ms:.0f}ms, ~{result.tokens_used} tokens)")
        print("-" * 60)
        print(result.content)
    else:
        print(f"ERROR: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    _cli()
