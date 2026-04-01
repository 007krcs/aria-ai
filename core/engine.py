"""
ARIA — Engine  (v3 — adaptive + cached + timeout-safe)
========================================================
Wraps local Ollama.  Every agent calls this.

Improvements over v2:
  • Auto-selects the best model for the available RAM (via core.adaptive)
  • Response cache — identical prompts return instantly
  • Per-call timeout with configurable limit
  • Graceful Ollama-not-running fallback (no hard SystemExit)
  • Warm-up pings model on first load to avoid 10 s cold-start
  • Thread-safe for concurrent agent use
  • Sanitised error messages — internal paths never leak to callers

Install Ollama:
    Linux/Mac:  curl -fsSL https://ollama.com/install.sh | sh
    Windows:    https://ollama.com/download

Pull a model (choose based on your RAM):
    ollama pull phi3:mini          # 2.3 GB  — any machine
    ollama pull llama3.2:3b        # 2.0 GB  — fast, smart
    ollama pull llama3.1:8b        # 4.7 GB  — 8 GB RAM+
    ollama pull llama3.1:70b       # 40 GB   — 64 GB RAM / GPU
"""

from __future__ import annotations

import re
import json
import time
import hashlib
import threading
from typing import Generator, Optional
from rich.console import Console

# Thread-local storage for per-thread requests sessions (Groq)
_tls = threading.local()

from core.config import (
    OLLAMA_BASE_URL, DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS,
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, LLM_PROVIDER,
)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE LRU RESPONSE CACHE
# ─────────────────────────────────────────────────────────────────────────────

class _ResponseCache:
    """
    In-memory LRU cache for LLM responses.
    Identical (prompt, model, temperature) → instant return.
    TTL: 10 minutes. Max: 256 entries.
    """

    def __init__(self, max_size: int = 256, ttl_s: int = 600):
        self._max   = max_size
        self._ttl   = ttl_s
        self._cache: dict[str, tuple[str, float]] = {}  # key → (response, expiry)
        self._lock  = threading.Lock()

    def _key(self, prompt: str, model: str, temperature: float) -> str:
        raw = f"{model}|{temperature}|{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        key = self._key(prompt, model, temperature)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            response, expiry = entry
            if time.time() > expiry:
                del self._cache[key]
                return None
            return response

    def set(self, prompt: str, model: str, temperature: float, response: str):
        # Don't cache empty, error, or offline responses
        if not response or len(response) < 5:
            return
        _error_phrases = ("offline mode", "ollama is not available", "ollama connection lost",
                          "request timed out", "model error:", "an error occurred")
        if any(p in response.lower() for p in _error_phrases):
            return
        key = self._key(prompt, model, temperature)
        with self._lock:
            # Evict oldest if full
            if len(self._cache) >= self._max:
                oldest = min(self._cache.items(), key=lambda x: x[1][1])
                del self._cache[oldest[0]]
            self._cache[key] = (response, time.time() + self._ttl)

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


_cache = _ResponseCache()


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class Engine:
    """
    Adaptive, cached, timeout-safe wrapper around the local Ollama REST API.

    Auto-selects model:
        Engine()                   → uses adaptive profile model
        Engine(model="phi3:mini")  → uses specified model
        Engine(model="auto")       → explicitly request adaptive selection
    """

    def __init__(
        self,
        model:       str   = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        timeout_s:   int   = 180,
        use_cache:   bool  = True,
    ):
        # Lazy import to avoid circular dependency at module load time
        self._adaptive_model = None
        try:
            from core.adaptive import AdaptiveManager
            mgr = AdaptiveManager.get()
            self._adaptive_model = mgr.get_model()
            # Only let adaptive override if user did NOT explicitly set DEFAULT_MODEL
            import os as _os
            _user_set_model = bool(_os.getenv("DEFAULT_MODEL", "").strip())
            if not _user_set_model and model in (DEFAULT_MODEL, "auto"):
                model = self._adaptive_model
            elif model == "auto":
                model = self._adaptive_model
        except Exception:
            pass  # If adaptive not available, use what was passed

        self.model        = model
        self.temperature  = temperature
        self.base_url     = OLLAMA_BASE_URL
        self.timeout_s    = timeout_s
        self.use_cache    = use_cache
        self._lock        = threading.Lock()

        # Select backend: groq (cloud) or ollama (local)
        if LLM_PROVIDER == "groq" or (LLM_PROVIDER == "auto" and GROQ_API_KEY):
            self._backend = "groq"
            self._available = True
            console.print(f"[green]Engine ready[/] — Groq ({GROQ_MODEL})")
        else:
            self._backend = "ollama"
            self._available = False
            self._verify_connection()

    # ── Connection ─────────────────────────────────────────────────────────────

    def _verify_connection(self):
        """Check Ollama is running. Non-fatal — system degrades gracefully."""
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=4)
            models = [m["name"] for m in r.json().get("models", [])]
            self._available = True

            # Auto-select best available model if the desired one isn't pulled
            if self.model not in models and models:
                best = self._best_available(models)
                console.print(
                    f"  [yellow]Model '{self.model}' not found.[/] "
                    f"Using '{best}' instead. "
                    f"(Run: ollama pull {self.model})"
                )
                self.model = best
            elif not models:
                console.print(
                    "[yellow]No Ollama models installed.[/] "
                    "Run: ollama pull phi3:mini"
                )
            else:
                console.print(
                    f"[green]Engine ready[/] — {self.model} "
                    f"(available: {', '.join(models[:3])}{'...' if len(models) > 3 else ''})"
                )

        except Exception:
            console.print(
                "[yellow]Ollama not running.[/] "
                "ARIA will work in offline/degraded mode. "
                "Start with: ollama serve"
            )
            self._available = False

    @staticmethod
    def _best_available(models: list[str]) -> str:
        """
        Pick the best model from what's installed.
        Priority: fastest-capable first. llama3.2 (3B) beats llama3.1:8b on
        CPU-only hardware — 3× faster first-token at only modest quality cost.
        Reserve 8B+ for deep/code tasks via task routing.
        """
        priority = [
            # Fastest capable models first (tested: <2s first-token when warm)
            "llama3.2",          # 3.2B — best speed/quality for chat on CPU
            "phi3:mini",         # 3.8B — very fast, great for factual tasks
            "phi3",
            # Larger models — use only when routing sends them explicitly
            "llama3.1:70b", "llama3.1:8b",
            "mistral:7b", "gemma2:9b", "phi3:medium",
            "tinyllama",
        ]
        model_names = [m.split(":")[0] for m in models]
        for p in priority:
            pname = p.split(":")[0]
            if pname in model_names:
                # Return full tag
                for m in models:
                    if m.startswith(pname):
                        return m
        return models[0]

    # ── Groq backend ──────────────────────────────────────────────────────────

    @staticmethod
    def _groq_session():
        """Return a thread-local requests.Session for Groq API calls."""
        import requests as _req
        import ssl as _ssl
        if not hasattr(_tls, "session") or _tls.session is None:
            s = _req.Session()
            adapter = _req.adapters.HTTPAdapter(
                pool_connections=4,
                pool_maxsize=8,
                max_retries=0,  # we handle retries ourselves
            )
            s.mount("https://", adapter)
            # Windows: disable SSL verification (no system CA bundle in many installs)
            s.verify = False
            _tls.session = s
        return _tls.session

    def _groq_generate(
        self,
        prompt:      str,
        system:      str           = "",
        model:       Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens:  Optional[int] = None,
        timeout_s:   Optional[int] = None,
    ) -> str:
        """Generate using Groq cloud API (OpenAI-compatible format).

        Uses thread-local sessions + retry-with-backoff to survive transient
        connection failures when called from a uvicorn thread-pool executor.
        """
        import requests as _req
        # When backend is Groq, ignore Ollama model names passed by agents —
        # they are meaningless to the Groq API. Use GROQ_MODEL unless an explicit
        # Groq-compatible model ID is given (contains "-" and digits, e.g. "llama-3.3-70b-versatile").
        _is_groq_model = model and ("-" in model) and any(c.isdigit() for c in model)
        use_model  = model if _is_groq_model else GROQ_MODEL
        use_temp   = temperature if temperature is not None else self.temperature
        use_tokens = max_tokens if max_tokens is not None else MAX_TOKENS
        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()[:4000]})
        messages.append({"role": "user", "content": prompt[:8000]})
        payload = {
            "model":       use_model,
            "messages":    messages,
            "temperature": use_temp,
            "max_tokens":  use_tokens,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }
        use_timeout = timeout_s or self.timeout_s
        url = f"{GROQ_BASE_URL}/chat/completions"
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # First attempt uses session pool; later attempts use direct post
                # to bypass any stale-connection issues in the uvicorn thread pool.
                if attempt == 0:
                    session = self._groq_session()
                    r = session.post(url, json=payload, headers=headers, timeout=use_timeout)
                else:
                    r = _req.post(url, json=payload, headers=headers,
                                  timeout=use_timeout, verify=False)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except _req.exceptions.Timeout:
                console.print(f"[yellow]Groq timeout (attempt {attempt+1}/{max_attempts})[/]")
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return "Request timed out. Try a shorter question."
            except _req.exceptions.HTTPError as e:
                # NOTE: use `is not None` not truthiness — Response[4xx] is falsy in requests
                status = e.response.status_code if e.response is not None else 0
                if status == 429:
                    wait = 2 ** attempt
                    console.print(f"[yellow]Groq rate limit — retrying in {wait}s[/]")
                    if attempt < max_attempts - 1:
                        time.sleep(wait)
                        continue
                    return "Groq rate limit reached. Please wait a moment and try again."
                if status == 404:
                    console.print(f"[red]Groq 404 — check GROQ_MODEL in .env (model '{use_model}' may not exist)[/]")
                    return "An error occurred while processing your request."
                if status in (0, 502, 503, 504):
                    console.print(f"[yellow]Groq HTTP error {status} (attempt {attempt+1}/{max_attempts})[/]")
                    _tls.session = None
                    if attempt < max_attempts - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                console.print(f"[red]Groq HTTP error {status}[/]")
                return "An error occurred while processing your request."
            except (_req.exceptions.ConnectionError, _req.exceptions.ChunkedEncodingError,
                    _req.exceptions.SSLError) as e:
                console.print(f"[yellow]Groq connection error (attempt {attempt+1}/{max_attempts}): {type(e).__name__}: {str(e)[:120]}[/]")
                _tls.session = None
                if attempt < max_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return "An error occurred while processing your request."
            except Exception as _unhandled:
                console.print(f"[red]Groq unhandled {type(_unhandled).__name__} (attempt {attempt+1}): {str(_unhandled)[:120]}[/]")
                if attempt < max_attempts - 1:
                    _tls.session = None
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return "An error occurred while processing your request."
            except Exception as e:
                console.print(f"[red]Groq error:[/] {type(e).__name__}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                    continue
                return "An error occurred while processing your request."
        return "An error occurred while processing your request."

    def _groq_stream(
        self,
        prompt:     str,
        system:     str           = "",
        model:      Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Stream tokens from Groq cloud API."""
        import requests as _req
        messages = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()[:4000]})
        messages.append({"role": "user", "content": prompt[:8000]})
        payload = {
            "model":       model or GROQ_MODEL,
            "messages":    messages,
            "temperature": self.temperature,
            "max_tokens":  max_tokens or MAX_TOKENS,
            "stream":      True,
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type":  "application/json",
        }
        try:
            session = self._groq_session()
            with session.post(
                f"{GROQ_BASE_URL}/chat/completions",
                json=payload, headers=headers,
                stream=True, timeout=self.timeout_s,
            ) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        data = line[6:]
                        if data == b"[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as e:
            yield f"\n[Groq stream error: {type(e).__name__}]"

    def is_available(self) -> bool:
        return self._available

    def list_models(self) -> list[str]:
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # ── Core generation ──────────────────────────────────────────────────────

    def generate(
        self,
        prompt:      str,
        system:      str            = "",
        model:       Optional[str]  = None,
        temperature: Optional[float]= None,
        max_tokens:  Optional[int]  = None,
        max_retries: int            = 3,
        timeout_s:   Optional[int]  = None,
        use_cache:   Optional[bool] = None,
    ) -> str:
        """
        Send a prompt, get a full response.
        Retries on timeout/500. Caches identical calls.
        """
        # Route to Groq if configured
        if self._backend == "groq":
            use_cache_flag = use_cache if use_cache is not None else self.use_cache
            use_temp = temperature if temperature is not None else self.temperature
            if use_cache_flag and use_temp <= 0.4:
                cached = _cache.get(prompt, model or GROQ_MODEL, use_temp)
                if cached is not None:
                    return cached
            result = self._groq_generate(prompt, system, model, temperature, max_tokens, timeout_s)
            if use_cache_flag and use_temp <= 0.4:
                _cache.set(prompt, model or GROQ_MODEL, use_temp, result)
            return result

        if not self._available:
            # Re-check — Ollama may have recovered since the last failure
            try:
                import requests as _req
                _r = _req.get(f"{self.base_url}/api/tags", timeout=5)
                if _r.status_code == 200:
                    self._available = True
                    print(f"[ENGINE] Ollama recovered, resuming. model={self.model}", flush=True)
                else:
                    print(f"[ENGINE] Ollama still down, status={_r.status_code}", flush=True)
                    return "I'm running in offline mode — Ollama is not available."
            except Exception as _e:
                print(f"[ENGINE] Recovery check failed: {type(_e).__name__}: {_e}", flush=True)
                return "I'm running in offline mode — Ollama is not available."

        use_model = model or self.model
        use_temp  = temperature if temperature is not None else self.temperature
        use_cache_flag = use_cache if use_cache is not None else self.use_cache

        # Cache check (skip for high-temperature creative prompts)
        if use_cache_flag and use_temp <= 0.4:
            cached = _cache.get(prompt, use_model, use_temp)
            if cached is not None:
                return cached

        payload = {
            "model":   use_model,
            "prompt":  prompt[:8000],  # hard cap — prevents token overflow
            "stream":  False,
            "options": {
                "temperature": use_temp,
                "num_predict": max_tokens if max_tokens is not None else MAX_TOKENS,
            },
        }
        if system and system.strip():
            payload["system"] = system.strip()[:2000]

        use_timeout = timeout_s or self.timeout_s

        import requests
        for attempt in range(max_retries):
            try:
                with self._lock:
                    r = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=use_timeout,
                    )

                if r.status_code == 500:
                    try:
                        err = r.json().get("error", "Internal error")
                    except Exception:
                        err = "Internal server error"
                    # Sanitise — strip local file paths from error
                    err = re.sub(r"[A-Za-z]:\\[^\s]*|/[^\s]+\.(py|go|bin)", "<path>", err)
                    console.print(f"[yellow]Ollama error:[/] {err}")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    return f"Model error: {err}"

                r.raise_for_status()
                response = r.json()["response"].strip()

                if use_cache_flag and use_temp <= 0.4:
                    _cache.set(prompt, use_model, use_temp, response)

                return response

            except requests.exceptions.Timeout:
                console.print(
                    f"[yellow]Timeout after {use_timeout}s "
                    f"(attempt {attempt+1}/{max_retries})[/]"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return "Request timed out. Try a shorter question or a smaller model."

            except requests.exceptions.ConnectionError:
                # Don't permanently mark offline — Ollama may be temporarily loading a model.
                # Retry up to max_retries, then give a clear error.
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return "Ollama is busy or restarting. Try again in a moment."

            except Exception as e:
                if attempt == max_retries - 1:
                    console.print(f"[red]Engine error:[/] {type(e).__name__}")
                    return "An error occurred while processing your request."
                time.sleep(1)

        return ""

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream(
        self,
        prompt:     str,
        system:     str           = "",
        model:      Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens as generated.
        max_tokens overrides the global MAX_TOKENS — use smaller values for
        live/fast queries to get first-token faster and avoid over-generation.
        Usage:
            for token in engine.stream("explain gravity"):
                print(token, end="", flush=True)
        """
        if self._backend == "groq":
            yield from self._groq_stream(prompt, system, model, max_tokens)
            return

        if not self._available:
            yield "Running in offline mode — Ollama not available."
            return

        import requests
        payload = {
            "model":   model or self.model,
            "prompt":  prompt[:8000],
            "system":  (system or "")[:2000],
            "stream":  True,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_tokens if max_tokens is not None else MAX_TOKENS,
            },
        }
        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=payload, stream=True, timeout=self.timeout_s
            ) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if token := chunk.get("response", ""):
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.ConnectionError:
            yield "\n[Ollama connection lost — try again]"
        except Exception as e:
            yield f"\n[Stream error: {type(e).__name__}]"

    # ── Structured JSON ───────────────────────────────────────────────────────

    def generate_json(self, prompt: str, system: str = "") -> dict:
        """
        Ask for a JSON response. Tries native JSON mode first,
        falls back to text + parse, then returns {"error": ...}.
        """
        json_system = (
            (system + "\n\n" if system else "") +
            "IMPORTANT: Respond ONLY with valid JSON. "
            "No explanation, no markdown, no code fences. Pure JSON only."
        )
        raw = self.generate(prompt, system=json_system, temperature=0.05,
                            use_cache=False)
        raw = raw.strip()

        # Strip code fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        # Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Extract JSON object from within text
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        # Extract JSON array
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass

        return {"error": "Could not parse JSON response", "raw": raw[:200]}

    # ── Classify intent ───────────────────────────────────────────────────────

    def classify(self, text: str, categories: list[str]) -> str:
        """Fast single-token classification. Uses temperature=0."""
        cats   = ", ".join(categories)
        prompt = (
            f"Classify the following text into exactly one category.\n"
            f"Categories: {cats}\n"
            f"Text: {text[:300]}\n"
            f"Reply with the category name only — one word:"
        )
        result = self.generate(prompt, temperature=0.0, timeout_s=30)
        for cat in categories:
            if cat.lower() in result.lower():
                return cat
        return categories[0]

    # ── Score quality ─────────────────────────────────────────────────────────

    def score(self, question: str, answer: str) -> float:
        """Rate answer quality 0.0–1.0. Used by Critic agent."""
        prompt = (
            f"Rate how well this answer addresses the question.\n"
            f"Question: {question[:200]}\n"
            f"Answer: {answer[:400]}\n"
            f"Reply with ONLY a decimal number 0.0–1.0. No other text."
        )
        raw   = self.generate(prompt, temperature=0.0, timeout_s=30)
        match = re.search(r"0?\.\d+|1\.0|^[01]$", raw.strip())
        if match:
            return min(1.0, max(0.0, float(match.group())))
        return 0.5

    # ── Warm up ───────────────────────────────────────────────────────────────

    def warmup(self):
        """
        Pre-load the model so the first real request isn't slow.
        Call this at startup in a background thread.
        """
        if not self._available:
            return
        try:
            import requests
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model":  self.model,
                    "prompt": "hi",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
                timeout=60,
            )
            console.print(f"  [dim]Engine warmed up ({self.model})[/]")
        except Exception:
            pass

    # ── Cache control ─────────────────────────────────────────────────────────

    def clear_cache(self):
        """Clear the response cache."""
        _cache.clear()
        console.print("[dim]Engine response cache cleared.[/]")

    @property
    def cache_size(self) -> int:
        return _cache.size

    def stats(self) -> dict:
        return {
            "model":       self.model,
            "available":   self._available,
            "temperature": self.temperature,
            "timeout_s":   self.timeout_s,
            "cache_size":  _cache.size,
            "adaptive_model": self._adaptive_model,
        }
