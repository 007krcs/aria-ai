"""
NOVA v3 — Async Agent Pool
===========================
N agents run simultaneously. No serial waiting.
Results stream back as each agent completes.

Key design decisions:
- asyncio + ThreadPoolExecutor for true parallel execution
- Each agent is independent — one slow agent doesn't block others
- Model-agnostic: auto-picks best Ollama model for each task type
- Partial results returned on timeout (never hang forever)
- Circuit breaker: unhealthy agents are bypassed automatically
"""

import asyncio
import time
import json
import hashlib
import threading
from typing import Any, Callable, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# AGENT HEALTH / CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentHealth:
    """
    Circuit breaker for each agent.
    If an agent fails too many times, it is bypassed automatically.
    Resets after a cooldown period.
    """
    name:           str
    failures:       int   = 0
    successes:      int   = 0
    last_failure:   float = 0.0
    avg_latency_ms: float = 500.0
    is_open:        bool  = False   # True = circuit open = agent bypassed
    FAILURE_THRESHOLD = 3
    COOLDOWN_SECONDS  = 60

    def record_success(self, latency_ms: float):
        self.successes += 1
        self.failures   = 0
        self.is_open    = False
        # Exponential moving average of latency
        self.avg_latency_ms = 0.8 * self.avg_latency_ms + 0.2 * latency_ms

    def record_failure(self):
        self.failures   += 1
        self.last_failure = time.time()
        if self.failures >= self.FAILURE_THRESHOLD:
            self.is_open = True

    def should_attempt(self) -> bool:
        if not self.is_open:
            return True
        # Auto-reset after cooldown
        if time.time() - self.last_failure > self.COOLDOWN_SECONDS:
            self.is_open  = False
            self.failures = 0
            return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# MODEL-AGNOSTIC SELECTOR
# Picks the best available local model for each task type
# ─────────────────────────────────────────────────────────────────────────────

class ModelSelector:
    """
    Auto-detects all installed Ollama models and picks the best one
    for each task type. Falls back gracefully if the preferred model
    is not installed.

    No hardcoded model names — works with whatever you have installed.
    """

    # Task → preferred model keywords (first match wins)
    PREFERENCES = {
        "code":       ["qwen", "coder", "deepseek", "codellama", "starcoder", "llama3.1", "llama3"],
        "math":       ["qwen", "llama3.1", "llama3", "mistral", "phi3", "phi"],
        "reasoning":  ["llama3.1", "llama3", "qwen", "mistral", "phi3", "phi"],
        "fast":       ["phi3", "phi", "llama3.2", "mistral", "gemma"],
        "multilingual":["qwen", "llama3.1", "mistral", "phi3"],
        "general":    ["llama3.1", "llama3", "qwen", "mistral", "phi3", "phi"],
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url  = base_url
        self._cache:   list[str] = []
        self._lock     = threading.Lock()
        self._last_refresh = 0.0

    def available_models(self) -> list[str]:
        """Return all non-embed models installed in Ollama. Cached for 60s."""
        now = time.time()
        if now - self._last_refresh < 60 and self._cache:
            return self._cache
        try:
            import requests
            r = requests.get(f"{self.base_url}/api/tags", timeout=4)
            models = [m["name"] for m in r.json().get("models", [])]
            models = [m for m in models if "embed" not in m.lower()]
            with self._lock:
                self._cache        = models
                self._last_refresh = now
            return models
        except Exception:
            return self._cache or []

    def best_for(self, task: str) -> str:
        """Return the best available model name for a task."""
        installed  = self.available_models()
        if not installed:
            return "phi3:mini"

        preferences = self.PREFERENCES.get(task, self.PREFERENCES["general"])
        for keyword in preferences:
            for model in installed:
                if keyword.lower() in model.lower():
                    return model

        return installed[0]  # fallback: first available

    def all_for(self, task: str, max_n: int = 4) -> list[str]:
        """Return up to max_n models suitable for a task (for parallel querying)."""
        installed   = self.available_models()
        preferences = self.PREFERENCES.get(task, self.PREFERENCES["general"])
        selected    = []
        for keyword in preferences:
            for model in installed:
                if keyword.lower() in model.lower() and model not in selected:
                    selected.append(model)
            if len(selected) >= max_n:
                break
        return selected or installed[:max_n]


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC AGENT POOL
# ─────────────────────────────────────────────────────────────────────────────

class AsyncAgentPool:
    """
    Runs N agents simultaneously using asyncio.
    Each agent gets its own thread. Results stream back as they complete.

    Usage:
        pool = AsyncAgentPool(engine, memory, logger)
        async for result in pool.run_all(query):
            print(result)  # arrives as each agent finishes

    Or get all results at once:
        results = await pool.run_all_gathered(query, timeout=60)
    """

    def __init__(self, engine, memory, logger, max_workers: int = 8):
        self.engine      = engine
        self.memory      = memory
        self.logger      = logger
        self.selector    = ModelSelector(engine.base_url)
        self.executor    = ThreadPoolExecutor(max_workers=max_workers)
        self.health      = {}   # agent_name → AgentHealth
        self._agent_registry: dict[str, Callable] = {}

    def register_agent(self, name: str, fn: Callable, task_type: str = "general"):
        """Register a callable as a named agent."""
        self._agent_registry[name] = fn
        self.health[name]          = AgentHealth(name=name)
        console.print(f"  [dim]Registered agent:[/] {name}")

    async def run_all(
        self,
        query:      str,
        context:    str = "",
        timeout:    float = 60.0,
        agent_names: Optional[list[str]] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run all (or specified) agents in parallel.
        Yields results as each agent completes — fastest agent first.
        Never waits for a slow agent to block a fast one.
        """
        agents = agent_names or list(self._agent_registry.keys())
        active = [a for a in agents if self.health.get(a, AgentHealth(a)).should_attempt()]

        if not active:
            yield {"type": "error", "error": "All agents in circuit-open state"}
            return

        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()

        async def run_one(name: str):
            t0  = time.time()
            fn  = self._agent_registry[name]
            try:
                result = await loop.run_in_executor(
                    self.executor, lambda: fn(query, context)
                )
                ms = int((time.time() - t0) * 1000)
                self.health[name].record_success(ms)
                await queue.put({
                    "type":    "result",
                    "agent":   name,
                    "result":  result,
                    "ms":      ms,
                    "done":    True,
                })
            except Exception as e:
                self.health[name].record_failure()
                await queue.put({
                    "type":  "agent_error",
                    "agent": name,
                    "error": str(e),
                    "done":  True,
                })

        # Fire all agents simultaneously
        tasks = [asyncio.create_task(run_one(name)) for name in active]
        done_count = 0

        deadline = time.time() + timeout
        while done_count < len(active):
            remaining = deadline - time.time()
            if remaining <= 0:
                yield {"type": "timeout", "agents_pending": len(active) - done_count}
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=min(remaining, 5.0))
                yield item
                done_count += 1
            except asyncio.TimeoutError:
                continue

        # Cancel any still-running tasks
        for t in tasks:
            if not t.done():
                t.cancel()

    async def run_all_gathered(
        self,
        query:   str,
        context: str = "",
        timeout: float = 60.0,
    ) -> dict:
        """
        Run all agents and wait for all to finish (or timeout).
        Returns merged dict of all results.
        """
        results = {}
        errors  = {}

        async for item in self.run_all(query, context, timeout):
            if item["type"] == "result":
                results[item["agent"]] = item["result"]
            elif item["type"] == "agent_error":
                errors[item["agent"]]  = item["error"]

        return {
            "results":    results,
            "errors":     errors,
            "agents_ok":  len(results),
            "agents_err": len(errors),
        }

    def pool_status(self) -> dict:
        """Status of all registered agents and their health."""
        return {
            name: {
                "healthy":       h.should_attempt(),
                "circuit_open":  h.is_open,
                "failures":      h.failures,
                "successes":     h.successes,
                "avg_latency_ms": round(h.avg_latency_ms, 1),
            }
            for name, h in self.health.items()
        }


# ─────────────────────────────────────────────────────────────────────────────
# RESULT MERGER
# Intelligently combines outputs from multiple async agents
# ─────────────────────────────────────────────────────────────────────────────

class ResultMerger:
    """
    Combines results from N parallel agents into a single coherent answer.

    Strategies:
    - BEST:      Pick highest-confidence single answer
    - CONSENSUS: Find agreement across agents
    - SYNTHESIS: LLM synthesises all answers into one
    - FIRST:     Return first completed result (fastest)
    """

    def __init__(self, engine):
        self.engine = engine

    def merge(self, query: str, results: dict, strategy: str = "synthesis") -> dict:
        if not results:
            return {"answer": "No agents returned results", "strategy": strategy, "sources": []}

        answers = {k: v.get("answer", str(v)) for k, v in results.items() if v}

        if strategy == "first" or len(answers) == 1:
            name, ans = next(iter(answers.items()))
            return {"answer": ans, "strategy": "first", "sources": [name]}

        if strategy == "best":
            best_name, best_ans, best_score = "", "", 0.0
            for name, ans in answers.items():
                score = self.engine.score(query, ans)
                if score > best_score:
                    best_score, best_name, best_ans = score, name, ans
            return {"answer": best_ans, "strategy": "best", "sources": [best_name],
                    "confidence": best_score}

        if strategy == "synthesis":
            context = "\n\n".join(
                f"[Agent: {name}]\n{ans[:400]}" for name, ans in answers.items()
            )
            prompt = (
                f"Multiple AI agents answered this question:\n"
                f"Question: {query}\n\n"
                f"{context}\n\n"
                f"Synthesise the best comprehensive answer, "
                f"combining insights from all agents. "
                f"Note any disagreements between agents."
            )
            synthesised = self.engine.generate(prompt, temperature=0.2)
            return {
                "answer":   synthesised,
                "strategy": "synthesis",
                "sources":  list(answers.keys()),
                "raw":      answers,
            }

        # Consensus: find the most common answer
        if strategy == "consensus":
            scores = {name: self.engine.score(query, ans) for name, ans in answers.items()}
            best   = max(scores, key=scores.get)
            return {"answer": answers[best], "strategy": "consensus",
                    "sources": [best], "all_scores": scores}

        return {"answer": list(answers.values())[0], "strategy": "fallback"}
