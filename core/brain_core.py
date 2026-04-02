"""
ARIA BrainCore — Central Cognitive Controller
==============================================
The single authority that processes every request.

Architecture rule:
  - Nothing answers the user without going through BrainCore
  - Agents are tools, not decision makers
  - Models are reasoning engines, not the identity
  - Memory is the persistent self

Pipeline for every request:
  1. Intent parsing        — what does the user actually want?
  2. Context building      — what do we know (memory + goals + history)?
  3. Confidence estimation — can we answer without a model?
  4. Model routing         — which engine fits this task?
  5. Agent delegation      — only if a specialist is genuinely needed
  6. Response composition  — shape the final answer
  7. Self-reflection       — was that good? what to remember?
  8. Memory writing        — store what matters
"""

from __future__ import annotations

import time
import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── lazy imports ─────────────────────────────────────────────────────────────
def _get_memory_hierarchy():
    try:
        from core.memory_hierarchy import MemoryHierarchy
        return MemoryHierarchy()
    except Exception:
        return None

def _get_goal_engine():
    try:
        from core.goal_engine import GoalEngine
        return GoalEngine()
    except Exception:
        return None

def _get_model_router():
    try:
        from core.model_router import ModelRouter
        return ModelRouter()
    except Exception:
        return None

def _get_reflector():
    try:
        from core.self_reflection import SelfReflector
        return SelfReflector()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INTENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

class IntentParser:
    """
    Classifies user intent without calling any model.
    Fast regex + keyword matching — never blocks.
    """

    INTENT_MAP = {
        "question":     ["what", "why", "how", "when", "where", "who", "which", "explain", "tell me"],
        "task":         ["open", "create", "make", "build", "write", "send", "set", "schedule", "remind"],
        "search":       ["find", "search", "look up", "google", "research", "latest", "news"],
        "analysis":     ["analyze", "analyse", "compare", "evaluate", "review", "assess", "examine"],
        "calculation":  ["calculate", "compute", "solve", "math", "equation", "formula"],
        "code":         ["code", "script", "function", "debug", "program", "python", "javascript"],
        "memory":       ["remember", "recall", "what did i", "earlier", "last time", "history"],
        "preference":   ["i prefer", "i like", "i want", "my favorite", "always use", "don't use"],
        "correction":   ["wrong", "incorrect", "fix this", "that's not right", "actually"],
        "stock":        ["stock", "price", "share", "market", "nse", "bse", "crypto"],
        "weather":      ["weather", "temperature", "forecast", "rain", "humidity"],
        "voice":        ["say", "speak", "read aloud", "narrate"],
        "system":       ["shutdown", "restart", "open app", "volume", "brightness", "wifi"],
    }

    COMPLEXITY = {
        "question":     "medium",
        "task":         "low",
        "search":       "low",
        "analysis":     "high",
        "calculation":  "medium",
        "code":         "high",
        "memory":       "low",
        "preference":   "low",
        "correction":   "low",
        "stock":        "low",
        "weather":      "low",
        "voice":        "low",
        "system":       "low",
    }

    def parse(self, query: str) -> Dict[str, Any]:
        q = query.lower().strip()
        intent = "question"
        confidence = 0.5

        for name, keywords in self.INTENT_MAP.items():
            hits = sum(1 for kw in keywords if kw in q)
            if hits > 0:
                score = hits / len(keywords) + (0.3 if q.startswith(keywords[0]) else 0)
                if score > confidence:
                    confidence = min(score, 0.99)
                    intent = name

        return {
            "intent": intent,
            "complexity": self.COMPLEXITY.get(intent, "medium"),
            "confidence": round(confidence, 2),
            "query_len": len(q.split()),
            "has_question_mark": "?" in query,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SELF MODEL — ARIA's persistent identity
# ─────────────────────────────────────────────────────────────────────────────

class SelfModel:
    """
    ARIA's concept of itself.
    Not consciousness — structured identity that shapes behavior.
    """

    PATH = PROJECT_ROOT / "data" / "aria_self_model.json"

    DEFAULTS = {
        "identity": {
            "name": "ARIA",
            "full_name": "Adaptive Reasoning Intelligence Architecture",
            "role": "Personal AI assistant and cognitive partner",
            "version": "2.0",
        },
        "capabilities": [
            "reasoning", "research", "coding", "analysis",
            "voice", "system control", "memory", "planning",
            "stock analysis", "document processing",
        ],
        "personality": {
            "tone": "direct and clear",
            "verbosity": "concise",
            "style": "professional but friendly",
            "language": "plain english, no jargon unless asked",
        },
        "operating_rules": [
            "Never fabricate facts. Say 'I don't know' when uncertain.",
            "Always cite sources for factual claims.",
            "Prefer concise answers. Expand only when asked.",
            "Respect user corrections immediately.",
            "Never repeat the same mistake twice in a session.",
        ],
        "uncertainty_threshold": 0.65,
        "created_at": datetime.now().isoformat(),
        "interaction_count": 0,
    }

    def __init__(self):
        self.PATH.parent.mkdir(exist_ok=True)
        if self.PATH.exists():
            try:
                self._data = json.loads(self.PATH.read_text())
            except Exception:
                self._data = dict(self.DEFAULTS)
        else:
            self._data = dict(self.DEFAULTS)
            self._save()

    def _save(self):
        self.PATH.write_text(json.dumps(self._data, indent=2))

    def get_system_prompt(self) -> str:
        ident = self._data["identity"]
        pers  = self._data["personality"]
        rules = self._data["operating_rules"]
        return (
            f"You are {ident['name']} ({ident['full_name']}). {ident['role']}.\n"
            f"Tone: {pers['tone']}. Style: {pers['style']}. Verbosity: {pers['verbosity']}.\n"
            f"Rules:\n" + "\n".join(f"- {r}" for r in rules)
        )

    def tick(self):
        self._data["interaction_count"] += 1
        if self._data["interaction_count"] % 10 == 0:
            self._save()

    @property
    def uncertainty_threshold(self) -> float:
        return self._data.get("uncertainty_threshold", 0.65)

    def update_preference(self, key: str, value: Any):
        self._data["personality"][key] = value
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# ARIA BRAIN CORE
# ─────────────────────────────────────────────────────────────────────────────

class AriaBrainCore:
    """
    The central cognitive authority.

    Every request flows through here. No agent answers without brain approval.
    """

    _instance: Optional["AriaBrainCore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.intent_parser = IntentParser()
        self.self_model    = SelfModel()

        # Lazy — only init when first needed
        self._memory   = None
        self._goals    = None
        self._router   = None
        self._reflector = None

        # Session working memory (in-process, not persisted)
        self._session: List[Dict] = []
        self._session_start = datetime.now().isoformat()

    @property
    def memory(self):
        if self._memory is None:
            self._memory = _get_memory_hierarchy()
        return self._memory

    @property
    def goals(self):
        if self._goals is None:
            self._goals = _get_goal_engine()
        return self._goals

    @property
    def router(self):
        if self._router is None:
            self._router = _get_model_router()
        return self._router

    @property
    def reflector(self):
        if self._reflector is None:
            self._reflector = _get_reflector()
        return self._reflector

    # ── Public API ───────────────────────────────────────────────────────────

    def process(self, query: str, engine=None, neural=None) -> str:
        """
        Synchronous single-shot processing.
        Returns the final answer string.
        """
        ctx = self._build_context(query)
        answer = self._generate(query, ctx, engine, neural)
        self._post_process(query, answer, ctx)
        return answer

    async def stream(self, query: str, engine=None, neural=None) -> AsyncGenerator[str, None]:
        """
        Async streaming processing.
        Yields SSE-formatted data chunks.
        """
        ctx = self._build_context(query)

        # Yield thinking signal
        yield f'data: {json.dumps({"type": "thinking", "text": "..."})}\n\n'

        answer = ""
        try:
            if neural and hasattr(neural, "stream"):
                async for chunk in neural.stream(query):
                    yield chunk
                    # accumulate for post-processing
                    if '"type": "done"' in chunk or '"type":"done"' in chunk:
                        try:
                            d = json.loads(chunk.replace("data: ", "").strip())
                            answer = d.get("text", answer)
                        except Exception:
                            pass
            else:
                answer = self._generate_sync(query, ctx, engine)
                yield f'data: {json.dumps({"type": "token", "text": answer})}\n\n'
                yield f'data: {json.dumps({"type": "done",  "text": answer})}\n\n'
        finally:
            if answer:
                self._post_process(query, answer, ctx)

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, query: str) -> Dict[str, Any]:
        intent_data = self.intent_parser.parse(query)

        # Retrieve relevant memories
        memory_ctx = ""
        if self.memory:
            try:
                hits = self.memory.search(query, k=3)
                if hits:
                    memory_ctx = "Relevant memory:\n" + "\n".join(
                        f"- {h['text']}" for h in hits if h.get("text")
                    )
            except Exception:
                pass

        # Active goals
        goal_ctx = ""
        if self.goals:
            try:
                active = self.goals.get_active()
                if active:
                    goal_ctx = "Active goals: " + "; ".join(g["goal"] for g in active[:3])
            except Exception:
                pass

        # Session history (last 3 turns)
        history_ctx = ""
        if self._session:
            last = self._session[-3:]
            history_ctx = "Recent conversation:\n" + "\n".join(
                f"User: {t['query']}\nARIA: {t['answer'][:200]}" for t in last
            )

        return {
            "intent":     intent_data,
            "memory":     memory_ctx,
            "goals":      goal_ctx,
            "history":    history_ctx,
            "system":     self.self_model.get_system_prompt(),
            "timestamp":  datetime.now().isoformat(),
        }

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate(self, query: str, ctx: Dict, engine, neural) -> str:
        if neural and hasattr(neural, "stream"):
            # Run async stream synchronously
            try:
                loop = asyncio.new_event_loop()
                answer = loop.run_until_complete(self._collect_stream(query, neural))
                loop.close()
                return answer
            except Exception:
                pass

        return self._generate_sync(query, ctx, engine)

    async def _collect_stream(self, query: str, neural) -> str:
        answer = ""
        async for chunk in neural.stream(query):
            if '"type": "done"' in chunk or '"type":"done"' in chunk:
                try:
                    d = json.loads(chunk.replace("data: ", "").strip())
                    answer = d.get("text", answer)
                except Exception:
                    pass
        return answer

    def _generate_sync(self, query: str, ctx: Dict, engine) -> str:
        if engine is None:
            return "I'm initializing. Please try again in a moment."

        # Build enriched prompt with context
        parts = [ctx["system"]]
        if ctx["memory"]:
            parts.append(ctx["memory"])
        if ctx["goals"]:
            parts.append(ctx["goals"])
        if ctx["history"]:
            parts.append(ctx["history"])

        system_prompt = "\n\n".join(parts)

        try:
            return engine.generate(query, system=system_prompt)
        except Exception as e:
            return f"I encountered an issue processing that request. ({type(e).__name__})"

    # ── Post processing ───────────────────────────────────────────────────────

    def _post_process(self, query: str, answer: str, ctx: Dict):
        # 1. Store in session working memory
        self._session.append({
            "query":     query,
            "answer":    answer,
            "intent":    ctx["intent"]["intent"],
            "timestamp": ctx["timestamp"],
        })
        # Keep session at max 20 turns
        if len(self._session) > 20:
            self._session = self._session[-20:]

        # 2. Tick self-model interaction counter
        self.self_model.tick()

        # 3. Write to long-term memory (async, non-blocking)
        if self.memory:
            try:
                self.memory.write_episodic(
                    f"Q: {query[:200]}\nA: {answer[:500]}",
                    metadata={"intent": ctx["intent"]["intent"], "ts": ctx["timestamp"]}
                )
            except Exception:
                pass

        # 4. Extract preferences from corrections
        intent = ctx["intent"]["intent"]
        if intent == "preference":
            if self.memory:
                try:
                    self.memory.write_semantic(
                        f"User preference: {query}",
                        metadata={"type": "preference", "ts": ctx["timestamp"]}
                    )
                except Exception:
                    pass

        # 5. Self-reflect (background, non-blocking)
        if self.reflector:
            try:
                self.reflector.reflect_async(query, answer, ctx["intent"])
            except Exception:
                pass

    # ── Utilities ─────────────────────────────────────────────────────────────

    def remember(self, text: str, memory_type: str = "semantic"):
        """Explicitly store something in memory."""
        if self.memory:
            if memory_type == "semantic":
                self.memory.write_semantic(text)
            elif memory_type == "episodic":
                self.memory.write_episodic(text)
            elif memory_type == "procedural":
                self.memory.write_procedural(text)

    def forget(self, query: str):
        """Remove memories matching a query."""
        if self.memory:
            self.memory.forget(query)

    def get_session_summary(self) -> str:
        """Summarize the current session."""
        if not self._session:
            return "No conversation yet in this session."
        turns = len(self._session)
        intents = [t["intent"] for t in self._session]
        from collections import Counter
        top = Counter(intents).most_common(3)
        return (
            f"Session: {turns} turns since {self._session_start[:16]}. "
            f"Top intents: {', '.join(f'{i}({c})' for i, c in top)}."
        )

    def set_goal(self, goal: str, goal_type: str = "session"):
        if self.goals:
            self.goals.add(goal, goal_type)

    def clear_session(self):
        self._session = []
        self._session_start = datetime.now().isoformat()


# ── Singleton accessor ────────────────────────────────────────────────────────
_brain: Optional[AriaBrainCore] = None

def get_brain() -> AriaBrainCore:
    global _brain
    if _brain is None:
        _brain = AriaBrainCore()
    return _brain
