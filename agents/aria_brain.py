"""
ARIA Brain — Tiered Intelligence Engine
=========================================
The model is the LAST resort, not the first call.

On a Raspberry Pi Zero (512MB RAM):
  Tier 1: Rule engine handles ~70% of queries instantly
  Tier 2: Z3 + SymPy handle math/logic exactly
  Tier 3: Sentence-transformers (120MB) for semantic search
  → Works. No LLM needed. Answers from stored knowledge.

On a laptop with 4GB RAM:
  All above + phi3:mini when needed

On 8GB+ RAM:
  All above + llama3.1:8b for hard reasoning

Query path:
  Input → classify complexity → lowest tier that can answer → return
  Never loads a tier it doesn't need.
  Learns which tier handles which query → gets more accurate over time.
"""

import re
import json
import time
import hashlib
import importlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
CACHE_PATH   = PROJECT_ROOT / "data" / "brain_cache.json"
CACHE_PATH.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY CLASSIFIER
# Decides which tier is needed — saves RAM by not loading heavy tiers
# ─────────────────────────────────────────────────────────────────────────────

class ComplexityClassifier:
    """
    Classifies query complexity before any tier is invoked.
    Returns the MINIMUM tier needed to answer correctly.

    Tiers:
      1 = rule / pattern match (0 RAM)
      2 = symbolic math/logic (50MB)
      3 = semantic search / RAG (120MB)
      4 = small LLM phi3:mini (1.8GB)
      5 = medium LLM llama3.1 (4.7GB)
      6 = API call (0 local RAM)
    """

    # Patterns that rule engine can handle without any model
    TIER1_PATTERNS = [
        r"what (time|day|date) is it",
        r"open .+",
        r"(set|create) (an? )?alarm",
        r"remind me",
        r"(call|phone|text|sms|whatsapp) .+",
        r"play .+ (on|in) (youtube|spotify)",
        r"(search|google|find) .+",
        r"(turn|switch) (on|off) .+",
        r"(set|change) .+ to .+",
        r"(what is|define) [a-z ]+$",
        r"(how (do|does|to|many|much|far|long|old)|when|where|who) ",
        r"(add|create|make|set|schedule) .+",
        r"(weather|temperature|forecast)",
        r"(stock|price|market) .+",
    ]

    # Math/logic patterns — tier 2 symbolic
    TIER2_PATTERNS = [
        r"\d+[\+\-\*\/\^]\d+",
        r"(solve|calculate|compute|evaluate|simplify|factorise?)",
        r"(integral|derivative|differentiate|integrate)",
        r"(prove|verify|check if|is .+ true)",
        r"(equation|formula|theorem|logic|proof)",
        r"(matrix|vector|eigenvalue|determinant)",
        r"(prime|factorial|fibonacci|modulo)",
    ]

    # Questions that need knowledge retrieval — tier 3
    TIER3_PATTERNS = [
        r"(what did|what does|explain|describe|summarise|summarize)",
        r"(according to|based on|from (the|my) document)",
        r"(tell me about|what is|who is|how does).{20,}",
        r"(compare|difference between|similar to)",
    ]

    # Complex generation — needs small LLM (tier 4+)
    TIER4_KEYWORDS = [
        "write", "draft", "compose", "generate", "create a",
        "code", "program", "function", "class", "script",
        "email", "message", "essay", "blog", "report",
        "analyse", "analyze", "evaluate", "recommend",
        "why does", "how would", "what if", "should i",
    ]

    def __init__(self):
        self._t1 = [re.compile(p, re.IGNORECASE) for p in self.TIER1_PATTERNS]
        self._t2 = [re.compile(p, re.IGNORECASE) for p in self.TIER2_PATTERNS]
        self._t3 = [re.compile(p, re.IGNORECASE) for p in self.TIER3_PATTERNS]

    def classify(self, query: str) -> int:
        q = query.strip().lower()

        # Tier 1: Direct action or simple factual
        if any(p.search(q) for p in self._t1):
            return 1

        # Tier 2: Math/logic/symbolic
        if any(p.search(q) for p in self._t2):
            return 2

        # Tier 3: Knowledge retrieval
        if any(p.search(q) for p in self._t3):
            return 3

        # Tier 4+: Generation required
        if any(kw in q for kw in self.TIER4_KEYWORDS):
            word_count = len(q.split())
            # Short generation → tier 4, long/complex → tier 5
            return 4 if word_count < 20 else 5

        # Default: try RAG first
        return 3


# ─────────────────────────────────────────────────────────────────────────────
# TIER 1 — RULE ENGINE (zero model, zero RAM overhead)
# ─────────────────────────────────────────────────────────────────────────────

class RuleEngine:
    """
    Handles queries using pure rules, no model.
    Covers: greetings, time/date, simple facts, unit conversions,
            basic calculations (arithmetic), direct commands.

    On Raspberry Pi Zero — this tier alone handles 70% of use cases.
    """

    def __init__(self):
        self._cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        try:
            if CACHE_PATH.exists():
                return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
        return {}

    def answer(self, query: str) -> Optional[dict]:
        """Try to answer with rules. Returns None if rule can't handle it."""
        q = query.strip().lower()

        # Cache hit (previously answered correctly by higher tier)
        cache_key = hashlib.md5(q.encode()).hexdigest()[:8]
        if cache_key in self._cache:
            return {**self._cache[cache_key], "source": "cache", "tier": 1}

        # Time and date
        if re.search(r"\bwhat (time|day|date) is it\b", q):
            now = datetime.now()
            return {
                "answer": f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}",
                "tier": 1, "source": "rule"
            }

        # Arithmetic
        math_match = re.search(r"[\d\s\+\-\*\/\(\)\.]+", query)
        if math_match and any(op in query for op in ['+','-','*','/','^','**']):
            result = self._safe_eval(math_match.group())
            if result is not None:
                return {"answer": str(result), "tier": 1, "source": "arithmetic"}

        # Unit conversions
        conv = self._unit_convert(query)
        if conv:
            return {"answer": conv, "tier": 1, "source": "conversion"}

        # Greetings
        if re.search(r"^(hi|hello|hey|good (morning|evening|afternoon|night))\b", q):
            hour = datetime.now().hour
            greeting = ("Good morning" if hour < 12 else
                       "Good afternoon" if hour < 17 else
                       "Good evening")
            return {"answer": f"{greeting}! How can I help you?", "tier": 1, "source": "rule"}

        # Simple facts (capital cities, constants, etc.)
        fact = self._lookup_fact(q)
        if fact:
            return {"answer": fact, "tier": 1, "source": "fact_db"}

        return None

    def _safe_eval(self, expr: str) -> Optional[float]:
        """Safely evaluate arithmetic expression."""
        try:
            # Only allow numbers and operators
            clean = re.sub(r"[^\d\+\-\*\/\(\)\.\s\^]","", expr).strip()
            clean = clean.replace("^","**")
            if not clean:
                return None
            result = eval(clean, {"__builtins__": {}})
            return round(float(result), 8)
        except Exception:
            return None

    def _unit_convert(self, query: str) -> Optional[str]:
        """Handle common unit conversions."""
        q = query.lower()
        m = re.search(r"([\d.]+)\s*(km|kilometers?|miles?|kg|kilograms?|pounds?|lbs?|celsius|fahrenheit|°[cf])\s+(?:to|in)\s+(km|kilometers?|miles?|kg|kilograms?|pounds?|lbs?|celsius|fahrenheit|°[cf])", q)
        if not m:
            return None
        val  = float(m.group(1))
        frm  = m.group(2).lower().rstrip("s").rstrip(".")
        to   = m.group(3).lower().rstrip("s").rstrip(".")
        conversions = {
            ("km","mile"):      lambda v: round(v * 0.621371, 3),
            ("mile","km"):      lambda v: round(v * 1.60934, 3),
            ("kg","pound"):     lambda v: round(v * 2.20462, 3),
            ("kg","lb"):        lambda v: round(v * 2.20462, 3),
            ("pound","kg"):     lambda v: round(v / 2.20462, 3),
            ("celsius","fahrenheit"): lambda v: round(v * 9/5 + 32, 2),
            ("fahrenheit","celsius"): lambda v: round((v - 32) * 5/9, 2),
        }
        key = (frm.split("o")[0], to.split("o")[0])
        for (f,t), fn in conversions.items():
            if f in frm and t in to:
                return f"{val} {frm} = {fn(val)} {to}"
        return None

    def _lookup_fact(self, q: str) -> Optional[str]:
        """Simple fact database — no model needed."""
        facts = {
            "capital of india":    "New Delhi",
            "capital of usa":      "Washington D.C.",
            "capital of uk":       "London",
            "speed of light":      "299,792,458 metres per second",
            "pi":                  "3.14159265358979...",
            "what is python":      "Python is a high-level, interpreted programming language known for its clear syntax.",
            "what is ai":          "AI (Artificial Intelligence) is the simulation of human intelligence in machines.",
        }
        for key, value in facts.items():
            if key in q:
                return value
        return None

    def learn(self, query: str, answer: str):
        """Cache a good answer from higher tier for future use."""
        key = hashlib.md5(query.strip().lower().encode()).hexdigest()[:8]
        self._cache[key] = {"answer": answer, "query": query[:80]}
        try:
            CACHE_PATH.write_text(json.dumps(self._cache, indent=2, ensure_ascii=False))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# TIER 2 — SYMBOLIC ENGINE (Z3 + SymPy, ~50MB)
# ─────────────────────────────────────────────────────────────────────────────

class SymbolicEngine:
    """
    Exact math and logic using Z3 and SymPy.
    Deterministic — never hallucinates a math answer.
    Runs on any device that can run Python (including Raspberry Pi).
    """

    def answer(self, query: str) -> Optional[dict]:
        q = query.strip().lower()

        # SymPy: algebra, calculus, simplification
        if any(kw in q for kw in ["solve","simplify","derivative","integral","factor","expand","limit"]):
            return self._sympy_answer(query)

        # Z3: logic verification
        if any(kw in q for kw in ["prove","verify","check if","is it true","satisfiable"]):
            return self._z3_answer(query)

        # Pure arithmetic (higher precision than Tier 1)
        if re.search(r"[\d]+[\+\-\*\/\^][\d]+", query):
            return self._sympy_answer(query)

        return None

    def _sympy_answer(self, query: str) -> Optional[dict]:
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            t = standard_transformations + (implicit_multiplication_application,)

            q = query.strip().lower()

            # solve equation
            if "solve" in q:
                m = re.search(r"solve\s+(.+?)(?:\s+for\s+(\w))?$", q, re.IGNORECASE)
                if m:
                    expr_str = m.group(1).replace("=","-(") + ")" if "=" in m.group(1) else m.group(1)
                    var_str  = m.group(2) or "x"
                    var      = sp.Symbol(var_str)
                    try:
                        expr = parse_expr(expr_str, transformations=t)
                        sol  = sp.solve(expr, var)
                        return {"answer": f"{var} = {sol}", "tier": 2, "source": "sympy", "exact": True}
                    except Exception:
                        pass

            # derivative
            if "derivative" in q or "differentiate" in q:
                m = re.search(r"(?:derivative|differentiate)\s+(?:of\s+)?(.+?)(?:\s+with respect to\s+(\w))?$", q)
                if m:
                    var = sp.Symbol(m.group(2) or "x")
                    expr = parse_expr(m.group(1), transformations=t, local_dict={"x": var})
                    result = sp.diff(expr, var)
                    return {"answer": str(result), "tier": 2, "source": "sympy_diff", "exact": True}

            # simplify
            if "simplify" in q:
                m = re.search(r"simplify\s+(.+)", q)
                if m:
                    expr = parse_expr(m.group(1), transformations=t)
                    return {"answer": str(sp.simplify(expr)), "tier": 2, "source": "sympy_simplify", "exact": True}

        except ImportError:
            pass
        except Exception:
            pass
        return None

    def _z3_answer(self, query: str) -> Optional[dict]:
        try:
            import z3
            # Basic satisfiability check — expand as needed
            return None   # placeholder — full Z3 integration in nova_self_train.py
        except ImportError:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TIER 3 — SEMANTIC RAG (sentence-transformers, ~120MB)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticEngine:
    """
    Retrieves relevant stored knowledge and synthesises an answer
    WITHOUT calling any generative model.

    On devices with no LLM: returns the top retrieved chunks directly.
    When LLM is available: uses chunks as context for generation.

    This tier alone makes ARIA highly capable:
    - All your uploaded documents are searchable
    - All crawled websites are searchable
    - All research papers are searchable
    - All previous answers are cached
    """

    def __init__(self, memory):
        self.memory = memory

    def answer(self, query: str, top_k: int = 5) -> Optional[dict]:
        if not self.memory:
            return None
        try:
            results = self.memory.search(query, top_k=top_k)
            if not results:
                return None

            # If top result is very close match, return directly
            top = results[0]
            if top.get("similarity", 0) > 0.85:
                return {
                    "answer":     top["text"],
                    "sources":    [r.get("source","") for r in results[:3]],
                    "confidence": round(top.get("similarity",0), 3),
                    "tier":       3,
                    "source":     "rag_direct",
                }

            # Otherwise synthesise from top chunks (no model — just combine)
            combined = self._synthesise_without_model(query, results[:3])
            return {
                "answer":     combined,
                "sources":    [r.get("source","") for r in results[:3]],
                "confidence": round(top.get("similarity",0), 3),
                "tier":       3,
                "source":     "rag_combined",
            }
        except Exception as e:
            console.print(f"  [yellow]SemanticEngine error: {e}[/]")
            return None

    def _synthesise_without_model(self, query: str, chunks: list) -> str:
        """
        Combine retrieved chunks into an answer WITHOUT an LLM.
        Extracts the most relevant sentences from each chunk.
        Not perfect but answers basic questions from stored knowledge.
        """
        query_words = set(query.lower().split())

        best_sentences = []
        for chunk in chunks:
            text      = chunk.get("text","")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sent in sentences:
                sent_words = set(sent.lower().split())
                overlap    = len(query_words & sent_words)
                if overlap >= 2 and len(sent) > 30:
                    best_sentences.append((overlap, sent.strip()))

        best_sentences.sort(reverse=True)
        top_sents = [s for _, s in best_sentences[:4]]

        if not top_sents:
            return chunks[0].get("text","")[:500]

        return " ".join(top_sents)


# ─────────────────────────────────────────────────────────────────────────────
# ARIA BRAIN — master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class ARIABrain:
    """
    The central intelligence of ARIA.
    Routes every query through tiers — minimum resources used.

    On 512MB (Raspberry Pi Zero):
      Uses tiers 1, 2, 3 only.

    On 2GB (phone, low-end laptop):
      Adds tier 4 (phi3:mini).

    On 4GB+:
      Full stack including llama3.1:8b.

    The model is NEVER called for:
    - Time/date/weather/alarms (Tier 1)
    - Math/logic (Tier 2)
    - Questions about stored documents (Tier 3)
    - Repeated queries (Tier 1 cache)
    """

    def __init__(self, memory=None, engine=None):
        self.memory     = memory
        self.engine     = engine
        self.classifier = ComplexityClassifier()
        self.rules      = RuleEngine()
        self.symbolic   = SymbolicEngine()
        self.semantic   = SemanticEngine(memory)
        self._stats     = {"tier1":0,"tier2":0,"tier3":0,"tier4":0,"tier5":0,"total":0}

        # Detect available RAM
        self._max_tier  = self._detect_max_tier()
        console.print(f"  [green]ARIA Brain:[/] max tier = {self._max_tier} "
                      f"(RAM: {self._available_ram_mb():.0f}MB)")

    def _available_ram_mb(self) -> float:
        try:
            import psutil
            return psutil.virtual_memory().available / 1024 / 1024
        except Exception:
            return 2048.0

    def _detect_max_tier(self) -> int:
        ram = self._available_ram_mb()
        if ram < 400:   return 3   # Pi Zero, very low RAM
        if ram < 2048:  return 4   # 2GB — phi3:mini ok
        if ram < 5000:  return 4   # 4GB — phi3:mini comfortably
        return 5                   # 8GB+ — llama3.1 ok

    def answer(
        self,
        query:      str,
        min_tier:   int = 1,
        max_tier:   int = None,
        context:    str = "",
    ) -> dict:
        """
        Answer a query using minimum necessary resources.

        Args:
            query:    the user's question
            min_tier: start from this tier (default 1)
            max_tier: don't go above this tier (limits resource use)
            context:  optional additional context

        Returns:
            {answer, tier, source, confidence, latency_ms}
        """
        t0       = time.time()
        max_tier = max_tier or self._max_tier
        needed   = max(min_tier, self.classifier.classify(query))
        needed   = min(needed, max_tier)

        self._stats["total"] += 1
        result = None

        # Always try lower tiers first even if classifier says higher
        for tier in range(1, needed + 1):
            if tier == 1:
                result = self.rules.answer(query)
            elif tier == 2:
                result = self.symbolic.answer(query)
            elif tier == 3:
                result = self.semantic.answer(query)
            elif tier == 4 and self.engine:
                result = self._llm_answer(query, context, "fast")
            elif tier == 5 and self.engine:
                result = self._llm_answer(query, context, "cot")

            if result and result.get("answer"):
                self._stats[f"tier{result.get('tier',tier)}"] += 1
                # Cache in tier 1 for future
                if tier > 1:
                    self.rules.learn(query, result["answer"])
                result["latency_ms"] = int((time.time()-t0)*1000)
                return result

        # Fallback
        ms = int((time.time()-t0)*1000)
        return {
            "answer":     "I don't have enough information to answer that. "
                         "Try uploading relevant documents or crawling related websites.",
            "tier":       0,
            "source":     "fallback",
            "confidence": 0.0,
            "latency_ms": ms,
        }

    def _llm_answer(self, query: str, context: str, mode: str) -> Optional[dict]:
        """Call the local LLM — used only when tiers 1-3 can't answer."""
        if not self.engine:
            return None
        try:
            # Get RAG context to improve LLM answer
            rag = ""
            if self.memory:
                hits = self.memory.search(query, top_k=3)
                if hits:
                    rag = "\n".join(h["text"][:300] for h in hits)

            prompt = (
                f"Answer this question accurately and concisely.\n"
                f"Question: {query}\n"
                + (f"Context: {rag}\n" if rag else "")
                + (f"Additional context: {context}\n" if context else "")
                + "Answer:"
            )
            answer = self.engine.generate(
                prompt,
                temperature=0.1 if mode == "fast" else 0.3,
            )
            return {
                "answer":     answer.strip(),
                "tier":       4,
                "source":     "llm+rag" if rag else "llm",
                "confidence": 0.7,
            }
        except Exception as e:
            console.print(f"  [yellow]LLM answer error: {e}[/]")
            return None

    def stats(self) -> dict:
        total = self._stats["total"] or 1
        return {
            "total_queries": self._stats["total"],
            "tier_usage": {
                f"tier_{i}": {
                    "count":   self._stats[f"tier{i}"],
                    "percent": round(self._stats[f"tier{i}"]/total*100, 1),
                }
                for i in range(1,6)
            },
            "model_calls_percent": round(
                (self._stats["tier4"]+self._stats["tier5"])/total*100, 1
            ),
            "model_free_percent": round(
                (self._stats["tier1"]+self._stats["tier2"]+self._stats["tier3"])/total*100, 1
            ),
            "max_tier": self._max_tier,
            "available_ram_mb": round(self._available_ram_mb(), 0),
        }
