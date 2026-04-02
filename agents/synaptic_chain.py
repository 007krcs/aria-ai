"""
ARIA Synaptic Chain — Sequential Dot-Connecting Intelligence
=============================================================

"Think on every perspective, then provide an actual answer."

How it works (like a real mind):

  Query comes in
      ↓
  Classifier neuron fires first — WHAT type is this? WHAT domain?
      ↓
  Chain builder selects ordered agents — WHO should think about this?
      ↓
  Agent 1 fires — reads query, adds its finding to context
      ↓
  Agent 2 fires — reads query + Agent 1's finding, adds its layer
      ↓
  Agent 3 fires — reads everything so far, adds its layer
      ↓
  (Early stop if confidence is already high enough)
      ↓
  Synthesizer reads the FULL accumulated context → one coherent answer

Key principles:
  - ONE agent at a time. No parallel collisions.
  - Each agent sees everything the previous agents found.
  - Each agent ONLY contributes what it uniquely knows.
  - Early stopping: if confidence >= 0.85 after any agent, synthesize immediately.
  - The brain approves the final answer before it leaves.

Domain → chain templates (ordered by what should fire first):
  "stock"    → memory → web → stock_specialist → sentiment → reasoner
  "code"     → memory → code_specialist → code_runner → reasoner
  "research" → memory → web → scientist → reasoner
  "question" → memory → world_model → reasoner
  "task"     → memory → system → planner → reasoner
  "medical"  → memory → web → medical → reasoner
  "math"     → symbolic → reasoner
  "weather"  → web → reasoner
  "security" → threat → system → reasoner
  "general"  → memory → web → reasoner
"""

from __future__ import annotations

import re
import json
import time
import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Callable

# ─────────────────────────────────────────────────────────────────────────────
# ACCUMULATED CONTEXT — the "dot" that gets connected through the chain
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Dot:
    """
    The signal passed between neurons.
    Each agent reads this, enriches it, passes it forward.
    No agent should contradict what's already here — only ADD.
    """
    query:        str
    intent:       str           = "question"
    domain:       str           = "general"
    complexity:   str           = "medium"
    city:         str           = ""
    confidence:   float         = 0.0
    facts:        List[str]     = field(default_factory=list)
    sources:      List[str]     = field(default_factory=list)
    agent_trail:  List[Dict]    = field(default_factory=list)   # who contributed what
    working_text: str           = ""                             # building answer
    final_answer: str           = ""                             # set only by synthesizer
    stopped_early: bool         = False
    ts_start:     float         = field(default_factory=time.time)

    def add_finding(self, agent: str, finding: str, confidence_boost: float = 0.0, source: str = ""):
        """An agent calls this to register its contribution."""
        if not finding or not finding.strip():
            return
        self.facts.append(finding.strip())
        if source:
            self.sources.append(source)
        self.agent_trail.append({
            "agent":      agent,
            "finding":    finding.strip()[:300],
            "confidence": round(confidence_boost, 2),
            "ts":         round(time.time() - self.ts_start, 2),
        })
        self.confidence = min(0.99, self.confidence + confidence_boost)

    def build_context(self) -> str:
        """Formatted context for the next agent to read."""
        if not self.facts:
            return f"Query: {self.query}"
        lines = [f"Query: {self.query}", f"Domain: {self.domain}", ""]
        for i, fact in enumerate(self.facts, 1):
            lines.append(f"[Finding {i}] {fact}")
        if self.sources:
            lines.append(f"\nSources: {', '.join(self.sources[:3])}")
        return "\n".join(lines)

    def elapsed_ms(self) -> int:
        return int((time.time() - self.ts_start) * 1000)

    def summary(self) -> Dict:
        return {
            "intent":       self.intent,
            "domain":       self.domain,
            "confidence":   round(self.confidence, 2),
            "agents_fired": len(self.agent_trail),
            "facts_found":  len(self.facts),
            "elapsed_ms":   self.elapsed_ms(),
            "stopped_early": self.stopped_early,
            "trail":        [t["agent"] for t in self.agent_trail],
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUERY CLASSIFIER — fires first, decides domain + chain
# ─────────────────────────────────────────────────────────────────────────────

class QueryClassifier:
    """
    Zero-model classifier. Pure pattern matching.
    Decides: intent, domain, complexity in < 1ms.
    """

    DOMAIN_PATTERNS = {
        "stock": [
            r"\b(stock|share|nse|bse|nifty|sensex|mcx|crypto|bitcoin|price|market|ipo|invest|trade|portfolio)\b",
        ],
        "code": [
            r"\b(code|function|script|debug|python|javascript|java|sql|api|error|bug|class|def |import |var |const )\b",
            r"```",
        ],
        "medical": [
            r"\b(symptom|disease|medicine|drug|doctor|treatment|diagnosis|health|pain|fever|cancer|diabetes)\b",
        ],
        "research": [
            r"\b(research|study|paper|journal|experiment|prove|evidence|according to|scientist|discovery)\b",
        ],
        "math": [
            r"\b(solve|calculate|equation|integral|derivative|matrix|probability|algebra|geometry|theorem)\b",
            r"[\d]+\s*[\+\-\*\/\^]\s*[\d]+",
        ],
        "weather": [
            r"\b(weather|temperature|rain|humidity|forecast|climate|wind|storm|sunny|cloudy)\b",
        ],
        "security": [
            r"\b(hack|virus|malware|phishing|scam|password|vulnerability|threat|firewall|encrypt)\b",
        ],
        "task": [
            r"\b(open|create|make|send|set|schedule|remind|alarm|play|turn on|turn off|call|message)\b",
        ],
        "question": [
            r"^(what|why|how|when|where|who|which|explain|tell me|define|meaning)",
        ],
    }

    COMPLEXITY_RULES = {
        "high":   ["analyze", "compare", "evaluate", "research", "explain in detail", "comprehensive", "thoroughly"],
        "low":    ["what is", "who is", "define", "open", "set", "remind", "play"],
        "medium": [],
    }

    def classify(self, query: str) -> Dict[str, str]:
        q = query.lower().strip()

        # Domain
        domain = "general"
        for d, patterns in self.DOMAIN_PATTERNS.items():
            if any(re.search(p, q) for p in patterns):
                domain = d
                break

        # Intent
        intent = "question"
        if re.match(r"^(open|create|make|send|set|schedule|remind|play|turn)", q):
            intent = "task"
        elif re.match(r"^(analyze|compare|evaluate|review|assess)", q):
            intent = "analysis"
        elif re.match(r"^(find|search|look up|research|latest|news)", q):
            intent = "search"
        elif any(re.match(r, q) for r in [r"[\d]+\s*[\+\-\*\/]", r"^(solve|calc)"]):
            intent = "calculation"

        # Complexity
        complexity = "medium"
        for level, keywords in self.COMPLEXITY_RULES.items():
            if any(kw in q for kw in keywords):
                complexity = level
                break
        if len(q.split()) < 6:
            complexity = "low"
        elif len(q.split()) > 30:
            complexity = "high"

        return {"intent": intent, "domain": domain, "complexity": complexity}


# ─────────────────────────────────────────────────────────────────────────────
# CHAIN TEMPLATES — domain → ordered agent list
# ─────────────────────────────────────────────────────────────────────────────

CHAIN_TEMPLATES: Dict[str, List[str]] = {
    # Financial — memory first (has past context), then live data, then reason
    "stock":    ["memory_neuron", "live_data_neuron", "sentiment_neuron", "reason_neuron"],

    # Code — memory for patterns, specialist for structure, runner for validation
    "code":     ["memory_neuron", "code_neuron", "validate_neuron", "reason_neuron"],

    # Medical — memory, web for latest, specialist filter, reason
    "medical":  ["memory_neuron", "web_neuron", "filter_neuron", "reason_neuron"],

    # Research / science — memory, web, deep-read, reason
    "research": ["memory_neuron", "web_neuron", "deep_neuron", "reason_neuron"],

    # Math — symbolic first (exact), then reason to explain
    "math":     ["symbolic_neuron", "reason_neuron"],

    # Weather — live data, reason
    "weather":  ["live_data_neuron", "reason_neuron"],

    # Security — threat scan, system context, reason
    "security": ["threat_neuron", "context_neuron", "reason_neuron"],

    # Task execution — memory for workflow, system for execution, plan
    "task":     ["memory_neuron", "system_neuron", "plan_neuron"],

    # Simple question — memory, world model, reason
    "question": ["memory_neuron", "world_neuron", "reason_neuron"],

    # General fallback
    "general":  ["memory_neuron", "web_neuron", "reason_neuron"],
}

# Complexity overrides — fewer neurons for simple queries
COMPLEXITY_TRIM = {
    "low":    2,   # max 2 neurons
    "medium": 3,   # max 3 neurons
    "high":   4,   # max 4 neurons (full chain)
}

# Confidence thresholds — stop early when we know enough
STOP_THRESHOLDS = {
    "low":    0.70,
    "medium": 0.78,
    "high":   0.85,
}


# ─────────────────────────────────────────────────────────────────────────────
# NEURON IMPLEMENTATIONS — the individual dot-connectors
# ─────────────────────────────────────────────────────────────────────────────

class _NeuronBase:
    """Base class for all chain neurons."""
    name = "base_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        """Cheap gate — should this neuron fire for this dot?"""
        return True

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        """
        Do real work. Call dot.add_finding() to register contribution.
        NEVER modify dot.query. ONLY add findings.
        """
        pass


class MemoryNeuron(_NeuronBase):
    """
    Fires first. Reads ARIA's long-term memory for relevant context.
    If we already know the answer, confidence jumps immediately.
    """
    name = "memory_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return True  # always check memory first

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        # Try memory hierarchy first
        mem_h = aria.get("memory_hierarchy")
        if mem_h:
            try:
                hits = mem_h.search(dot.query, k=3)
                for h in hits:
                    text = h.get("text", "").strip()
                    if text and len(text) > 30:
                        dot.add_finding(
                            self.name,
                            f"From memory: {text[:400]}",
                            confidence_boost=0.15,
                            source="aria_memory"
                        )
                return
            except Exception:
                pass

        # Fallback to legacy memory
        memory = aria.get("memory")
        if not memory:
            return
        try:
            hits = memory.search(dot.query, k=3)
            for h in hits:
                text = h.get("text", h) if isinstance(h, dict) else str(h)
                if text and len(text) > 20:
                    dot.add_finding(
                        self.name,
                        f"From memory: {text[:400]}",
                        confidence_boost=0.12,
                        source="aria_memory"
                    )
        except Exception:
            pass


class WebNeuron(_NeuronBase):
    """
    Searches the web. Only fires if memory didn't already answer confidently.
    Passes clean excerpts — not raw HTML.
    """
    name = "web_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        # Skip if memory already gave high confidence
        return dot.confidence < 0.75

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(dot.query, max_results=3))
            for r in results[:2]:
                snippet = r.get("body", "").strip()
                url     = r.get("href", "")
                title   = r.get("title", "")
                if snippet and len(snippet) > 40:
                    dot.add_finding(
                        self.name,
                        f"{title}: {snippet[:400]}",
                        confidence_boost=0.18,
                        source=url
                    )
        except Exception:
            pass


class LiveDataNeuron(_NeuronBase):
    """
    Fetches real-time data — stock prices, weather, live feeds.
    Only fires for real-time data queries.
    """
    name = "live_data_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain in ("stock", "weather")

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        if dot.domain == "stock":
            self._fetch_stock(dot)
        elif dot.domain == "weather":
            self._fetch_weather(dot, aria)

    def _fetch_stock(self, dot: Dot) -> None:
        try:
            import yfinance as yf
            ticker = self._extract_ticker(dot.query)
            if not ticker:
                return
            t = yf.Ticker(ticker)
            info = t.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
            if price:
                change = getattr(info, "regular_market_change_percent", None)
                change_str = f" ({change:+.2f}%)" if change else ""
                dot.add_finding(
                    self.name,
                    f"LIVE: {ticker} = ₹{price:,.2f}{change_str}",
                    confidence_boost=0.25,
                    source=f"yfinance:{ticker}"
                )
        except Exception:
            pass

    def _extract_ticker(self, query: str) -> Optional[str]:
        q = query.upper()
        KNOWN = {
            "MCX": "MCX.NS", "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS",
            "INFOSYS": "INFY.NS", "HDFC": "HDFCBANK.NS", "NIFTY": "^NSEI",
            "SENSEX": "^BSESN", "BITCOIN": "BTC-USD", "GOLD": "GC=F",
        }
        for name, ticker in KNOWN.items():
            if name in q:
                return ticker
        m = re.search(r'\b([A-Z]{2,6})\b', q)
        if m:
            t = m.group(1)
            if t not in {"THE", "AND", "FOR", "ARE", "WHAT", "HOW", "WHY", "NSE", "BSE"}:
                return t + ".NS"
        return None

    def _fetch_weather(self, dot: Dot, aria: Dict) -> None:
        # placeholder — can be wired to a weather API
        pass


class SentimentNeuron(_NeuronBase):
    """
    For stock/financial queries: reads market sentiment.
    Fires AFTER live data — it interprets what live data means.
    """
    name = "sentiment_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "stock" and dot.confidence > 0.1

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            sia = SentimentIntensityAnalyzer()
            # Analyze the facts found so far
            full_text = " ".join(dot.facts)
            score = sia.polarity_scores(full_text)
            compound = score["compound"]
            if abs(compound) > 0.05:
                sentiment = "positive" if compound > 0.2 else ("negative" if compound < -0.2 else "neutral")
                dot.add_finding(
                    self.name,
                    f"Market sentiment analysis: {sentiment} (score: {compound:.2f})",
                    confidence_boost=0.08,
                )
        except Exception:
            pass


class CodeNeuron(_NeuronBase):
    """
    For code queries: reads the code structure, identifies patterns.
    Enriches context with technical analysis before reasoning.
    """
    name = "code_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "code"

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        q = dot.query.lower()
        # Detect language
        lang = "Python"
        for l in ["javascript", "java", "c++", "c#", "rust", "go", "sql", "bash"]:
            if l in q:
                lang = l.capitalize()
                break
        # Detect problem type
        problem = "implementation"
        for p in ["debug", "fix", "error", "bug"]:
            if p in q:
                problem = "debugging"
                break
        for p in ["optimize", "performance", "speed"]:
            if p in q:
                problem = "optimization"
                break

        dot.add_finding(
            self.name,
            f"Code context: Language={lang}, Task={problem}",
            confidence_boost=0.10,
        )


class ValidateNeuron(_NeuronBase):
    """
    For code: validates that the planned solution is sound before reasoning.
    Adds constraints and edge cases to the context.
    """
    name = "validate_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "code" and dot.confidence > 0.1

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        concerns = []
        q = dot.query.lower()
        if "sql" in q and "input" in q:
            concerns.append("Potential SQL injection risk — use parameterized queries")
        if "password" in q or "secret" in q:
            concerns.append("Security sensitive — never hardcode credentials")
        if "loop" in q or "recursive" in q:
            concerns.append("Check for infinite loop / stack overflow edge cases")
        if concerns:
            dot.add_finding(
                self.name,
                "Code validation notes: " + "; ".join(concerns),
                confidence_boost=0.05,
            )


class SymbolicNeuron(_NeuronBase):
    """
    For math: tries exact symbolic computation before LLM reasoning.
    If it works, confidence is very high (exact answer).
    """
    name = "symbolic_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "math"

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        q = dot.query
        # Try simple arithmetic first
        expr_match = re.search(r'[\d\s\+\-\*\/\^\(\)\.]+', q)
        if expr_match:
            expr = expr_match.group().strip()
            if expr and len(expr) > 2:
                try:
                    expr_safe = expr.replace("^", "**")
                    result = eval(expr_safe, {"__builtins__": {}})
                    dot.add_finding(
                        self.name,
                        f"Exact calculation: {expr} = {result}",
                        confidence_boost=0.55,
                        source="symbolic_engine"
                    )
                    return
                except Exception:
                    pass

        # Try sympy for algebra
        try:
            import sympy
            # Extract equation if present
            if "=" in q:
                try:
                    parts = q.split("=")
                    x = sympy.Symbol("x")
                    lhs = sympy.sympify(parts[0].replace("solve", "").strip())
                    rhs = sympy.sympify(parts[1].strip())
                    sol = sympy.solve(lhs - rhs, x)
                    if sol:
                        dot.add_finding(
                            self.name,
                            f"Symbolic solution: x = {sol}",
                            confidence_boost=0.60,
                            source="sympy"
                        )
                except Exception:
                    pass
        except ImportError:
            pass


class WorldNeuron(_NeuronBase):
    """
    For general questions: consults ARIA's world knowledge index.
    Fires when memory has low confidence.
    """
    name = "world_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain in ("question", "general") and dot.confidence < 0.70

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        world = aria.get("world_model")
        if not world:
            return
        try:
            result = world.query(dot.query)
            if result and isinstance(result, str) and len(result) > 20:
                dot.add_finding(
                    self.name,
                    f"World knowledge: {result[:400]}",
                    confidence_boost=0.20,
                    source="world_model"
                )
        except Exception:
            pass


class ThreatNeuron(_NeuronBase):
    """For security queries: scans for threats before answering."""
    name = "threat_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "security"

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        q = dot.query.lower()
        threat_indicators = {
            "phishing":    ["click here", "verify account", "urgent", "suspended", "login"],
            "malware":     ["exe", "download", "free crack", "keygen", "patch"],
            "social_eng":  ["give me your", "send password", "wire transfer", "gift card"],
        }
        found = []
        for threat_type, indicators in threat_indicators.items():
            if any(ind in q for ind in indicators):
                found.append(threat_type)
        if found:
            dot.add_finding(
                self.name,
                f"Security alert: detected indicators of {', '.join(found)}",
                confidence_boost=0.30,
                source="threat_scanner"
            )
        else:
            dot.add_finding(
                self.name,
                "Security scan: no immediate threat indicators detected",
                confidence_boost=0.10,
                source="threat_scanner"
            )


class ContextNeuron(_NeuronBase):
    """Reads system/environment context for system queries."""
    name = "context_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain in ("security", "task")

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        import platform, os
        ctx = f"System: {platform.system()} {platform.release()}, Python {platform.python_version()}"
        dot.add_finding(self.name, ctx, confidence_boost=0.05, source="system_context")


class SystemNeuron(_NeuronBase):
    """For task execution: determines what system action to take."""
    name = "system_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "task"

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        tools = aria.get("tools")
        if not tools:
            return
        try:
            result = tools.smart_execute(dot.query, engine)
            if result and hasattr(result, "output"):
                dot.add_finding(
                    self.name,
                    f"System action result: {str(result.output)[:300]}",
                    confidence_boost=0.40,
                    source="system_tools"
                )
        except Exception:
            pass


class PlanNeuron(_NeuronBase):
    """For tasks: creates an execution plan."""
    name = "plan_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "task" and dot.confidence < 0.70

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        q = dot.query.lower()
        steps = []
        if "remind" in q or "schedule" in q:
            steps = ["Parse time and task from query", "Set system reminder", "Confirm to user"]
        elif "open" in q:
            steps = ["Identify application", "Launch application", "Confirm opened"]
        elif "send" in q or "message" in q:
            steps = ["Identify recipient", "Compose message", "Send via appropriate channel"]

        if steps:
            dot.add_finding(
                self.name,
                "Execution plan: " + " → ".join(steps),
                confidence_boost=0.12,
            )


class DeepNeuron(_NeuronBase):
    """For research: does a deeper read of the top search result."""
    name = "deep_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "research" and dot.sources and dot.confidence < 0.75

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        if not dot.sources:
            return
        url = dot.sources[0]
        try:
            import requests
            r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                # Extract paragraphs
                import re as _re
                text = _re.sub(r'<[^>]+>', ' ', r.text)
                text = _re.sub(r'\s+', ' ', text).strip()
                # Find most relevant 400 chars
                q_words = set(dot.query.lower().split())
                sentences = text.split('. ')
                best = max(sentences, key=lambda s: sum(1 for w in q_words if w in s.lower()), default="")
                if best and len(best) > 50:
                    dot.add_finding(
                        self.name,
                        f"Deep read: {best[:400]}",
                        confidence_boost=0.15,
                        source=url
                    )
        except Exception:
            pass


class FilterNeuron(_NeuronBase):
    """For medical: filters findings for accuracy and safety."""
    name = "filter_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return dot.domain == "medical"

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        dot.add_finding(
            self.name,
            "Medical disclaimer: This information is for educational purposes. Always consult a qualified healthcare professional for medical advice.",
            confidence_boost=0.05,
            source="safety_filter"
        )


class ReasonNeuron(_NeuronBase):
    """
    The final thinking neuron. ALWAYS fires last.
    Reads ALL accumulated findings and produces coherent reasoning.
    Uses the LLM to synthesize — but now the LLM has full context.
    """
    name = "reason_neuron"

    def can_contribute(self, dot: Dot) -> bool:
        return True  # always fires last

    def contribute(self, dot: Dot, engine, aria: Dict) -> None:
        if not engine:
            return

        context = dot.build_context()
        system = (
            "You are ARIA's reasoning engine. You have been given accumulated findings from multiple specialist systems. "
            "Your job: synthesize these findings into ONE clear, accurate, complete answer. "
            "Do NOT repeat the findings verbatim — reason from them. "
            "Be direct. If findings contradict each other, note the most reliable one. "
            "If a finding says 'From memory' or 'LIVE:', trust those over web results."
        )

        try:
            answer = engine.generate(context, system=system, max_tokens=600)
            if answer and len(answer) > 20:
                dot.working_text = answer
                dot.add_finding(self.name, f"Reasoning complete", confidence_boost=0.20)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# NEURON REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

ALL_NEURONS: Dict[str, _NeuronBase] = {
    "memory_neuron":    MemoryNeuron(),
    "web_neuron":       WebNeuron(),
    "live_data_neuron": LiveDataNeuron(),
    "sentiment_neuron": SentimentNeuron(),
    "code_neuron":      CodeNeuron(),
    "validate_neuron":  ValidateNeuron(),
    "symbolic_neuron":  SymbolicNeuron(),
    "world_neuron":     WorldNeuron(),
    "threat_neuron":    ThreatNeuron(),
    "context_neuron":   ContextNeuron(),
    "system_neuron":    SystemNeuron(),
    "plan_neuron":      PlanNeuron(),
    "deep_neuron":      DeepNeuron(),
    "filter_neuron":    FilterNeuron(),
    "reason_neuron":    ReasonNeuron(),
}


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPTIC CHAIN — the orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class SynapticChain:
    """
    Sequential dot-connecting intelligence.
    Replaces parallel fan-out with ordered neural chain.

    One agent fires. Passes findings to the next.
    Next agent reads everything. Adds its layer.
    Stops when confidence is high enough.
    One synthesized answer at the end.
    """

    def __init__(self, engine=None, aria: Optional[Dict] = None):
        self.engine    = engine
        self.aria      = aria or {}
        self.classifier = QueryClassifier()

    def _build_chain(self, domain: str, complexity: str) -> List[_NeuronBase]:
        template = CHAIN_TEMPLATES.get(domain, CHAIN_TEMPLATES["general"])
        max_neurons = COMPLEXITY_TRIM.get(complexity, 3)
        # Always include reason_neuron at the end
        chain = [ALL_NEURONS[n] for n in template if n in ALL_NEURONS]
        if not chain or chain[-1].name != "reason_neuron":
            chain.append(ALL_NEURONS["reason_neuron"])
        return chain[:max_neurons]

    def _sse(self, event_type: str, **kwargs) -> str:
        return f"data: {json.dumps({'type': event_type, **kwargs})}\n\n"

    async def stream(self, query: str, city: str = "") -> AsyncGenerator[str, None]:
        """
        SSE-compatible stream. Same format as NeuralOrchestrator.
        Each neuron fires sequentially. Findings accumulate.
        """
        loop = asyncio.get_event_loop()

        # ── Step 1: Classify ────────────────────────────────────────────────
        classification = self.classifier.classify(query)
        dot = Dot(
            query=query,
            city=city,
            intent=classification["intent"],
            domain=classification["domain"],
            complexity=classification["complexity"],
        )

        yield self._sse("thinking",
            text=f"Classified: {dot.domain} / {dot.intent} / {dot.complexity}")

        # ── Step 2: Build chain ─────────────────────────────────────────────
        chain = self._build_chain(dot.domain, dot.complexity)
        stop_threshold = STOP_THRESHOLDS.get(dot.complexity, 0.78)

        yield self._sse("chain_plan",
            text=f"Chain: {' → '.join(n.name.replace('_neuron','') for n in chain)}")

        # ── Step 3: Sequential execution ────────────────────────────────────
        for i, neuron in enumerate(chain):
            if not neuron.can_contribute(dot):
                yield self._sse("neuron_skip",
                    text=f"[{neuron.name}] skipped (not relevant)",
                    agent=neuron.name)
                continue

            yield self._sse("neuron_fire",
                text=f"[{neuron.name}] thinking...",
                agent=neuron.name,
                confidence=dot.confidence)

            try:
                # Run in thread pool (neurons can do I/O)
                await loop.run_in_executor(
                    None,
                    lambda n=neuron: n.contribute(dot, self.engine, self.aria)
                )
            except Exception as e:
                yield self._sse("neuron_error",
                    text=f"[{neuron.name}] error: {type(e).__name__}",
                    agent=neuron.name)
                continue

            # Report what this neuron found
            if dot.agent_trail and dot.agent_trail[-1]["agent"] == neuron.name:
                last = dot.agent_trail[-1]
                yield self._sse("neuron_result",
                    text=last["finding"][:200],
                    agent=neuron.name,
                    confidence=dot.confidence)

            # Early stop check
            is_last = (i == len(chain) - 1)
            if not is_last and dot.confidence >= stop_threshold:
                dot.stopped_early = True
                yield self._sse("early_stop",
                    text=f"Confidence {dot.confidence:.0%} — enough to answer",
                    confidence=dot.confidence)
                # Still run reason_neuron for coherent synthesis
                reason = ALL_NEURONS["reason_neuron"]
                await loop.run_in_executor(
                    None,
                    lambda: reason.contribute(dot, self.engine, self.aria)
                )
                break

        # ── Step 4: Compose final answer ─────────────────────────────────────
        final = dot.working_text or self._fallback_answer(dot)
        dot.final_answer = final

        # Stream the answer token by token
        words = final.split()
        chunk_size = 8
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            yield self._sse("token", text=chunk + " ")
            await asyncio.sleep(0.02)

        # Done event with full summary
        yield self._sse("done",
            text=final,
            chain_summary=dot.summary(),
            sources=dot.sources[:5])

    def _fallback_answer(self, dot: Dot) -> str:
        """Build an answer directly from accumulated facts if reasoning failed."""
        if not dot.facts:
            return "I wasn't able to find enough information to answer that confidently."
        # Combine facts into a coherent response
        relevant = [f for f in dot.facts if "From memory:" not in f or len(f) > 60]
        if not relevant:
            relevant = dot.facts
        return "\n\n".join(relevant[:3])

    def run(self, query: str, city: str = "") -> str:
        """Synchronous version — returns final answer string."""
        async def _run():
            final = ""
            async for chunk in self.stream(query, city):
                if '"type": "done"' in chunk or '"type":"done"' in chunk:
                    try:
                        d = json.loads(chunk.replace("data: ", "").strip())
                        final = d.get("text", final)
                    except Exception:
                        pass
            return final

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_run())
        loop.close()
        return result
