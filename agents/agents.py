"""
ARIA — Agent definitions
Four agents. Each has one job. They communicate through the shared memory bus.

Researcher  → finds relevant knowledge (RAG + web)
Reasoner    → thinks step-by-step through the problem
Critic      → scores the answer and decides if it's good enough
Meta-Agent  → routes, orchestrates, and decides when to adapt

All agents share the same Engine and Memory. They talk through structured dicts.
"""

import re
import time
from typing import Optional
from rich.console import Console

from core.engine import Engine
from core.memory import Memory
from tools.logger import Logger

console = Console()


# ── Base Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """All agents inherit from this."""

    name = "base"

    def __init__(self, engine: Engine, memory: Memory, logger: Logger):
        self.engine = engine
        self.memory = memory
        self.logger = logger

    def _timed_run(self, fn, *args, **kwargs):
        """Run a function and return (result, latency_ms)."""
        t0     = time.time()
        result = fn(*args, **kwargs)
        ms     = int((time.time() - t0) * 1000)
        return result, ms


# ── Researcher Agent ──────────────────────────────────────────────────────────

class ResearcherAgent(BaseAgent):
    """
    Finds relevant knowledge before the engine answers.
    First checks memory (ChromaDB). If insufficient, falls back to web search.
    """

    name = "researcher"

    SYSTEM = """You are a precise research assistant.
Your job: extract the key facts from the provided context that are most relevant to the question.
Be concise. Cite which source each fact comes from. Do not add facts not in the context."""

    def run(self, query: str, domain: Optional[str] = None) -> dict:
        """
        Returns:
            {
              "context":    str,   # formatted knowledge for the Reasoner
              "sources":    list,  # where knowledge came from
              "from_memory": bool, # True = used ChromaDB, False = no memory found
              "web_used":   bool,  # True = also fetched from web
            }
        """
        t0 = time.time()

        # Step 1: Search memory
        context, found = self.memory.build_context(query, domain=domain)
        sources        = []
        web_used       = False

        if found:
            hits    = self.memory.search(query, domain=domain)
            sources = list({h["source"] for h in hits})
            console.print(f"  [dim]Researcher:[/] found {len(hits)} memory chunks")
            context = context[:2000]  # cap context size for speed
        else:
            # Step 2: Try free web search (DuckDuckGo — no API key)
            console.print(f"  [dim]Researcher:[/] memory miss → web search")
            web_context = self._web_search(query)
            if web_context:
                context  = web_context[:1500]  # cap web context — smaller = faster model
                web_used = True
                sources  = ["web_search"]

        ms = int((time.time() - t0) * 1000)
        self.logger.log_agent_run(
            agent_name=self.name, task=query,
            result=context[:200], score=0.7 if found else 0.4,
            latency_ms=ms
        )
        return {
            "context":     context,
            "sources":     sources,
            "from_memory": found,
            "web_used":    web_used,
        }

    def _web_search(self, query: str) -> str:
        """
        Fast web search — 3 free backends, tries each in order, 10s max total.
        Backend 1: duckduckgo_search library (~1-2s)
        Backend 2: DuckDuckGo instant answer API (~1s)
        Backend 3: Wikipedia API (~1s)
        """
        console.print(f"  [dim]Web search:[/] {query[:60]}")

        # Backend 1: duckduckgo_search library
        try:
            from ddgs import DDGS
            with DDGS(timeout=8) as ddgs:
                results = list(ddgs.text(query, max_results=5, timelimit="m"))
            if results:
                lines = [f"Web search results for: {query}\n"]
                for r in results[:4]:
                    body = r.get("body", "")
                    # Keep only first 150 chars of each result — enough context, much faster
                    short_body = body[:150] + "..." if len(body) > 150 else body
                    lines.append(f"- {r.get('title','')}: {short_body}")
                console.print(f"  [dim]Web:[/] {len(results)} results found")
                return "\n".join(lines)
        except ImportError:
            pass
        except Exception as e:
            console.print(f"  [dim]DDG search error: {e}[/]")

        # Backend 2: DuckDuckGo instant answer API
        try:
            import requests as req
            r = req.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                timeout=6,
                headers={"User-Agent": "Mozilla/5.0 ARIA/1.0"},
            )
            data = r.json()
            parts = []
            if data.get("AbstractText"):
                parts.append(f"Summary: {data['AbstractText']}")
            if data.get("Answer"):
                parts.append(f"Answer: {data['Answer']}")
            for topic in data.get("RelatedTopics", [])[:4]:
                if isinstance(topic, dict) and topic.get("Text"):
                    parts.append(f"- {topic['Text']}")
            if parts:
                console.print(f"  [dim]Web:[/] instant answer found")
                return f"Web results for: {query}\n" + "\n".join(parts)
        except Exception as e:
            console.print(f"  [dim]DDG instant error: {e}[/]")

        # Backend 3: Wikipedia API
        try:
            import requests as req
            r = req.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search",
                    "srsearch": query, "format": "json",
                    "srlimit": 3, "srprop": "snippet",
                },
                timeout=6,
            )
            items = r.json().get("query", {}).get("search", [])
            if items:
                lines = [f"Wikipedia results for: {query}\n"]
                for item in items:
                    snippet = item["snippet"].replace('<span class="searchmatch">', '').replace("</span>", "")
                    lines.append(f"- {item['title']}: {snippet}")
                console.print(f"  [dim]Web:[/] Wikipedia {len(items)} results")
                return "\n".join(lines)
        except Exception as e:
            console.print(f"  [dim]Wikipedia error: {e}[/]")

        console.print("  [yellow]All web backends failed — using model knowledge[/]")
        return ""


# ── Reasoner Agent ────────────────────────────────────────────────────────────

class ReasonerAgent(BaseAgent):
    """
    The main reasoning layer. Supports three modes:

    FAST   — direct answer (best for simple factual questions)
    COT    — Chain of Thought (step-by-step reasoning, better quality)
    VERIFY — CoT + self-verification pass (highest quality, slowest)

    Mode is selected automatically based on question complexity,
    or can be forced by passing mode= argument.
    """

    name = "reasoner"

    # ── System prompts per mode ──────────────────────────────────────────────

    SYSTEM_FAST = (
        "You are a helpful assistant. Answer clearly and concisely. "
        "Use the provided context. Do not make up facts."
    )

    SYSTEM_COT = (
        "You are a careful reasoning assistant. "
        "Before answering, think through the problem step by step. "
        "Use the provided context as your primary source. "
        "Do not make up facts. If unsure, say so clearly."
    )

    SYSTEM_VERIFY = (
        "You are a rigorous reasoning assistant. "
        "Think step by step. After giving an answer, verify it by checking "
        "your reasoning against the context for errors or gaps. "
        "Correct yourself if needed before giving the final answer."
    )

    def run(
        self,
        query: str,
        context: str = "",
        language: str = "English",
        mode: str = "auto",
    ) -> dict:
        """
        Reason through a query.

        mode options:
            "auto"   — picks mode based on question complexity (default)
            "fast"   — direct answer, no reasoning steps
            "cot"    — Chain of Thought step-by-step
            "verify" — CoT + self-verification

        Returns:
            { answer, reasoning, steps, confidence, mode_used }
        """
        t0 = time.time()

        # Auto-select mode based on question complexity
        if mode == "auto":
            mode = self._detect_mode(query)

        console.print(f"  [dim]Reasoner ({mode})...[/]")

        if mode == "fast":
            result = self._fast(query, context, language)
        elif mode == "verify":
            result = self._verify(query, context, language)
        else:
            result = self._cot(query, context, language)

        ms = int((time.time() - t0) * 1000)
        result["mode_used"] = mode
        result["latency_ms"] = ms

        self.logger.log_agent_run(
            agent_name=self.name, task=query,
            result=result["answer"][:200], latency_ms=ms
        )
        return result

    # ── Mode detection ────────────────────────────────────────────────────────

    def _detect_mode(self, query: str) -> str:
        """
        Pick reasoning mode based on question signals.

        fast   → simple factual lookups (what is X, who is X, define X)
        cot    → multi-step questions (how, why, explain, compare, analyse)
        verify → high-stakes questions (should I, is it true that, prove)
        """
        q = query.lower().strip()

        fast_signals = [
            "what is ", "what are ", "who is ", "who are ",
            "when was ", "where is ", "define ", "what does ",
            "how many ", "list ", "name ",
        ]
        verify_signals = [
            "is it true", "prove", "should i", "should we",
            "is this correct", "verify", "fact check", "are you sure",
        ]
        cot_signals = [
            "how ", "why ", "explain", "compare", "analyse", "analyze",
            "what would happen", "difference between", "pros and cons",
            "step by step", "walk me through", "reason", "evaluate",
        ]

        for s in verify_signals:
            if s in q:
                return "verify"
        for s in cot_signals:
            if q.startswith(s) or f" {s}" in q:
                return "cot"
        for s in fast_signals:
            if q.startswith(s):
                return "fast"

        # Default to CoT for anything ambiguous — better quality
        return "cot"

    # ── FAST mode ─────────────────────────────────────────────────────────────

    def _fast(self, query: str, context: str, language: str) -> dict:
        ctx   = f"\nContext:\n{context[:800]}\n" if context else ""
        lang  = f" Respond in {language}." if language != "English" else ""
        prompt = f"{ctx}\nQuestion: {query}{lang}\n\nAnswer concisely:"
        answer = self.engine.generate(prompt, system=self.SYSTEM_FAST)
        return {"answer": answer.strip(), "reasoning": "", "steps": [], "confidence": 0.0}

    # ── COT mode — Chain of Thought ───────────────────────────────────────────

    def _cot(self, query: str, context: str, language: str) -> dict:
        """
        Forces the model to think step by step before answering.

        Prompt structure:
            1. Context (if any)
            2. Question
            3. "Think through this step by step:"
            4. Model writes numbered steps
            5. "Therefore, my answer is:"
            6. Model writes clean final answer

        This dramatically improves accuracy on phi3:mini and small models
        because it prevents the model from jumping to a conclusion.
        """
        ctx  = f"\nContext (use this as your source):\n{context[:1000]}\n" if context else ""
        lang = f"\nRespond in {language}." if language != "English" else ""

        prompt = (
            f"{ctx}"
            f"\nQuestion: {query}"
            f"{lang}"
            f"\n\nThink through this step by step:"
            f"\n1."
        )

        raw = self.engine.generate(prompt, system=self.SYSTEM_COT)

        # Parse steps and final answer
        steps, answer = self._parse_cot(raw, query)

        return {
            "answer":    answer,
            "reasoning": raw,
            "steps":     steps,
            "confidence": 0.0,
        }

    # ── VERIFY mode — CoT + self-check ───────────────────────────────────────

    def _verify(self, query: str, context: str, language: str) -> dict:
        """
        Two-pass reasoning:
        Pass 1: CoT reasoning → draft answer
        Pass 2: Self-verification → check draft against context, correct if wrong

        Best for factual claims, "is it true that" questions, important decisions.
        """
        # Pass 1: Generate CoT answer
        cot_result = self._cot(query, context, language)
        draft      = cot_result["answer"]

        # Pass 2: Verify the draft
        ctx = f"\nContext:\n{context[:600]}\n" if context else ""
        verify_prompt = (
            f"{ctx}"
            f"\nQuestion: {query}"
            f"\n\nDraft answer: {draft}"
            f"\n\nVerify this answer:"
            f"\n- Is it factually correct based on the context?"
            f"\n- Are there any errors or missing information?"
            f"\n- If incorrect, what is the right answer?"
            f"\n\nVerification:"
        )

        verification = self.engine.generate(
            verify_prompt,
            system=self.SYSTEM_VERIFY,
            temperature=0.1,
        )

        # Extract final answer from verification
        final_answer = draft  # default: keep original
        v_lower = verification.lower()

        # If verification found problems, extract the corrected answer
        if any(w in v_lower for w in ["incorrect", "wrong", "error", "actually", "correction"]):
            # Try to get corrected answer
            correction_prompt = (
                f"Based on this verification:\n{verification[:500]}\n\n"
                f"Give the corrected final answer to: {query}\n\nFinal answer:"
            )
            final_answer = self.engine.generate(correction_prompt, temperature=0.1)

        return {
            "answer":       final_answer.strip(),
            "reasoning":    cot_result["reasoning"],
            "verification": verification,
            "steps":        cot_result["steps"],
            "confidence":   0.0,
        }

    # ── Parse CoT output ──────────────────────────────────────────────────────

    def _parse_cot(self, raw: str, query: str) -> tuple[list, str]:
        """
        Extract numbered steps and final answer from CoT output.

        Handles formats:
            1. Step one text
            2. Step two text
            Therefore: Final answer

            Step 1: ...
            Final answer: ...
        """
        lines  = raw.strip().split("\n")
        steps  = []
        answer_lines = []
        in_answer = False

        answer_markers = [
            "therefore", "final answer", "in conclusion", "to summarize",
            "so the answer", "the answer is", "in summary", "conclusion:",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            lower = stripped.lower()
            is_answer_marker = any(lower.startswith(m) for m in answer_markers)

            if is_answer_marker:
                in_answer = True
                # Extract text after the marker
                for marker in answer_markers:
                    if lower.startswith(marker):
                        rest = stripped[len(marker):].lstrip(":").strip()
                        if rest:
                            answer_lines.append(rest)
                        break
            elif in_answer:
                answer_lines.append(stripped)
            else:
                # Check if it looks like a numbered step
                import re
                if re.match(r"^\d+[.)\s]", stripped) or stripped.startswith("Step"):
                    steps.append(stripped)

        # If no clear answer section found, use last 2 lines
        if not answer_lines and lines:
            answer_lines = [l.strip() for l in lines[-3:] if l.strip()]

        answer = " ".join(answer_lines).strip()

        # Final fallback — if answer is too short use the whole output
        if len(answer) < 20:
            answer = raw.strip()

        return steps, answer  # confidence set by Critic


# ── Critic Agent ──────────────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    """
    Scores every answer 0.0–1.0.
    If score is below threshold, asks the Reasoner to try again (up to 2 retries).
    This is what makes the system self-correcting.
    """

    name = "critic"

    SYSTEM = """You are a strict quality evaluator.
Evaluate answers on: accuracy, completeness, relevance, and clarity.
Be harsh — only give high scores to genuinely good answers."""

    def score(self, query: str, answer: str, context: str = "") -> dict:
        """
        Returns:
            {
              "score":    float (0.0 - 1.0),
              "feedback": str,
              "pass":     bool
            }
        """
        t0     = time.time()
        context_note = f"\nAvailable context:\n{context[:500]}\n" if context else ""

        prompt = (
            f"{context_note}"
            f"\nQuestion: {query}"
            f"\nAnswer: {answer}"
            f"\n\nEvaluate this answer. Respond with JSON only:"
            f'\n{{"score": 0.0-1.0, "feedback": "brief reason", "issues": ["list", "of", "issues"]}}'
        )

        console.print("  [dim]Critic scoring...[/]")
        result = self.engine.generate_json(prompt, system=self.SYSTEM)
        score  = float(result.get("score", 0.5))
        ms     = int((time.time() - t0) * 1000)

        self.logger.log_agent_run(
            agent_name=self.name, task=f"scoring: {query[:80]}",
            result=str(score), score=score, latency_ms=ms
        )

        return {
            "score":    score,
            "feedback": result.get("feedback", ""),
            "pass":     score >= 0.65,
        }


# ── Meta-Agent — The Orchestrator ─────────────────────────────────────────────

class MetaAgent(BaseAgent):
    """
    The brain of ARIA. Routes every query to the right path:

    1. Classify intent
    2. Check if specialist agent exists for this domain
    3. Run Researcher → Reasoner → Critic pipeline
    4. If Critic fails: retry with more context
    5. Log everything
    6. Return final answer

    The adaptation loop (in pipelines/adaptation.py) reads the logs and
    decides when to spawn new specialist agents.
    """

    name = "meta"

    INTENTS = [
        "factual_question",   # what/who/where/when
        "reasoning_task",     # why/how/explain
        "code_task",          # write/debug/explain code
        "document_query",     # questions about ingested documents
        "creative_task",      # write/compose/create
        "conversation",       # small talk, greetings
        "unknown",
    ]

    def __init__(
        self,
        engine: Engine,
        memory: Memory,
        logger: Logger,
        researcher: ResearcherAgent,
        reasoner: ReasonerAgent,
        critic: CriticAgent,
    ):
        super().__init__(engine, memory, logger)
        self.researcher = researcher
        self.reasoner   = reasoner
        self.critic     = critic
        self._specialist_registry: dict[str, str] = {}  # domain → model name

    def register_specialist(self, domain: str, model_name: str):
        """Register a fine-tuned specialist model for a domain."""
        self._specialist_registry[domain] = model_name
        console.print(f"[green]Specialist registered:[/] {domain} → {model_name}")

    def run(self, query: str, stream: bool = False) -> dict:
        """
        Full pipeline: classify → research → reason → critique → respond.

        Returns:
            {
              "answer":     str,
              "intent":     str,
              "confidence": float,
              "sources":    list,
              "reasoning":  str,
              "latency_ms": int,
            }
        """
        t0 = time.time()
        console.print(f"\n[bold]Meta-Agent:[/] processing query...")

        # ── Step 1: Detect language ─────────────────────────────────────────
        language = self._detect_language(query)
        console.print(f"  [dim]Language:[/] {language}")

        # ── Step 2: Classify intent ─────────────────────────────────────────
        intent = self.engine.classify(query, self.INTENTS)
        console.print(f"  [dim]Intent:[/] {intent}")

        # ── Step 3: Check for specialist ────────────────────────────────────
        engine_to_use = None
        if intent in self._specialist_registry:
            spec_model  = self._specialist_registry[intent]
            engine_to_use = Engine(model=spec_model)
            console.print(f"  [dim]Specialist:[/] using {spec_model}")

        # ── Step 4: Research ─────────────────────────────────────────────────
        research = self.researcher.run(query, domain=intent)
        context  = research["context"]

        # ── Step 5: Reason (with retry) ──────────────────────────────────────
        best_answer   = ""
        best_score    = 0.0
        best_reasoning = ""
        best_result    = {}

        for attempt in range(1):  # single attempt — faster on low-RAM machines
            result    = self.reasoner.run(query, context=context, language=language)
            answer    = result["answer"]
            reasoning = result["reasoning"]

            # ── Step 6: Critique ─────────────────────────────────────────────
            critique = self.critic.score(query, answer, context=context)
            score    = critique["score"]

            console.print(f"  [dim]Attempt {attempt+1}:[/] score={score:.2f} — {critique['feedback'][:60]}")

            if score > best_score:
                best_score     = score
                best_answer    = answer
                best_reasoning = reasoning
                best_result    = result

        # ── Step 7: Log and return ────────────────────────────────────────────
        ms = int((time.time() - t0) * 1000)
        self.logger.log_interaction(
            query=query,
            response=best_answer,
            agent_used=self.name,
            intent=intent,
            confidence=best_score,
            latency_ms=ms,
            success=best_score >= 0.5,
        )

        console.print(f"  [green]Done[/] — confidence={best_score:.2f}, {ms}ms")

        return {
            "answer":       best_answer,
            "intent":       intent,
            "confidence":   round(best_score, 3),
            "sources":      research["sources"],
            "reasoning":    best_reasoning,
            "steps":        best_result.get("steps", []),
            "mode_used":    best_result.get("mode_used", "cot"),
            "verification": best_result.get("verification", ""),
            "language":     language,
            "latency_ms":   ms,
        }

    def _detect_language(self, text: str) -> str:
        try:
            from langdetect import detect
            code = detect(text)
            lang_map = {
                "hi": "Hindi", "en": "English", "fr": "French",
                "de": "German", "es": "Spanish", "ar": "Arabic",
                "zh-cn": "Chinese", "ja": "Japanese", "pt": "Portuguese",
            }
            return lang_map.get(code, "English")
        except Exception:
            return "English"
