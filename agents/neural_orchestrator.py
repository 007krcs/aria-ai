"""
ARIA NeuralOrchestrator — Neuromorphic 27-Agent Intelligence Engine (v1)
=========================================================================

Every agent is a neuron. Neurons fire signals. Signals propagate. The network
learns which connections matter. This replaces OmegaOrchestrator's flat
parallel fan-out with a two-wave cascading architecture:

  Wave 1 (fast neurons, 0–6s):
      fast_reasoner, memory_retriever, web_searcher, world_model_lookup,
      trend_watcher, browser_controller, network_inspector, code_specialist,
      document_reader, media_controller, calendar_context

  ── Signal propagation ──
      • RESULT signals from Wave 1 written to SynapticState workspace
      • web_searcher EXCITES → fast_reasoner, chain_reasoner, nova_reasoner
      • memory_retriever EXCITES → fast_reasoner
      • High-confidence (≥0.85) results INHIBIT competing agents
      • Lateral inhibition: winner suppresses losers in SynapticState

  Wave 2 (reasoning neurons, 6–16s):
      chain_reasoner, nova_reasoner, sci_researcher, planner_agent,
      system_controller, summarizer_agent, code_runner, automation_controller,
      symbolic_executor

      Each Wave-2 agent calls enrich_context() BEFORE generating — reads all
      Wave-1 results and incorporates them into its LLM prompt. This is the
      critical difference: Wave-2 agents don't reason in a vacuum.

  Synthesis:
      Try SynapticState.try_form_consensus() — if 2+ agents agree, use it.
      Otherwise: LLM merge of top-K results.

  Hebbian learning (background):
      For each wave1→wave2 pair that both produced results, reinforce their
      synapse weight. Over many queries, high-synergy pairs grow stronger.
      Weights decay slowly. Saved to data/synaptic_weights.json.

SSE output format is identical to OmegaOrchestrator so the frontend is unaware.
Extra SSE event types for the UI to optionally visualise:
    {"type": "neural_signal",   "from": "...", "to": "...", "stype": "EXCITATORY", ...}
    {"type": "agent_fired",     "agent": "...", "wave": 1, "confidence": 0.8}
    {"type": "agent_suppressed","agent": "...", "by": "...", "reason": "..."}
    {"type": "consensus",       "agents": [...], "confidence": 0.87}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import re
from typing import AsyncGenerator, Optional, Any
from datetime import datetime

from core.neural_bus      import NeuralBus, NeuralSignal, SignalType
from core.synaptic_state  import SynapticState
from core.neural_agent    import NeuralAgentMixin
from agents.omega_orchestrator import IntentMap, AgentResult

# ── Algo-core imports (graceful fallback if not yet available) ────────────────
try:
    from agents.algo_core import PatternEngine, DecisionEngine, AdaptiveLearner
    _ALGO_CORE = True
except ImportError:
    _ALGO_CORE = False

# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOPOLOGY — WHO EXCITES AND INHIBITS WHOM
# ─────────────────────────────────────────────────────────────────────────────

# When agent_A fires, it emits EXCITATORY to each agent in its list.
# Wave-2 agents boosted here run with higher priority.
EXCITE_MAP: dict[str, list[str]] = {
    "web_searcher":       ["fast_reasoner", "chain_reasoner", "nova_reasoner", "trend_watcher", "sci_researcher", "cot_thinker"],
    "memory_retriever":   ["fast_reasoner", "chain_reasoner", "cot_thinker"],
    "sci_researcher":     ["chain_reasoner", "nova_reasoner", "cot_thinker", "physics_solver"],
    "code_specialist":    ["code_runner", "app_auditor"],
    "browser_controller": ["document_reader", "web_builder", "scam_scanner"],
    "network_inspector":  ["system_controller", "threat_detector"],
    "world_model_lookup": ["fast_reasoner", "chain_reasoner"],
    "trend_watcher":      ["chain_reasoner"],
    "document_reader":    ["summarizer_agent", "chain_reasoner", "cot_thinker"],
    "calendar_context":   ["planner_agent"],
    "media_controller":   [],
    "fast_reasoner":      [],
    "os_profiler":        ["terminal_runner", "threat_detector", "system_controller"],
    "terminal_runner":    ["app_auditor", "code_runner"],
    "threat_detector":    ["system_controller"],
    "cot_thinker":        ["chain_reasoner", "nova_reasoner"],
    "physics_solver":     ["chain_reasoner"],
    "academic_solver":    ["chain_reasoner", "nova_reasoner"],
    "quantum_reasoner":   ["chain_reasoner", "nova_reasoner"],
    "web_builder":        ["code_runner", "app_auditor"],
    "scam_scanner":       ["chain_reasoner"],
    "lang_detector":      ["fast_reasoner", "chain_reasoner"],
    "stock_analyst":      ["chain_reasoner", "nova_reasoner"],
    "intent_classifier":  ["fast_reasoner", "system_controller", "planner_agent"],
    "undo_tracker":       ["chain_reasoner"],
    "sentiment_analyst":  ["chain_reasoner", "nova_reasoner", "stock_analyst"],
    "win_kernel_agent":   ["system_controller", "terminal_runner", "chain_reasoner"],
    "knowledge_verifier": ["chain_reasoner", "fast_reasoner"],
    "env_context":        ["fast_reasoner", "system_controller"],
}

# (agent, min_confidence) → inhibit target agent
INHIBIT_MAP: dict[str, list[tuple[str, float]]] = {
    "fast_reasoner":   [("chain_reasoner", 0.90)],
    "chain_reasoner":  [("nova_reasoner", 0.88)],
    "code_specialist": [("code_runner", 0.95)],    # already has code → skip execution
    "nova_reasoner":   [],
}

# Wave 1: fast agents (no peer context needed — they are the first pulse)
WAVE1_AGENTS = [
    "fast_reasoner",
    "memory_retriever",
    "web_searcher",
    "world_model_lookup",
    "trend_watcher",
    "browser_controller",
    "network_inspector",
    "code_specialist",
    "document_reader",
    "media_controller",
    "calendar_context",
    # New Wave-1 agents (fast detection / sensing)
    "os_profiler",          # detect OS + kernel + capabilities
    "threat_detector",      # security scan
    "activity_context",     # personalization context from user history
    "scam_scanner",         # phishing / fake-site detection before content shown
    "lang_detector",        # auto-detect non-English input, respond in same language
    "stock_analyst",        # 12-layer stock ranking when market/share queries detected
    "intent_classifier",    # zero-LLM action classification — gates wave-2 selection
    "undo_tracker",         # tracks undo context when undo/repeat detected
    "sentiment_analyst",    # AMIA 9-signal psychology+sentiment+operator+institutional
    "win_kernel_agent",     # Windows shell/Win+R/PowerShell/CMD/Git/Chrome/Java/Code
    "knowledge_verifier",   # anti-hallucination — ground every claim in verified knowledge
    "env_context",          # environment context — what apps/projects the user has
]

# Wave 2: reasoning agents (run AFTER Wave 1, WITH peer context)
WAVE2_AGENTS = [
    "chain_reasoner",
    "nova_reasoner",
    "sci_researcher",
    "planner_agent",
    "system_controller",
    "summarizer_agent",
    "code_runner",
    "automation_controller",
    "symbolic_executor",
    "desktop_controller",
    # New Wave-2 agents (deep reasoning / execution)
    "cot_thinker",          # human-like chain of thought
    "terminal_runner",      # run/fix/test in terminal
    "web_builder",          # read-web → implement code
    "app_auditor",          # test + find limitations + fix
    "auto_optimizer",       # self-tune model performance
]

# Academic agents are pure Python (SymPy) — instant, no LLM needed for exact subjects.
# Added to WAVE1 so they fire in parallel and return before synthesis.
WAVE1_AGENTS += [
    "physics_solver",       # kinematics/optics/EM/thermo/waves — exact SymPy solve
    "academic_solver",      # math/chemistry/biology/CS/logic — structured step-by-step
]
WAVE2_AGENTS += [
    "physics_solver",       # also in Wave 2 as LLM fallback when SymPy couldn't solve
    "academic_solver",
    "quantum_reasoner",     # quantum-inspired superposition reasoning (4 parallel paths)
]

WAVE1_TIMEOUT_S    = 25.0   # physics/academic solvers: SymPy instant + Groq ~10s
WAVE2_TIMEOUT_S    = 45.0
SIGNAL_SETTLE_S    = 0.05   # reduced: less dead-wait between waves
INHIBIT_THRESHOLD  = 0.98   # only suppress when near-certain (prevents intent_classifier killing fast_reasoner)

# Agents that must NEVER be inhibited — they provide the actual answer
_NO_INHIBIT = {"fast_reasoner", "chain_reasoner", "nova_reasoner", "cot_thinker"}


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class NeuralOrchestrator:
    """
    Neuromorphic 27-agent parallel intelligence engine.
    Drop-in replacement for OmegaOrchestrator — same stream() signature.
    """

    def __init__(
        self,
        aria_components: dict,
        neural_bus:      NeuralBus,
        synaptic_state:  SynapticState,
    ):
        self.aria    = aria_components
        self.engine  = aria_components.get("engine")
        self.memory  = aria_components.get("memory")
        self.bus     = neural_bus
        self.state   = synaptic_state

        # Subscribe the orchestrator to ALL signals (monitoring)
        self.bus.subscribe_all(self._on_any_signal, agent_name="orchestrator")

        # Signal event queue for streaming to frontend
        self._signal_queue: asyncio.Queue = None   # created per stream() call

        self._started  = False
        self._bg_task: Optional[asyncio.Task] = None

        # Model names from env
        import os
        _cfg_model = os.getenv("DEFAULT_MODEL", "aria-custom")
        # aria-custom is fine-tuned on Q&A pairs — it echoes training format for open questions.
        # Prefer llama3.1/llama3.2 for reasoning; fall back to cfg model only if needed.
        _reasoning_candidates = ["llama3.2", "llama3.1", "llama3", "phi3"]
        _eng = self.engine
        _available = _eng.list_models() if _eng and hasattr(_eng, "list_models") else []
        _reasoning_model = _cfg_model  # default
        for _cand in _reasoning_candidates:
            for _m in _available:
                if _m.lower().startswith(_cand):
                    _reasoning_model = _m
                    break
            else:
                continue
            break
        self._fast_model = _reasoning_model
        self._deep_model = "llama3.1:8b"

        # Pre-warm the reasoning model in background so first request is fast.
        # Run in a daemon thread — completes before any user request on typical startup.
        if self._fast_model and self._fast_model != _cfg_model and _eng:
            import threading as _threading
            def _prewarm_reasoning():
                try:
                    _eng.generate("hello", model=self._fast_model,
                                  max_tokens=1, use_cache=False, timeout_s=90)
                except Exception:
                    pass
            _threading.Thread(target=_prewarm_reasoning, daemon=True).start()

        # ── Upgrade A: agent confidence tracking ──────────────────────────────
        # Initialise all known agents at 0.5 (50% prior confidence)
        _all_agents = list(set(WAVE1_AGENTS + WAVE2_AGENTS))
        self._agent_confidence: dict[str, float] = {a: 0.5 for a in _all_agents}

        # ── Upgrade A: agent descriptions for semantic selection ──────────────
        self._agent_descriptions: dict[str, str] = {
            "fast_reasoner":      "quick general question answering reasoning facts",
            "memory_retriever":   "retrieve past conversations user history memory",
            "web_searcher":       "live web search internet real-time news",
            "world_model_lookup": "world knowledge entity facts encyclopedia",
            "trend_watcher":      "trending topics current events live market trends",
            "browser_controller": "open browser navigate url web pages",
            "network_inspector":  "network diagnostics ip connectivity wifi",
            "code_specialist":    "code programming software development",
            "document_reader":    "read parse pdf document file",
            "media_controller":   "music video media player audio volume",
            "calendar_context":   "calendar schedule events appointments",
            "os_profiler":        "operating system hardware cpu ram profiling",
            "threat_detector":    "security threat malware virus scan protection",
            "activity_context":   "user activity personalization history profile",
            "scam_scanner":       "phishing scam fraud website detection",
            "lang_detector":      "language detection translation multilingual",
            "stock_analyst":      "stock market shares investment equity finance",
            "intent_classifier":  "classify intent action type categorize",
            "undo_tracker":       "undo revert cancel previous action",
            "sentiment_analyst":  "sentiment emotion psychology market mood",
            "win_kernel_agent":   "windows shell powershell cmd terminal command",
            "knowledge_verifier": "verify facts knowledge grounded anti-hallucination",
            "env_context":        "environment projects apps installed software context",
            "chain_reasoner":     "step by step logical reasoning chain of thought",
            "nova_reasoner":      "deep advanced multi-step reasoning nova",
            "sci_researcher":     "science research academic papers methodology",
            "planner_agent":      "plan steps workflow roadmap multi-step task",
            "system_controller":  "system control file manager process manager",
            "summarizer_agent":   "summarize condense abstract shorten text",
            "code_runner":        "execute run code terminal output result",
            "automation_controller": "automate repetitive tasks workflow automation",
            "symbolic_executor":  "math calculation symbolic computation equation",
            "desktop_controller": "desktop gui click mouse keyboard interaction",
            "cot_thinker":        "chain of thought reasoning human-like thinking",
            "terminal_runner":    "terminal bash run fix test command line",
            "web_builder":        "build website frontend backend implement code web",
            "app_auditor":        "audit test application quality bugs limitations",
            "auto_optimizer":     "optimize performance tuning model configuration",
            "physics_solver":     "physics problem kinematics dynamics optics thermodynamics electromagnetism waves quantum",
            "academic_solver":    "math chemistry biology history economics geography literature reasoning logic computer science astronomy",
            "quantum_reasoner":   "complex analytical deep reasoning multi-step analysis explain why how compare evaluate argue debate philosophy strategy",
        }

    # ── Startup ───────────────────────────────────────────────────────────────

    def start_background(self) -> None:
        if self._started:
            return
        try:
            loop = asyncio.get_event_loop()
            self._bg_task = loop.create_task(self._background_loop())
            self._started = True
            print("  [OK] NeuralOrchestrator background loop started")
        except Exception as e:
            print(f"  [WARN] NeuralOrchestrator bg start failed: {e}")

    def _on_any_signal(self, signal: NeuralSignal) -> None:
        """Called by bus for every signal — put into SSE queue for streaming."""
        if self._signal_queue is not None:
            try:
                self._signal_queue.put_nowait(signal)
            except asyncio.QueueFull:
                pass

    # ── Main stream entry point ────────────────────────────────────────────────

    async def stream(self, query: str, city: str = "") -> AsyncGenerator[str, None]:
        """
        Main entry — same signature as OmegaOrchestrator.stream().
        Yields SSE-formatted strings.
        """
        t_start = time.time()
        self._signal_queue = asyncio.Queue(maxsize=200)

        # Reset per-query state on bus and workspace
        self.bus.reset_per_query()
        self.state.clear()

        query_hash = hashlib.sha1(query.strip().lower().encode()).hexdigest()[:12]
        intent     = IntentMap.classify(query)

        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        # ── Tier 0: Instant rules (no LLM) ───────────────────────────────────
        instant = await self._tier0(query)
        if instant:
            yield sse({"type": "status", "text": "⚡ Answered instantly"})
            yield sse({"type": "token",  "text": instant})
            yield sse({"type": "done",   "mode": "instant",
                       "ms": int((time.time() - t_start) * 1000)})
            asyncio.create_task(self._background_store(query, instant, 0.95, "instant"))
            return

        # ── Pre-fetch live stock data for stock/share queries ─────────────────
        _live_stock_ctx = ""
        _stock_kw = ("share", "stock", "price", "buy", "sell", "invest", "analyse", "analyze")
        if any(kw in query.lower() for kw in _stock_kw):
            _ticker = self._extract_ticker(query.lower())
            if _ticker:
                _sd = await self._fetch_stock(_ticker)
                if _sd and _sd.get("ok") and _sd.get("price"):
                    _arrow = "▲" if _sd["change_pct"] >= 0 else "▼"
                    _live_stock_ctx = (
                        f"\n\n[LIVE DATA — {datetime.now().strftime('%d %b %Y %I:%M %p')}]\n"
                        f"Ticker: {_sd['ticker']}  |  Name: {_sd['name']}\n"
                        f"Price: {_sd['currency']} {_sd['price']:,.2f}  |  "
                        f"Change: {_arrow}{abs(_sd['change_pct'])}%\n"
                        f"52W High: {_sd.get('52w_high','—')}  |  52W Low: {_sd.get('52w_low','—')}\n"
                        f"P/E: {_sd.get('pe_ratio','—')}  |  Sector: {_sd.get('sector','—')}\n"
                        f"Use this real data in your analysis. Do NOT make up prices.\n"
                    )
                    # Inject into query context for all agents
                    query = query + _live_stock_ctx

        # ── Select agents (Upgrade A: semantic + confidence-weighted) ───────────
        w1_names = self._select_wave1(query=query, intent=intent)
        w2_names = self._select_wave2(intent, set(), query=query)  # inhibited updated after wave1

        total = len(w1_names) + len(w2_names)
        yield sse({
            "type":   "status",
            "text":   f"🧠 Neural network activating {total} agents across 2 waves…",
            "wave1":  w1_names,
            "wave2":  w2_names,
        })
        await asyncio.sleep(0)

        # ── Wave 1: fast agents fire (no peer context) ────────────────────────
        yield sse({"type": "status", "text": "⚡ Wave 1 — fast agents firing…"})
        wave1_results: list[AgentResult] = []
        first_text = ""

        async for result in self._run_wave(
            wave=1, agent_names=w1_names,
            query=query, city=city, intent=intent,
            query_hash=query_hash,
            timeout_s=WAVE1_TIMEOUT_S,
            with_peer_context=False,
        ):
            wave1_results.append(result)

            # Stream first usable text result
            if not first_text and result.content and result.agent_type == "text":
                first_text = result.content
                yield sse({"type": "agent_fired",  "agent": result.agent,
                           "wave": 1, "confidence": round(result.confidence, 2)})
                for chunk in self._chunk(first_text):
                    yield sse({"type": "token", "text": chunk})
                    await asyncio.sleep(0)
            else:
                yield sse({"type": "agent_fired", "agent": result.agent,
                           "wave": 1, "confidence": round(result.confidence, 2)})

            # Stream any pending neural signals to UI
            async for sig_sse in self._drain_signal_queue():
                yield sig_sse

        # ── Signal propagation between waves ──────────────────────────────────
        yield sse({"type": "status", "text": "🔀 Propagating neural signals…"})
        await asyncio.sleep(SIGNAL_SETTLE_S)

        # Apply lateral inhibition and discover inhibited agents
        inhibited = await self._apply_post_wave1_signals(
            wave1_results, topic=query_hash, sse_fn=lambda o: sse(o)
        )
        # Stream inhibition events
        for agent_name in inhibited:
            yield sse({"type": "agent_suppressed", "agent": agent_name,
                       "reason": "lateral inhibition from high-confidence peer"})

        # Drain any signals that fired during inhibition processing
        async for sig_sse in self._drain_signal_queue():
            yield sig_sse

        # ── Fast-path: skip Wave 2 if Wave 1 already has a high-confidence answer ──
        # "data" agents (stock_analyst, scam_scanner, etc.) return complete self-contained
        # answers — no need for Wave2 reasoning to enrich them.
        _top_w1 = max((r.confidence for r in wave1_results if r.content), default=0.0)
        _has_data_answer = any(
            r.agent_type == "data" and r.confidence >= 0.80 and r.content
            for r in wave1_results
        )
        _skip_wave2 = _has_data_answer or _top_w1 >= 0.92

        # ── Wave 2: reasoning agents WITH peer context ─────────────────────────
        w2_names_filtered = [] if _skip_wave2 else self._select_wave2(intent, inhibited, query=query)
        if w2_names_filtered:
            yield sse({"type": "status",
                       "text": f"🌊 Wave 2 — {len(w2_names_filtered)} reasoning agents enriching with peer context…"})
            wave2_results: list[AgentResult] = []

            async for result in self._run_wave(
                wave=2, agent_names=w2_names_filtered,
                query=query, city=city, intent=intent,
                query_hash=query_hash,
                timeout_s=WAVE2_TIMEOUT_S,
                with_peer_context=True,
            ):
                wave2_results.append(result)
                yield sse({"type": "agent_fired", "agent": result.agent,
                           "wave": 2, "confidence": round(result.confidence, 2)})
                async for sig_sse in self._drain_signal_queue():
                    yield sig_sse
        else:
            wave2_results = []

        # ── Consensus + Synthesis ─────────────────────────────────────────────
        all_results = wave1_results + wave2_results
        yield sse({"type": "status", "text": "🔮 Synthesizing neural consensus…"})
        await asyncio.sleep(0)

        # Try consensus from SynapticState first
        consensus = self.state.try_form_consensus(topic=query_hash)
        if consensus:
            final_text = str(consensus.value)
            conf = consensus.confidence
            contributing = self.state.read_all_live()
            agents_agreed = list({e.author for e in contributing if e.author != "consensus"})
            yield sse({"type": "consensus",
                       "agents": agents_agreed,
                       "confidence": round(conf, 2)})
        else:
            # LLM synthesis from top results
            synthesis = await self._synthesize(query, intent, all_results)
            final_text = synthesis if synthesis else (first_text or "")
            conf = self._quick_score(final_text, all_results)

        # Stream final answer (replace partial wave-1 stream if we have something better)
        if final_text and final_text != first_text:
            yield sse({"type": "replace", "text": ""})
            for chunk in self._chunk(final_text):
                yield sse({"type": "token", "text": chunk})
                await asyncio.sleep(0)

        ms = int((time.time() - t_start) * 1000)
        yield sse({
            "type":        "done",
            "mode":        "neural",
            "agents_used": len(all_results),
            "wave1":       len(wave1_results),
            "wave2":       len(wave2_results),
            "ms":          ms,
            "confidence":  round(conf, 2),
            "text":        final_text,
        })

        # ── Background: Hebbian update + memory store ─────────────────────────
        asyncio.create_task(
            self._background_post_query(
                query, final_text, conf, wave1_results, wave2_results
            )
        )
        self._signal_queue = None

    # ── Wave execution ─────────────────────────────────────────────────────────

    async def _run_wave(
        self,
        wave:            int,
        agent_names:     list[str],
        query:           str,
        city:            str,
        intent:          dict,
        query_hash:      str,
        timeout_s:       float,
        with_peer_context: bool = False,
    ) -> AsyncGenerator[AgentResult, None]:
        """
        Fire all agents in the wave simultaneously.
        with_peer_context=True: each agent gets self.state.build_context() prepended.
        Yields AgentResult as they complete (fastest first).
        """
        loop  = asyncio.get_event_loop()
        queue = asyncio.Queue()

        async def run_one(name: str):
            # Skip suppressed agents
            if self.bus.is_suppressed(name):
                await queue.put(None)
                return
            try:
                # Build peer context for wave-2 agents
                peer_ctx = ""
                if with_peer_context:
                    peer_ctx = self.state.build_context(
                        exclude_author=name,
                        max_chars=800,
                    )
                fn = self._get_agent_fn(
                    name, query, city, intent, query_hash, wave, peer_ctx
                )
                res = await asyncio.wait_for(
                    loop.run_in_executor(None, fn),
                    timeout=timeout_s,
                )
                if res:
                    await queue.put(res)
                    # Emit fire signals from bus topology
                    await self._emit_excitations(name, res.confidence, query_hash, wave)
            except (asyncio.TimeoutError, Exception) as e:
                pass
            finally:
                await queue.put(None)

        tasks = [asyncio.create_task(run_one(n)) for n in agent_names]
        done  = 0
        deadline = time.time() + timeout_s + 1.0

        while done < len(tasks):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(
                    queue.get(), timeout=min(remaining, 2.0)
                )
                if item is None:
                    done += 1
                elif isinstance(item, AgentResult):
                    yield item
            except asyncio.TimeoutError:
                continue

        for t in tasks:
            if not t.done():
                t.cancel()

    # ── Signal propagation ─────────────────────────────────────────────────────

    async def _apply_post_wave1_signals(
        self,
        results:   list[AgentResult],
        topic:     str,
        sse_fn:    Any,
    ) -> set[str]:
        """
        Apply lateral inhibition for high-confidence wave-1 results.
        Returns set of inhibited agent names.
        """
        all_inhibited: set[str] = set()
        for result in sorted(results, key=lambda r: -r.confidence):
            if result.confidence >= INHIBIT_THRESHOLD:
                suppressed = self.state.apply_lateral_inhibition(
                    winner_agent=result.agent,
                    topic=topic,
                    confidence_threshold=INHIBIT_THRESHOLD,
                )
                for s in suppressed:
                    # Never suppress core reasoning agents — they provide the actual answer
                    if s in _NO_INHIBIT:
                        continue
                    all_inhibited.add(s)
                    # Mark as suppressed on bus too
                    self.bus._apply_side_effects(
                        NeuralSignal(
                            signal_type=SignalType.INHIBITORY,
                            source_agent=result.agent,
                            topic=topic,
                            payload={"reason": f"lateral inhibition conf={result.confidence:.2f}"},
                            confidence=result.confidence,
                            target_agent=s,
                            ttl=30.0,
                        )
                    )
        return all_inhibited

    async def _emit_excitations(
        self,
        source_agent: str,
        confidence:   float,
        topic:        str,
        wave:         int,
    ) -> None:
        """Emit EXCITATORY signals to downstream agents per EXCITE_MAP."""
        targets = EXCITE_MAP.get(source_agent, [])
        for target in targets:
            self.bus.emit(
                signal_type=SignalType.EXCITATORY,
                source_agent=source_agent,
                topic=topic,
                payload={"reason": f"{source_agent} has new data (conf={confidence:.2f})"},
                confidence=confidence * 0.8,
                target_agent=target,
                wave=wave,
                ttl=15.0,
            )
        # Lateral inhibition from excite map
        for (target, min_conf) in INHIBIT_MAP.get(source_agent, []):
            if confidence >= min_conf:
                self.bus.emit(
                    signal_type=SignalType.INHIBITORY,
                    source_agent=source_agent,
                    topic=topic,
                    payload={"reason": f"high-confidence answer from {source_agent}"},
                    confidence=confidence,
                    target_agent=target,
                    ttl=25.0,
                )

    # ── Signal queue drain ────────────────────────────────────────────────────

    async def _drain_signal_queue(self) -> AsyncGenerator[str, None]:
        """Yield SSE events for any queued neural signals."""
        if not self._signal_queue:
            return
        while True:
            try:
                sig: NeuralSignal = self._signal_queue.get_nowait()
                yield f"data: {json.dumps({'type': 'neural_signal', **sig.to_dict()})}\n\n"
            except asyncio.QueueEmpty:
                break

    # ── Agent selection ───────────────────────────────────────────────────────

    def _semantic_agent_score(self, query: str, agent_name: str) -> float:
        """
        Upgrade A: Semantic score for agent selection.
        Combines PatternEngine similarity with Hebbian confidence history.
        """
        if not _ALGO_CORE:
            return self._agent_confidence.get(agent_name, 0.5)
        desc = self._agent_descriptions.get(agent_name, agent_name)
        sim = PatternEngine.semantic_similarity(query, desc, n=3)
        conf = self._agent_confidence.get(agent_name, 0.5)
        # Weighted: 60% semantic relevance, 40% historical confidence
        return 0.6 * sim + 0.4 * conf

    def _select_wave1(self, query: str = "", intent: dict = None) -> list[str]:
        if intent is None:
            intent = {}
        """Select wave-1 agents based on intent flags."""
        _is_solver_query = intent.get("is_physics") or intent.get("is_math")

        # For physics/math: skip LLM-heavy fast_reasoner + web_searcher — solver handles it.
        # Only memory_retriever (no LLM) is kept for context.
        if _is_solver_query:
            agents = ["memory_retriever"]
        else:
            agents = ["fast_reasoner", "memory_retriever", "web_searcher"]

        # Academic/physics agents fire immediately in Wave 1 — pure Python (SymPy)
        if _is_solver_query:
            agents.append("physics_solver")
            agents.append("academic_solver")
        elif intent.get("is_complex") or intent.get("is_science"):
            agents.append("academic_solver")

        if intent.get("is_live") or intent.get("is_trend"):
            agents.append("trend_watcher")
        if intent.get("is_code") or intent.get("is_exec") or intent.get("is_terminal"):
            agents.append("code_specialist")
        if intent.get("is_calendar"):
            agents.append("calendar_context")
        # Desktop only if NOT a physics/math/academic query
        if (intent.get("is_desktop") or intent.get("is_terminal")) and not intent.get("is_physics") and not intent.get("is_math"):
            agents.append("desktop_controller")
        if intent.get("is_browser") or intent.get("is_web_impl"):
            agents.append("browser_controller")
        if intent.get("is_document"):
            agents.append("document_reader")
        if intent.get("is_media"):
            agents.append("media_controller")
        if intent.get("is_network") or intent.get("is_security"):
            agents.append("network_inspector")
        # Always fire OS profiler for system/terminal/security queries
        if intent.get("is_terminal") or intent.get("is_system") or intent.get("is_security"):
            agents.append("os_profiler")
        # Threat detector on any security signal
        if intent.get("is_security"):
            agents.append("threat_detector")
        # Scam scanner fires whenever query contains a URL
        import re as _re_check
        if _re_check.search(r"https?://|(?:www\.)?[a-z0-9-]+\.[a-z]{2,}", query, _re_check.IGNORECASE):
            agents.append("scam_scanner")
        # Language detector fires on non-ASCII or translate/language queries
        _lang_kw = {"translate", "in hindi", "in french", "in german", "in japanese",
                    "in spanish", "en espanol", "auf deutsch", "en francais",
                    "grammar", "correct my", "language", "detect language"}
        if (intent.get("lang") not in ("en", "") or
                any(k in query.lower() for k in _lang_kw) or
                any(ord(c) > 127 for c in query)):
            agents.append("lang_detector")
        # Stock analyst fires on market/share/stock queries
        _stock_kw = {"stock", "share", "market", "top 10", "best stocks", "invest",
                     "nifty", "nasdaq", "sensex", "ftse", "dax", "nikkei",
                     "equity", "portfolio", "dividend", "pe ratio", "bull", "bear"}
        if any(k in query.lower() for k in _stock_kw):
            agents.append("stock_analyst")
        # Intent classifier always fires — zero-LLM regex gate
        agents.append("intent_classifier")
        # Undo tracker fires on undo/repeat signals
        _undo_kw = {"undo", "revert", "go back", "cancel that", "again", "repeat", "take that back"}
        if any(k in query.lower() for k in _undo_kw):
            agents.append("undo_tracker")
        # Sentiment analyst fires on psychology/sentiment/operator/market queries
        _sent_kw = {
            "sentiment", "psychology", "operator", "whale", "institutional",
            "fear", "greed", "panic", "euphoria", "fii", "dii", "mutual fund",
            "promoter", "block deal", "bulk deal", "earnings", "result day",
            "fomo", "bull trap", "bear trap", "distribution", "accumulation",
            "jp morgan", "goldman", "warren buffett", "rakesh", "market mood",
            "analyze stock", "buy or sell", "should i buy", "trade or invest",
        }
        if any(k in query.lower() for k in _sent_kw):
            agents.append("sentiment_analyst")
        # Windows kernel agent fires on shell/system/code/app queries
        _win_kw = {
            "powershell", "cmd", "command prompt", "bash", "git", "win+r",
            "run dialog", "open chrome", "chrome profile", "edge profile",
            "java project", "maven", "gradle", "pom.xml", "build.gradle",
            "write code", "write python", "write java", "write script",
            "create algorithm", "code for", "algorithm for", "write a program",
            "open app", "launch app", "start app", "open notepad", "open calculator",
            "task manager", "control panel", "regedit", "open run", "taskmgr",
        }
        if any(k in query.lower() for k in _win_kw):
            agents.append("win_kernel_agent")
        # Activity context always fires (very fast — reads cached profile)
        agents.append("activity_context")
        # Always include world model for named-entity queries
        agents.append("world_model_lookup")

        # ── Upgrade A: semantic boost — add extra Wave-1 agents that score high ──
        # For physics/math queries: physics_solver and academic_solver are already
        # in the list. Skip all LLM-heavy agents to prevent them being added back
        # via semantic similarity — physics/math solver handles these exactly.
        _llm_heavy = (
            {"fast_reasoner", "web_searcher", "trend_watcher",
             "stock_analyst", "sci_researcher", "app_auditor",
             "win_kernel_agent", "chain_reasoner", "nova_reasoner"}
            if _is_solver_query else set()
        )

        if _ALGO_CORE and query:
            semantic_threshold = 0.25
            for candidate in WAVE1_AGENTS:
                if candidate not in agents and candidate not in _llm_heavy:
                    score = self._semantic_agent_score(query, candidate)
                    if score >= semantic_threshold:
                        agents.append(candidate)

        return agents

    def _select_wave2(self, intent: dict, inhibited: set[str], query: str = "") -> list[str]:
        """Select wave-2 agents, excluding inhibited ones."""
        agents = []

        if intent.get("is_math"):
            agents.append("symbolic_executor")
        if intent.get("is_math") or intent.get("is_long"):
            agents.append("chain_reasoner")
        if intent.get("is_science"):
            agents.append("sci_researcher")
        if intent.get("is_plan") or intent.get("is_multi_step"):
            agents.append("planner_agent")
        if intent.get("is_summary"):
            agents.append("summarizer_agent")
        if intent.get("is_system"):
            agents.append("system_controller")
        if intent.get("is_exec"):
            agents.append("code_runner")
        if intent.get("is_automate"):
            agents.append("automation_controller")
        if intent.get("is_long") or intent.get("is_science"):
            agents.append("nova_reasoner")
        # New: CoT for any complex reasoning query
        if intent.get("is_long") or intent.get("is_complex") or intent.get("is_science"):
            agents.append("cot_thinker")
        # Terminal runner for run/fix/test queries
        if intent.get("is_terminal") or intent.get("is_exec"):
            agents.append("terminal_runner")
        # Web builder for read-and-implement tasks
        if intent.get("is_web_impl"):
            agents.append("web_builder")
        # App auditor for test/fix/audit queries
        if intent.get("is_app_test"):
            agents.append("app_auditor")
        # Physics solver fires on any physics query
        if intent.get("is_physics") or intent.get("is_math"):
            agents.append("physics_solver")
        # Academic solver fires on math, science, humanities, reasoning
        if (intent.get("is_math") or intent.get("is_science") or intent.get("is_physics")
                or intent.get("is_long") or intent.get("is_complex")):
            agents.append("academic_solver")
        # Auto optimizer fires on performance/tuning queries
        if intent.get("is_tune"):
            agents.append("auto_optimizer")
        # Quantum reasoner fires on complex analytical/multi-step reasoning queries
        # NOT on physics/math (those use dedicated solvers)
        _analytical_kw = {
            "explain", "why", "how does", "analyze", "analyse", "compare",
            "difference", "impact", "effect", "strategy", "evaluate",
            "argue", "debate", "philosophy", "critically", "implications",
            "pros and cons", "advantages", "disadvantages", "discuss",
            "what causes", "reason for", "purpose of", "significance",
        }
        if (not intent.get("is_physics") and not intent.get("is_math")
                and (intent.get("is_complex") or intent.get("is_long")
                     or intent.get("is_science")
                     or (query and any(k in query.lower() for k in _analytical_kw)))):
            agents.append("quantum_reasoner")

        # Always run chain_reasoner for complex queries
        if not agents or intent.get("is_long"):
            if "chain_reasoner" not in agents:
                agents.append("chain_reasoner")

        # Remove inhibited
        return [a for a in agents if a not in inhibited]

    # ── Tier 0 ────────────────────────────────────────────────────────────────

    async def _tier0(self, query: str) -> Optional[str]:
        """Instant answers — no LLM needed."""
        q = query.lower().strip().rstrip("?")
        if any(x in q for x in ["what time", "current time", "what's the time", "time now"]):
            return f"It's **{datetime.now().strftime('%I:%M %p')}** on {datetime.now().strftime('%A, %B %d %Y')}."
        if any(x in q for x in ["what day", "today's date", "what's today", "today date"]):
            return f"Today is **{datetime.now().strftime('%A, %B %d, %Y')}**."
        # Safe math eval
        expr = re.sub(r'(what is|calculate|compute|=|\?)', '', q).strip()
        if re.match(r'^[\d\s\+\-\*\/\(\)\.\%\^]+$', expr) and len(expr) < 40:
            try:
                result = eval(re.sub(r'\^', '**', expr), {"__builtins__": {}})
                return f"{query.strip()} = **{result}**"
            except Exception:
                pass
        # Live stock price lookup — "price of MCX", "MCX share price", "what is MCX trading at"
        _price_kw = ("price of ", "share price", "stock price", "trading at", "current price",
                     "what is.*price", "quote.*", "₹.*share")
        if any(kw in q for kw in ("share price", "stock price", "trading at", "current price")):
            ticker = self._extract_ticker(q)
            if ticker:
                data = await self._fetch_stock(ticker)
                if data and data.get("ok"):
                    arrow = "▲" if data["change_pct"] >= 0 else "▼"
                    return (
                        f"**{data['name']} ({data['ticker']})**\n\n"
                        f"| | |\n|---|---|\n"
                        f"| **Live Price** | {data['currency']} {data['price']:,.2f} |\n"
                        f"| **Change** | {arrow} {abs(data['change_pct'])}% |\n"
                        f"| **52W High** | {data['currency']} {data.get('52w_high','—')} |\n"
                        f"| **52W Low** | {data['currency']} {data.get('52w_low','—')} |\n"
                        f"| **P/E Ratio** | {data.get('pe_ratio','—')} |\n"
                        f"| **Sector** | {data.get('sector','—')} |\n\n"
                        f"*Data via Yahoo Finance — {datetime.now().strftime('%d %b %Y %I:%M %p')}*"
                    )
        return None

    _INDIAN_TICKERS = {
        "mcx": "MCX.NS", "reliance": "RELIANCE.NS", "tcs": "TCS.NS",
        "infosys": "INFY.NS", "infy": "INFY.NS", "hdfc": "HDFCBANK.NS",
        "icici": "ICICIBANK.NS", "wipro": "WIPRO.NS", "sbi": "SBIN.NS",
        "bajaj finance": "BAJFINANCE.NS", "bajfinance": "BAJFINANCE.NS",
        "airtel": "BHARTIARTL.NS", "tata motors": "TATAMOTORS.NS",
        "asian paints": "ASIANPAINT.NS", "titan": "TITAN.NS",
        "ongc": "ONGC.NS", "ntpc": "NTPC.NS", "l&t": "LT.NS",
    }

    def _extract_ticker(self, q: str) -> Optional[str]:
        """Extract a stock ticker from the query string."""
        # Check known Indian names first
        for name, ticker in self._INDIAN_TICKERS.items():
            if name in q:
                return ticker
        # Look for uppercase ticker patterns like "AAPL", "MCX"
        m = re.search(r'\b([A-Z]{2,6})\b', q.upper())
        if m:
            return m.group(1)
        return None

    async def _fetch_stock(self, ticker: str) -> Optional[dict]:
        """Fetch live stock data via yfinance in executor."""
        try:
            import yfinance as _yf
            loop = asyncio.get_event_loop()
            def _get():
                t = _yf.Ticker(ticker)
                info = t.info or {}
                price = (info.get("currentPrice") or info.get("regularMarketPrice")
                         or info.get("previousClose") or 0)
                hist = t.history(period="5d", interval="1d")
                ch = 0.0
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    curr = float(hist["Close"].iloc[-1])
                    ch = round((curr - prev) / prev * 100, 2) if prev else 0.0
                return {
                    "ok": True, "ticker": ticker,
                    "name": info.get("longName") or info.get("shortName") or ticker,
                    "price": round(float(price), 2),
                    "currency": info.get("currency", "INR"),
                    "change_pct": ch,
                    "pe_ratio": round(float(info.get("trailingPE") or 0), 1) or "—",
                    "52w_high": round(float(info.get("fiftyTwoWeekHigh") or 0), 2) or "—",
                    "52w_low":  round(float(info.get("fiftyTwoWeekLow") or 0), 2) or "—",
                    "sector": info.get("sector", "—"),
                }
            return await loop.run_in_executor(None, _get)
        except Exception:
            return None

    # ── Agent function factory ────────────────────────────────────────────────

    def _get_agent_fn(
        self,
        name:       str,
        query:      str,
        city:       str,
        intent:     dict,
        query_hash: str,
        wave:       int,
        peer_ctx:   str,
    ):
        """
        Return a synchronous callable that executes the named agent.
        peer_ctx is injected into the LLM system prompt for wave-2 agents.
        After completion, the result is written to SynapticState automatically.
        """
        aria     = self.aria
        engine   = self.engine
        memory   = self.memory
        state    = self.state
        bus      = self.bus
        fast_mdl = self._fast_model
        deep_mdl = self._deep_model

        def _write_result(agent_name: str, content: str, confidence: float):
            """Write agent result to shared workspace."""
            if state and content:
                state.write(
                    key=f"result:{agent_name}:{query_hash}",
                    value=content, author=agent_name,
                    confidence=confidence, signal_type=SignalType.RESULT,
                    ttl=35.0,
                )

        def _peer_sys(base_system: str) -> str:
            """Append peer context to system prompt if available."""
            if peer_ctx:
                return base_system + f"\n\n{peer_ctx}"
            return base_system

        # ── A1: Fast Reasoner ─────────────────────────────────────────────────
        if name == "fast_reasoner":
            def fast_reasoner():
                try:
                    _today = datetime.now().strftime("%A, %B %d, %Y")
                    lang_hint = ("\nRespond in Hindi." if intent.get("lang") == "hi"
                                 else "\nRespond in Telugu." if intent.get("lang") == "te" else "")
                    # aria-custom is ChatML-based (phi3) fine-tuned on Q:A pairs.
                    # Fix: prime the answer by appending "ARIA:" so the model completes
                    # from mid-answer rather than starting a new Q:A cycle.
                    system = (
                        f"You are ARIA, a powerful personal AI assistant. Today is {_today}. "
                        "You are answering as ARIA. Give a direct, complete, helpful response. "
                        "Use markdown with headings and bullet points. "
                        "NEVER start with 'Q:' or 'A:'. NEVER repeat the question."
                        f"{lang_hint}"
                    )
                    # Use fast_mdl (llama3.2:3b or best available reasoning model).
                    # llama3.2:3b is ~2GB — fits in available RAM even with backend running.
                    # It follows system prompts correctly, unlike aria-custom (exam Q:A trained).
                    result = engine.generate(
                        query, system=system, temperature=0.4,
                        model=fast_mdl, max_tokens=300, use_cache=False
                    )
                    # Discard Q: echo artifacts (shouldn't happen with llama3.2 but guard anyway)
                    if result and re.match(r"^\s*Q\s*:", result.strip(), re.IGNORECASE):
                        result = ""
                    _write_result(name, result, 0.75)
                    return AgentResult(name, result or "", 0.75, agent_type="text")
                except Exception:
                    return None
            return fast_reasoner

        # ── A2: Chain Reasoner ────────────────────────────────────────────────
        if name == "chain_reasoner":
            def chain_reasoner():
                try:
                    system = _peer_sys(
                        "I am ARIA. Think step by step, then give final answer. "
                        "Use markdown. Incorporate any peer findings listed below."
                    )
                    result = engine.generate(
                        f"Think through this carefully and answer in detail:\n{query}\n\nAnswer:",
                        system=system, temperature=0.3, model=fast_mdl
                    )
                    conf = 0.80
                    _write_result(name, result, conf)
                    return AgentResult(name, result or "", conf, agent_type="text")
                except Exception:
                    return None
            return chain_reasoner

        # ── A3: NOVA Reasoner ─────────────────────────────────────────────────
        if name == "nova_reasoner":
            def nova_reasoner():
                try:
                    nova = aria.get("nova")
                    if nova and hasattr(nova, "reason"):
                        res = nova.reason(query, n_simulations=2)
                        answer = res.get("answer", "") if isinstance(res, dict) else str(res)
                        conf   = res.get("confidence", 0.82) if isinstance(res, dict) else 0.82
                        _write_result(name, answer, conf)
                        return AgentResult(name, answer, conf, agent_type="text")
                    # Fallback: deep model
                    system = _peer_sys(
                        "I am ARIA, expert reasoning AI. Use multi-step reasoning with "
                        "self-verification. Incorporate peer findings. Be thorough."
                    )
                    result = engine.generate(query, system=system, temperature=0.2,
                                             model=deep_mdl)
                    _write_result(name, result, 0.82)
                    return AgentResult(name, result or "", 0.82, agent_type="text")
                except Exception:
                    return None
            return nova_reasoner

        # ── A3c: Academic Solver ──────────────────────────────────────────────
        if name == "academic_solver":
            def academic_solver():
                try:
                    from agents.academic_solver import academic_agent_fn
                    return academic_agent_fn(query, engine=engine, peer_ctx=peer_ctx)
                except Exception:
                    return None
            return academic_solver

        # ── A3d: Quantum Reasoner — superposition + amplitude amplification ──
        if name == "quantum_reasoner":
            def quantum_reasoner():
                try:
                    from agents.quantum_reasoner import quantum_agent_fn
                    return quantum_agent_fn(query, engine=engine, peer_ctx=peer_ctx)
                except Exception:
                    return None
            return quantum_reasoner

        # ── A3b: Physics Solver ───────────────────────────────────────────────
        if name == "physics_solver":
            def physics_solver():
                try:
                    from agents.physics_solver import physics_agent_fn
                    return physics_agent_fn(query, engine=engine, peer_ctx=peer_ctx)
                except Exception:
                    return None
            return physics_solver

        # ── A4: Symbolic Executor ─────────────────────────────────────────────
        if name == "symbolic_executor":
            def symbolic_executor():
                try:
                    nova = aria.get("nova")
                    if nova and hasattr(nova, "symbolic"):
                        res = nova.symbolic.execute(query)
                        if res and res.get("result") is not None:
                            text = f"**Exact result:** `{res['result']}`"
                            if res.get("steps"):
                                text += "\n\n**Steps:**\n" + "\n".join(
                                    f"{i+1}. {s}" for i, s in enumerate(res["steps"][:5])
                                )
                            _write_result(name, text, 1.0)
                            return AgentResult(name, text, 1.0, agent_type="data")
                except Exception:
                    pass
                return None
            return symbolic_executor

        # ── B1: Memory Retriever ──────────────────────────────────────────────
        if name == "memory_retriever":
            def memory_retriever():
                try:
                    if not memory:
                        return None
                    results = memory.search(query, top_k=5)
                    if not results:
                        return None
                    snippets = []
                    sources  = []
                    for r in results:
                        text = r.get("text", r.get("document", ""))[:200]
                        src  = r.get("metadata", {}).get("source", "memory")
                        if text:
                            snippets.append(text)
                            sources.append(src)
                    if not snippets:
                        return None
                    content = "\n\n".join(snippets[:3])
                    _write_result(name, content, 0.75)
                    return AgentResult(name, content, 0.75, sources=sources, agent_type="data")
                except Exception:
                    return None
            return memory_retriever

        # ── B2: Web Searcher ──────────────────────────────────────────────────
        if name == "web_searcher":
            def web_searcher():
                try:
                    search_q = query
                    if city and intent.get("is_live") and "weather" in query.lower():
                        search_q = f"{query} in {city}"

                    sources  = []
                    snippets = []

                    # ── Primary: ResearchSearchEngine (multi-source) ───────
                    try:
                        from agents.research_search_engine import ResearchSearchEngine
                        rse = ResearchSearchEngine()
                        rse_results = rse.search(search_q, max_results=5)
                        for r in rse_results:
                            text = (r.abstract or r.title or "")[:280]
                            if text:
                                snippets.append(f"**{r.title}** [{r.source}]: {text}")
                                sources.append({"title": r.title, "url": r.url, "source": r.source})
                    except Exception:
                        pass

                    # ── Fallback: Bing scrape ──────────────────────────────
                    if not snippets:
                        try:
                            import requests as _req
                            hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                            resp = _req.get("https://www.bing.com/search",
                                            params={"q": search_q, "count": 5},
                                            headers=hdrs, timeout=7)
                            import re as _re
                            titles = _re.findall(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', resp.text)
                            snips  = _re.findall(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', resp.text)
                            for i, (url, title) in enumerate(titles[:5]):
                                if url.startswith("http"):
                                    snip = _re.sub(r'<[^>]+>', '', snips[i] if i < len(snips) else "")[:250]
                                    clean_title = _re.sub(r'<[^>]+>', '', title).strip()
                                    if clean_title:
                                        snippets.append(f"**{clean_title}**: {snip}")
                                        sources.append({"title": clean_title, "url": url, "source": "bing"})
                        except Exception:
                            pass

                    # ── Last resort: DDG ───────────────────────────────────
                    if not snippets:
                        try:
                            from ddgs import DDGS
                            with DDGS() as ddg:
                                results = list(ddg.text(search_q, max_results=4))
                        except Exception:
                            try:
                                from duckduckgo_search import DDGS
                                with DDGS() as ddg:
                                    results = list(ddg.text(search_q, max_results=4))
                            except Exception:
                                results = []
                        for r in results:
                            title   = r.get("title", "")
                            snippet = r.get("body", r.get("snippet", ""))
                            url     = r.get("href",  r.get("url", ""))
                            if snippet:
                                snippets.append(f"**{title}**: {snippet}")
                                sources.append({"title": title, "url": url, "source": "duckduckgo"})

                    if not snippets:
                        return None
                    content = "\n\n".join(snippets[:4])[:800]
                    _write_result(name, content, 0.82)
                    return AgentResult(name, content, 0.82,
                                       sources=sources, agent_type="search")
                except Exception:
                    return None
            return web_searcher

        # ── B3: Scientific Researcher ─────────────────────────────────────────
        if name == "sci_researcher":
            def sci_researcher():
                try:
                    researcher = aria.get("research")
                    if researcher and hasattr(researcher, "search"):
                        res     = researcher.search(query, max_results=3)
                        papers  = res if isinstance(res, list) else []
                        snippets = []
                        sources  = []
                        for p in papers[:3]:
                            title    = p.get("title", "")
                            abstract = p.get("abstract", p.get("summary", ""))[:200]
                            url      = p.get("url", p.get("link", ""))
                            if abstract:
                                snippets.append(f"**{title}**: {abstract}")
                                sources.append({"title": title, "url": url})
                        content = "\n\n".join(snippets)
                        _write_result(name, content, 0.85)
                        return AgentResult(name, content, 0.85,
                                           sources=sources, agent_type="search")
                except Exception:
                    pass
                # Fallback: web search for academic content
                try:
                    system = _peer_sys(
                        "I am ARIA research assistant. Find and explain relevant research."
                    )
                    result = engine.generate(
                        f"Research findings on: {query}",
                        system=system, temperature=0.3, model=deep_mdl
                    )
                    _write_result(name, result, 0.75)
                    return AgentResult(name, result or "", 0.75, agent_type="text")
                except Exception:
                    return None
            return sci_researcher

        # ── B4: World Model Lookup ────────────────────────────────────────────
        if name == "world_model_lookup":
            def world_model_lookup():
                try:
                    world = aria.get("world")
                    if not world or not hasattr(world, "query"):
                        return None
                    nouns = [w for w in query.split() if w[0].isupper() and len(w) > 2]
                    if not nouns:
                        return None
                    facts = []
                    for noun in nouns[:3]:
                        res = world.query(noun)
                        if res:
                            facts.extend(res[:2])
                    if not facts:
                        return None
                    content = "\n".join(str(f) for f in facts[:5])
                    _write_result(name, content, 0.90)
                    return AgentResult(name, content, 0.90, agent_type="data")
                except Exception:
                    return None
            return world_model_lookup

        # ── C1: Code Specialist ───────────────────────────────────────────────
        if name == "code_specialist":
            def code_specialist():
                try:
                    code_eng = aria.get("code_engine")
                    if code_eng and hasattr(code_eng, "generate"):
                        res  = code_eng.generate(query)
                        code = res.get("code", "") if isinstance(res, dict) else str(res)
                        if code:
                            content = f"```python\n{code}\n```"
                            _write_result(name, content, 0.85)
                            return AgentResult(name, content, 0.85, agent_type="code")
                    system = _peer_sys(
                        "I am ARIA expert programmer. Write clean, working code with "
                        "proper code blocks and brief comments."
                    )
                    result = engine.generate(query, system=system, temperature=0.2,
                                             model=deep_mdl)
                    _write_result(name, result, 0.80)
                    return AgentResult(name, result or "", 0.80, agent_type="code")
                except Exception:
                    return None
            return code_specialist

        # ── C2: Planner Agent ─────────────────────────────────────────────────
        if name == "planner_agent":
            def planner_agent():
                try:
                    planner = aria.get("planner")
                    if planner and hasattr(planner, "decompose"):
                        plan  = planner.decompose(query)
                        steps = plan.get("steps", []) if isinstance(plan, dict) else []
                        if steps:
                            lines = [
                                f"{i+1}. {s.get('action', s) if isinstance(s, dict) else s}"
                                for i, s in enumerate(steps[:8])
                            ]
                            content = "**Plan:**\n" + "\n".join(lines)
                            _write_result(name, content, 0.80)
                            return AgentResult(name, content, 0.80, agent_type="plan")
                    system = _peer_sys(
                        "I am ARIA. Break this into clear numbered steps. "
                        "Incorporate peer knowledge listed below."
                    )
                    result = engine.generate(f"Create step-by-step plan: {query}",
                                             system=system, temperature=0.3,
                                             model=fast_mdl)
                    _write_result(name, result, 0.75)
                    return AgentResult(name, result or "", 0.75, agent_type="plan")
                except Exception:
                    return None
            return planner_agent

        # ── C3: Trend Watcher ─────────────────────────────────────────────────
        if name == "trend_watcher":
            def trend_watcher():
                try:
                    scanner = aria.get("scanner")
                    if scanner and hasattr(scanner, "scan"):
                        trends  = scanner.scan(query)
                        content = "\n".join(
                            str(t) for t in
                            (trends if isinstance(trends, list) else [trends])[:3]
                        )
                        if content:
                            _write_result(name, content, 0.75)
                            return AgentResult(name, content, 0.75, agent_type="data")
                except Exception:
                    pass
                return None
            return trend_watcher

        # ── C4: Calendar Context ──────────────────────────────────────────────
        if name == "calendar_context":
            def calendar_context():
                try:
                    cal = aria.get("calendar")
                    if cal and hasattr(cal, "natural_query"):
                        res = cal.natural_query(query)
                        if res:
                            content = str(res)
                            _write_result(name, content, 0.90)
                            return AgentResult(name, content, 0.90, agent_type="data")
                except Exception:
                    pass
                return None
            return calendar_context

        # ── C5: Summarizer ────────────────────────────────────────────────────
        if name == "summarizer_agent":
            def summarizer_agent():
                try:
                    urls = re.findall(r'https?://\S+', query)
                    if urls:
                        crawler = aria.get("crawler")
                        if crawler and hasattr(crawler, "crawl"):
                            content_raw = crawler.crawl(urls[0], max_pages=1)
                            if content_raw:
                                system = _peer_sys(
                                    "I am ARIA. Summarise in 3-5 bullet points. "
                                    "Focus on key facts. Use peer context to add depth."
                                )
                                result = engine.generate(
                                    f"Summarise:\n{str(content_raw)[:2000]}",
                                    system=system, temperature=0.3, model=fast_mdl
                                )
                                _write_result(name, result, 0.80)
                                return AgentResult(name, result or "", 0.80, agent_type="text")
                    system = _peer_sys("I am ARIA. Provide a clear, structured summary.")
                    result = engine.generate(query, system=system, temperature=0.3,
                                             model=fast_mdl)
                    _write_result(name, result, 0.70)
                    return AgentResult(name, result or "", 0.70, agent_type="text")
                except Exception:
                    return None
            return summarizer_agent

        # ── D: Desktop Controller ─────────────────────────────────────────────
        if name == "desktop_controller":
            def desktop_controller():
                try:
                    desktop = aria.get("desktop")
                    if not desktop or not hasattr(desktop, "execute_nl"):
                        return None
                    result = desktop.execute_nl(query)
                    if result and result.get("ok"):
                        content = result.get("result", "")
                        _write_result(name, content, 0.95)
                        return AgentResult(name, content, 0.95, agent_type="text")
                    elif result:
                        content = f"Desktop: {result.get('result', '')}"
                        _write_result(name, content, 0.50)
                        return AgentResult(name, content, 0.50, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"Desktop error: {e}", 0.1)
                return None
            return desktop_controller

        # ── E1: Browser Controller ────────────────────────────────────────────
        if name == "browser_controller":
            def browser_controller():
                try:
                    browser = aria.get("browser")
                    if not browser or not hasattr(browser, "run_nl"):
                        return None
                    result = browser.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:600]
                        _write_result(name, content, 0.90)
                        return AgentResult(name, content, 0.90, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"Browser: {e}", 0.1)
                return None
            return browser_controller

        # ── E2: System Controller ─────────────────────────────────────────────
        if name == "system_controller":
            def system_controller():
                try:
                    sys_ag = aria.get("sys_agent")
                    if not sys_ag or not hasattr(sys_ag, "run_nl"):
                        return None
                    result = sys_ag.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:600]
                        _write_result(name, content, 0.90)
                        return AgentResult(name, content, 0.90, agent_type="data")
                except Exception as e:
                    return AgentResult(name, f"System: {e}", 0.1)
                return None
            return system_controller

        # ── E3: Document Reader ───────────────────────────────────────────────
        if name == "document_reader":
            def document_reader():
                try:
                    doc_ag = aria.get("doc_agent")
                    if not doc_ag or not hasattr(doc_ag, "run_nl"):
                        return None
                    result = doc_ag.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:800]
                        _write_result(name, content, 0.85)
                        return AgentResult(name, content, 0.85, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"Document: {e}", 0.1)
                return None
            return document_reader

        # ── E4: Code Runner ───────────────────────────────────────────────────
        if name == "code_runner":
            def code_runner():
                try:
                    exec_ag = aria.get("code_exec")
                    if not exec_ag or not hasattr(exec_ag, "run_nl"):
                        return None
                    result = exec_ag.run_nl(query)
                    if result:
                        output  = result.get("output", result.get("result", str(result)))[:600]
                        lang    = result.get("lang", "python")
                        content = f"**Executed ({lang}):**\n```\n{output}\n```"
                        _write_result(name, content, 0.90)
                        return AgentResult(name, content, 0.90, agent_type="code")
                except Exception as e:
                    return AgentResult(name, f"Exec error: {e}", 0.1)
                return None
            return code_runner

        # ── E5: Media Controller ──────────────────────────────────────────────
        if name == "media_controller":
            def media_controller():
                try:
                    media = aria.get("media")
                    if not media or not hasattr(media, "run_nl"):
                        return None
                    result = media.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:300]
                        _write_result(name, content, 0.95)
                        return AgentResult(name, content, 0.95, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"Media: {e}", 0.1)
                return None
            return media_controller

        # ── E6: Network Inspector ─────────────────────────────────────────────
        if name == "network_inspector":
            def network_inspector():
                try:
                    net = aria.get("network")
                    if not net or not hasattr(net, "run_nl"):
                        return None
                    result = net.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:600]
                        _write_result(name, content, 0.90)
                        return AgentResult(name, content, 0.90, agent_type="data")
                except Exception as e:
                    return AgentResult(name, f"Network: {e}", 0.1)
                return None
            return network_inspector

        # ── E7: Automation Controller ─────────────────────────────────────────
        if name == "automation_controller":
            def automation_controller():
                try:
                    auto = aria.get("automation")
                    if not auto or not hasattr(auto, "run_nl"):
                        return None
                    result = auto.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:400]
                        _write_result(name, content, 0.85)
                        return AgentResult(name, content, 0.85, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"Automation: {e}", 0.1)
                return None
            return automation_controller

        # ── F1: OS Profiler ───────────────────────────────────────────────────
        if name == "os_profiler":
            def os_profiler():
                try:
                    det = aria.get("os_detector")
                    if not det or not hasattr(det, "detect"):
                        return None
                    profile = det.detect()
                    content = (
                        f"**OS:** {profile.os_name} {profile.os_version} | "
                        f"**Kernel:** {getattr(profile,'kernel_version','unknown')} | "
                        f"**Shell:** {profile.shell} | "
                        f"**Admin:** {profile.is_admin} | "
                        f"**RAM:** {getattr(profile,'ram_available_gb',0):.1f}GB free | "
                        f"**Arch:** {getattr(profile,'cpu_arch','unknown')}"
                    )
                    if getattr(profile, "is_wsl", False):
                        content += " | **WSL:** Yes"
                    _write_result(name, content, 0.95)
                    return AgentResult(name, content, 0.95, agent_type="data")
                except Exception as e:
                    return AgentResult(name, f"OS profile error: {e}", 0.1)
            return os_profiler

        # ── F2: Threat Detector ───────────────────────────────────────────────
        if name == "threat_detector":
            def threat_detector():
                try:
                    sec = aria.get("sec_monitor")
                    if not sec or not hasattr(sec, "scan_processes"):
                        return None
                    threats = sec.scan_processes()
                    if not threats:
                        content = "✅ No suspicious processes detected."
                    else:
                        lines = [f"⚠️ **{t.name}** (PID {t.pid}): {t.reason}"
                                 for t in threats[:5] if hasattr(t,'name')]
                        content = "**Security Scan:**\n" + "\n".join(lines)
                    _write_result(name, content, 0.90)
                    return AgentResult(name, content, 0.90, agent_type="data")
                except Exception as e:
                    return AgentResult(name, f"Security scan: {e}", 0.1)
            return threat_detector

        # ── F3: Activity Context ──────────────────────────────────────────────
        if name == "activity_context":
            def activity_context():
                try:
                    trainer = aria.get("activity")
                    if not trainer or not hasattr(trainer, "build_user_profile"):
                        return None
                    profile = trainer.build_user_profile()
                    if not profile:
                        return None
                    content = (
                        f"**User profile:** style={getattr(profile,'communication_style','unknown')} | "
                        f"expertise={getattr(profile,'expertise_level',{}) if isinstance(getattr(profile,'expertise_level',None),str) else 'varied'} | "
                        f"topics={', '.join(getattr(profile,'preferred_topics',[])[:3])}"
                    )
                    _write_result(name, content, 0.70)
                    return AgentResult(name, content, 0.70, agent_type="data")
                except Exception:
                    return None
            return activity_context

        # ── F4: CoT Thinker ───────────────────────────────────────────────────
        if name == "cot_thinker":
            def cot_thinker():
                try:
                    cot = aria.get("cot")
                    if cot and hasattr(cot, "think"):
                        chain = cot.think(query, strategy="human_like")
                        if chain and hasattr(chain, "final_answer"):
                            content = chain.final_answer
                            conf    = getattr(chain, "confidence", 0.82)
                            _write_result(name, content, conf)
                            return AgentResult(name, content, conf, agent_type="text")
                    # Fallback: LLM with CoT prompt
                    system = _peer_sys(
                        "I am ARIA. Think like a human expert:\n"
                        "1) What do I know?\n2) What's the key issue?\n"
                        "3) Step-by-step reasoning\n4) Verify my conclusion\n\n"
                        "Incorporate any peer context listed below."
                    )
                    result = engine.generate(
                        f"Think carefully and answer: {query}",
                        system=system, temperature=0.3, model=deep_mdl
                    )
                    _write_result(name, result, 0.82)
                    return AgentResult(name, result or "", 0.82, agent_type="text")
                except Exception:
                    return None
            return cot_thinker

        # ── F5: Terminal Runner ───────────────────────────────────────────────
        if name == "terminal_runner":
            def terminal_runner():
                try:
                    term = aria.get("terminal")
                    if not term or not hasattr(term, "run_nl"):
                        return None
                    result = term.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:800]
                        ok      = result.get("ok", True)
                        conf    = 0.92 if ok else 0.40
                        _write_result(name, content, conf)
                        return AgentResult(name, content, conf, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"Terminal: {e}", 0.1)
                return None
            return terminal_runner

        # ── F6: Web Builder (read URL → implement) ────────────────────────────
        if name == "web_builder":
            def web_builder():
                try:
                    impl = aria.get("web_impl")
                    if not impl or not hasattr(impl, "run_nl"):
                        return None
                    result = impl.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:800]
                        conf    = 0.85 if result.get("ok") else 0.50
                        _write_result(name, content, conf)
                        return AgentResult(name, content, conf, agent_type="code")
                except Exception as e:
                    return AgentResult(name, f"Web builder: {e}", 0.1)
                return None
            return web_builder

        # ── F7: App Auditor (test + limitations + fix) ────────────────────────
        if name == "app_auditor":
            def app_auditor():
                try:
                    tester = aria.get("app_tester")
                    if not tester or not hasattr(tester, "run_nl"):
                        return None
                    result = tester.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:800]
                        _write_result(name, content, 0.88)
                        return AgentResult(name, content, 0.88, agent_type="text")
                except Exception as e:
                    return AgentResult(name, f"App audit: {e}", 0.1)
                return None
            return app_auditor

        # ── F8: Auto Optimizer (self-tune) ────────────────────────────────────
        if name == "auto_optimizer":
            def auto_optimizer():
                try:
                    tuner = aria.get("auto_tuner")
                    if not tuner or not hasattr(tuner, "run_nl"):
                        return None
                    result = tuner.run_nl(query)
                    if result:
                        content = result.get("result", str(result))[:400]
                        _write_result(name, content, 0.80)
                        return AgentResult(name, content, 0.80, agent_type="data")
                except Exception as e:
                    return AgentResult(name, f"Auto-tune: {e}", 0.1)
                return None
            return auto_optimizer

        # ── F9: Scam Scanner (phishing / fake-site detection) ─────────────────
        # ── F10: Language Detector + Multi-Language Responder ────────────────
        if name == "lang_detector":
            def lang_detector():
                try:
                    trust_lang = aria.get("trust_language")
                    if not trust_lang or not hasattr(trust_lang, "run_nl"):
                        return None
                    result = trust_lang.run_nl(query)
                    if not result:
                        return None
                    # "Sources for this query:" is a TrustLanguageAgent fallback for English
                    # queries it can't handle — not a useful answer, skip it.
                    if result.strip().startswith("Sources for this query"):
                        return None
                    _write_result(name, result, 0.78)
                    return AgentResult(name, result, 0.78, agent_type="text")
                except Exception:
                    return None
            return lang_detector

        # ── F11: Stock Analyst — 12-layer QuantumStockAgent ──────────────────
        if name == "stock_analyst":
            def stock_analyst():
                try:
                    q_stock = aria.get("quantum_stock")
                    if not q_stock or not hasattr(q_stock, "run_nl"):
                        return None
                    result = q_stock.run_nl(query)
                    if not result:
                        return None
                    _write_result(name, result, 0.88)
                    return AgentResult(name, result, 0.88, agent_type="data")
                except Exception:
                    return None
            return stock_analyst

        if name == "scam_scanner":
            def scam_scanner():
                try:
                    import re as _re
                    url_m = _re.search(
                        r"(https?://[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)",
                        query,
                    )
                    if not url_m:
                        return None     # no URL in query — skip
                    url = url_m.group(1)
                    from agents.scam_detector import ScamDetectorAgent
                    detector = ScamDetectorAgent(engine=self.engine)
                    report   = detector.scan(url)
                    # Write verdict to workspace so other agents see it
                    confidence = report.trust_score / 100.0
                    _write_result(name, report.summary(), confidence)
                    return AgentResult(name, report.summary(), confidence, agent_type="data")
                except Exception as e:
                    return AgentResult(name, f"Scam scan error: {e}", 0.1)
            return scam_scanner

        # ── F12: Intent Classifier — zero-LLM action gate ────────────────────
        if name == "intent_classifier":
            def intent_classifier():
                try:
                    from agents.auto_executor import classify_action_fast
                    action_type, risk = classify_action_fast(query)
                    result = json.dumps({
                        "action_type": action_type,
                        "risk":        risk,
                        "is_action":   action_type not in ("question",),
                    })
                    _write_result(name, result, 0.95)
                    return AgentResult(name, result, 0.95, agent_type="intent")
                except Exception:
                    return None
            return intent_classifier

        # ── F13: Undo Tracker — context for undo/repeat operations ───────────
        if name == "undo_tracker":
            def undo_tracker():
                try:
                    conv = aria.get("conversation")
                    if not conv or not hasattr(conv, "get_or_create_session"):
                        return None
                    session = conv.get_or_create_session("default")
                    ctx = f"Last action: {session.last_action or 'none'}"
                    if session.last_args:
                        ctx += f" | Args: {list(session.last_args.items())[:3]}"
                    if session.undo_stack:
                        ctx += f" | Undo available (depth={len(session.undo_stack)})"
                    _write_result(name, ctx, 0.88)
                    return AgentResult(name, ctx, 0.88, agent_type="context")
                except Exception:
                    return None
            return undo_tracker

        if name == "sentiment_analyst":
            def sentiment_analyst():
                try:
                    agent = aria.get("sentiment_agent")
                    if not agent or not hasattr(agent, "analyze"):
                        return None
                    import asyncio as _aio
                    loop = _aio.new_event_loop()
                    analysis = loop.run_until_complete(
                        agent.analyze(query, raw_context=query)
                    )
                    loop.close()
                    # Write structured result to SynapticState
                    result_text = (
                        f"AMIA Score: {analysis.total_score:+.1f} | "
                        f"Verdict: {analysis.verdict} | "
                        f"Confidence: {analysis.confidence:.0f}% | "
                        f"Mood: {analysis.psychology_mood} | "
                        f"Horizon: {analysis.time_horizon}"
                    )
                    if analysis.operator_alert:
                        result_text += f" | OPERATOR: {analysis.operator_alert}"
                    _write_result(name, result_text, analysis.confidence / 100)
                    return AgentResult(name, result_text, analysis.confidence / 100,
                                       agent_type="market_intel")
                except Exception as _e:
                    return None
            return sentiment_analyst

        # ── F16: Knowledge Verifier — ground response in verified facts ──────────
        if name == "knowledge_verifier":
            def knowledge_verifier():
                try:
                    ke = aria.get("knowledge_engine")
                    if not ke or not hasattr(ke, "answer_grounded"):
                        return None
                    result = ke.answer_grounded(query)
                    if not result or not result.get("grounded"):
                        return None
                    content = result.get("answer", "")[:600]
                    conf    = result.get("confidence", 0.5)
                    if conf < 0.3:
                        return None
                    _write_result(name, content, conf)
                    return AgentResult(name, content, conf, agent_type="text")
                except Exception:
                    return None
            return knowledge_verifier

        # ── F17: Environment Context — what apps/projects the user has ───────────
        if name == "env_context":
            def env_context():
                try:
                    el = aria.get("env_learner")
                    if not el or not hasattr(el, "get_context_for_query"):
                        return None
                    ctx = el.get_context_for_query(query)
                    if not ctx:
                        return None
                    _write_result(name, ctx[:400], 0.75)
                    return AgentResult(name, ctx[:400], 0.75, agent_type="context")
                except Exception:
                    return None
            return env_context

        # ── F15: Windows Kernel Agent — Win+R/PS/CMD/Git/Chrome/Java/Code ───────
        if name == "win_kernel_agent":
            def win_kernel_agent():
                try:
                    wk = aria.get("win_kernel")
                    if not wk or not hasattr(wk, "run_nl"):
                        return None
                    import asyncio as _aio
                    loop = _aio.new_event_loop()
                    result = loop.run_until_complete(wk.run_nl(query))
                    loop.close()
                    if not result or not result.get("ok"):
                        return None
                    content = result.get("message") or result.get("output") or str(result)[:600]
                    _write_result(name, content, 0.92)
                    return AgentResult(name, content, 0.92, agent_type="text")
                except Exception as _e:
                    return None
            return win_kernel_agent

        # Default: no-op
        def noop():
            return None
        return noop

    # ── Synthesis ─────────────────────────────────────────────────────────────

    async def _synthesize(
        self,
        query:   str,
        intent:  dict,
        results: list[AgentResult],
    ) -> str:
        """
        LLM synthesis of top-K results.
        Upgrade B: uses DecisionEngine.weighted_vote on agent confidences.
        Upgrade C: tracks surprise to boost agent weights post-synthesis.
        Wave-2 results already have peer context baked in, so they're highest value.
        """
        if not results or not self.engine:
            return ""

        # ── Upgrade B: confidence-weighted vote to pick primary result ────────
        if _ALGO_CORE:
            vote_options = [
                (r.agent, self._agent_confidence.get(r.agent, 0.5) * r.confidence)
                for r in results if r.content and r.agent_type in ("text", "plan", "code")
            ]
            if vote_options:
                winner_agent, vote_conf = DecisionEngine.weighted_vote(vote_options)
            else:
                winner_agent = ""
        else:
            winner_agent = ""

        # Prefer wave-2 and high-confidence results; put voted winner first
        sorted_results = sorted(results, key=lambda r: (
            r.agent == winner_agent,
            r.confidence
        ), reverse=True)
        top = sorted_results[:6]

        def _is_junk(content: str) -> bool:
            """Detect raw data structures / OS dumps / metadata not suitable as user-facing answers."""
            s = content.strip()
            # Raw Python dict from calendar/planner agents (e.g. {'events': [], 'answer': '...'})
            if s.startswith("{") and s.endswith("}") and ":" in s:
                return True
            # Process list from security/OS profiler agents
            if "PID" in s and any(x in s for x in ("Normal process", "svchost", "System Idle", "High CPU")):
                return True
            # Activity/user profile dump from activity_context agent
            if s.startswith("User profile:") or ("style=" in s and "expertise=" in s and "topics=" in s):
                return True
            # Neural wave status lines leaked as content
            _status_prefixes = ("Neural network activating", "Wave 1", "Wave 2",
                                 "Propagating neural", "Synthesizing neural", "fast agents firing")
            if any(s.startswith(p) for p in _status_prefixes):
                return True
            # Schedule-dump for non-calendar query
            if "Your schedule is clear" in s or "events: []" in s:
                _cal_kw = {"schedule", "calendar", "meeting", "appointment", "event", "today's plan"}
                if not any(k in query.lower() for k in _cal_kw):
                    return True
            # Model echoed Q: format — aria-custom training artifact
            if re.match(r"^\s*Q\s*:", s, re.IGNORECASE):
                return True
            # Extremely short/empty response
            if len(s) < 8:
                return True
            return False

        # For capability/greeting queries, web results are always off-topic
        _aria_self_query = any(k in query.lower() for k in (
            "what can you do", "what are you", "who are you", "your capabilities",
            "what do you do", "introduce yourself", "tell me about yourself",
            "what features", "your features", "help me understand you",
        ))

        pieces = []
        _seen_content: list[str] = []  # dedup tracker
        for r in top:
            if r.content and r.agent_type in ("text", "plan", "code"):
                if _is_junk(r.content):
                    continue
                # Skip web_searcher for ARIA self-description queries — web results are irrelevant
                if _aria_self_query and r.agent == "web_searcher":
                    continue
                # Deduplicate: skip if >70% overlap with an already-added piece
                _short = r.content[:200].strip()
                if any(_short[:80] in s or s[:80] in _short for s in _seen_content):
                    continue
                _seen_content.append(_short)
                marker = " [VOTED PRIMARY]" if r.agent == winner_agent else ""
                pieces.append(f"[{r.agent}{marker}]:\n{r.content[:600]}")
        if not pieces:
            # Never surface raw intent/context metadata as the answer
            text_results = [r for r in sorted_results
                            if r.agent_type in ("text", "plan", "code", "data")
                            and not _is_junk(r.content or "")]
            return text_results[0].content if text_results else ""

        # For ARIA self-description queries: fast_reasoner already answered with primed Q:A prompt.
        # Skip synthesis (would re-generate via aria-custom and produce off-topic content).
        if _aria_self_query:
            primary = next((r for r in sorted_results if r.agent == winner_agent and r.content), None)
            if primary and not _is_junk(primary.content):
                return primary.content
            # Fallback: return best piece content
            return pieces[0].split(":\n", 1)[-1].strip() if pieces else ""

        # ── Priority bypass: physics/academic solvers return exact computed answers ──
        # When SymPy / structured solver produces a result, return it directly —
        # LLM synthesis would only dilute or corrupt the precise step-by-step answer.
        _SOLVER_AGENTS = ("physics_solver", "academic_solver")
        for _solver in _SOLVER_AGENTS:
            _solver_result = next(
                (r for r in results
                 if r.agent == _solver
                 and r.content
                 and not _is_junk(r.content)
                 and len(r.content.strip()) > 40),
                None,
            )
            if _solver_result:
                return _solver_result.content

        _today = datetime.now().strftime("%A, %B %d, %Y")
        system = (
            f"You are ARIA, a helpful personal AI assistant. Today is {_today}. "
            "Your job is to answer the user's question accurately and completely. "
            "Use the [VOTED PRIMARY] agent result as your main source. "
            "Ignore any agent findings that are off-topic or not relevant to the question. "
            "Use markdown: headings, bullet points, bold text. Aim for 20-50 lines. "
            "IMPORTANT: Answer the question directly. Do NOT start with 'Q:' or 'A:'. "
            "Do NOT generate civic, political, or unrelated advice unless explicitly asked."
        )
        prompt = (
            f"User's question: {query}\n\n"
            f"Agent findings (use [VOTED PRIMARY] as primary source; ignore off-topic results):\n"
            + "\n\n---\n".join(pieces) +
            f"\n\nNow write a complete, well-structured answer to: {query}"
        )
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.engine.generate(
                        prompt, system=system, temperature=0.4,
                        model=self._fast_model,
                        use_cache=False,
                        max_tokens=512,
                    )
                ),
                timeout=45.0,
            )
            if not result:
                return sorted_results[0].content if sorted_results else ""

            # ── Upgrade C: surprise-driven weight update ──────────────────────
            if _ALGO_CORE:
                for r in results:
                    if not r.content:
                        continue
                    # How much did this agent reduce surprise vs final synthesis?
                    surprise = AdaptiveLearner.compute_surprise(result, r.content)
                    # Low surprise = agent was predictive = boost its confidence
                    contribution = 1.0 - surprise  # 0–1; higher = more useful
                    old_conf = self._agent_confidence.get(r.agent, 0.5)
                    # Soft Hebbian: use Oja's rule variant
                    new_conf = AdaptiveLearner.hebbian_update(
                        weight=old_conf,
                        pre_activation=contribution,
                        post_activation=r.confidence,
                        lr=0.02,
                    )
                    self._agent_confidence[r.agent] = new_conf

            # ── Ground the synthesized response through knowledge engine ─────
            try:
                ke = self.aria.get("knowledge_engine")
                if ke and hasattr(ke, "ground"):
                    grounded = ke.ground(result, query)
                    ke.queue_absorb(result, source="aria_synthesis", domain="conversation")
                    return grounded.response
            except Exception:
                pass
            return result
        except Exception:
            return sorted_results[0].content if sorted_results else ""

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _quick_score(self, text: str, results: list[AgentResult]) -> float:
        if not text or not results:
            return 0.5
        avg_conf = sum(r.confidence for r in results) / len(results)
        length_bonus = min(0.15, len(text) / 2000)
        return min(1.0, avg_conf + length_bonus)

    # ── Text chunking ─────────────────────────────────────────────────────────

    @staticmethod
    def _chunk(text: str, size: int = 60) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i:i + size]) + " ")
        return chunks or [text]

    # ── Background loop ────────────────────────────────────────────────────────

    async def _background_loop(self) -> None:
        cycle = 0
        while True:
            try:
                await asyncio.sleep(300)
                cycle += 1
                # Evict expired workspace entries
                self.state.evict_expired()
                # Proactive alerts
                proactive = self.aria.get("proactive")
                if proactive and hasattr(proactive, "check_all"):
                    loop = asyncio.get_event_loop()
                    await asyncio.wait_for(
                        loop.run_in_executor(None, proactive.check_all),
                        timeout=30.0
                    )
                # World refresh
                if cycle % 4 == 0:
                    world = self.aria.get("world")
                    if world and hasattr(world, "refresh_stale"):
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, world.refresh_stale)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    async def _background_post_query(
        self,
        query:         str,
        answer:        str,
        confidence:    float,
        wave1_results: list[AgentResult],
        wave2_results: list[AgentResult],
    ) -> None:
        """Hebbian updates + memory store (runs after stream() completes)."""
        loop = asyncio.get_event_loop()

        # Hebbian: reinforce wave1→wave2 pairs that both produced results
        w1_agents = {r.agent for r in wave1_results if r.content}
        w2_agents = {r.agent for r in wave2_results if r.content}
        for src in w1_agents:
            for tgt in w2_agents:
                self.state.reinforce(src, tgt)
                self.bus.reinforce_synapse(src, tgt)

        # Same-wave co-firing reinforcement
        for agents in [list(w1_agents), list(w2_agents)]:
            for i, src in enumerate(agents):
                for tgt in agents[i + 1:]:
                    self.state.reinforce(src, tgt)

        # Store to memory
        try:
            if self.memory and hasattr(self.memory, "store"):
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.memory.store(
                            text=f"Q: {query}\nA: {answer}",
                            metadata={"source": "neural_chat", "confidence": confidence}
                        )
                    ),
                    timeout=10.0,
                )
        except Exception:
            pass

        # ── Activity trainer: record interaction for personalization ───────────
        try:
            trainer = self.aria.get("activity")
            if trainer and hasattr(trainer, "record_interaction"):
                loop.run_in_executor(
                    None,
                    lambda: trainer.record_interaction(
                        query=query,
                        response=answer,
                        task_type=self._infer_task_type(query),
                    )
                )
        except Exception:
            pass

        # ── Auto tuner: record response for performance tracking ───────────────
        try:
            tuner = self.aria.get("auto_tuner")
            if tuner and hasattr(tuner, "record_response"):
                task_type = self._infer_task_type(query)
                loop.run_in_executor(
                    None,
                    lambda: tuner.record_response(
                        query=query, response=answer,
                        task_type=task_type,
                        model=self._fast_model,
                        temperature=0.4,
                        score=confidence,
                        latency_ms=int((time.time() - (time.time())) * 1000),
                    )
                )
        except Exception:
            pass

    def _infer_task_type(self, query: str) -> str:
        """Quick task type inference for activity + auto_tuner recording."""
        q = query.lower()
        if any(k in q for k in ["code","python","javascript","function","debug","script"]):
            return "code"
        if any(k in q for k in ["calculate","solve","math","equation","integral"]):
            return "math"
        if any(k in q for k in ["plan","steps","how to","workflow","roadmap"]):
            return "planning"
        if any(k in q for k in ["summarize","summary","tldr","brief","overview"]):
            return "summary"
        if any(k in q for k in ["terminal","run","execute","bash","shell","command"]):
            return "terminal"
        if any(k in q for k in ["threat","security","virus","scan","safe"]):
            return "security"
        if any(k in q for k in ["explain","what is","define","describe","tell me"]):
            return "factual"
        return "chat"

    async def _background_store(
        self, query: str, answer: str, confidence: float, mode: str
    ) -> None:
        try:
            loop = asyncio.get_event_loop()
            if self.memory and hasattr(self.memory, "store"):
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.memory.store(
                            text=f"Q: {query}\nA: {answer}",
                            metadata={"source": mode, "confidence": confidence}
                        )
                    ),
                    timeout=8.0,
                )
        except Exception:
            pass

    # ── Upgrade D: User feedback → agent confidence update ────────────────────

    def feedback_positive(self, agents_used: list[str]) -> None:
        """
        Call when user accepts/likes a response.
        Increases confidence for involved agents: +0.05 * (1 - current).
        """
        for agent in agents_used:
            c = self._agent_confidence.get(agent, 0.5)
            self._agent_confidence[agent] = min(1.0, c + 0.05 * (1.0 - c))

    def feedback_negative(self, agents_used: list[str]) -> None:
        """
        Call when user rejects/dislikes a response.
        Decreases confidence for involved agents: -0.03 * current.
        """
        for agent in agents_used:
            c = self._agent_confidence.get(agent, 0.5)
            self._agent_confidence[agent] = max(0.0, c - 0.03 * c)

    def get_agent_confidences(self) -> dict[str, float]:
        """Return copy of all current agent confidence scores."""
        return dict(self._agent_confidence)

    # ── Status ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "orchestrator": "NeuralOrchestrator",
            "bus":          self.bus.stats(),
            "workspace":    self.state.stats(),
            "weight_pairs": len(self.state.get_all_weights()),
            "synapse_count":len(self.bus.get_all_synapses()),
            "agent_confidences": self._agent_confidence,
        }
