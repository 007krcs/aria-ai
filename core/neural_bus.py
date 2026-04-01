"""
ARIA Neural Bus — Inter-agent signal propagation medium (v1)
============================================================

Every agent communicates through this shared medium, exactly like neurons
firing through synapses.  Signals carry a type, strength, source, target
(None = broadcast), and a TTL.  The bus maintains:

  • A rolling signal buffer (recent history visible to all)
  • Per-pair synapse strength  (strengthened by co-firing, decayed over time)
  • Suppression state          (INHIBITORY signals suppress target until TTL)
  • Excitation boosts          (EXCITATORY signals raise scheduling priority)
  • Peer context builder       (agents call this BEFORE generating LLM response)

Signal types:
    EXCITATORY — amplify peer, raise their scheduling priority
    INHIBITORY — stand down, you're not needed
    RESULT     — I produced output, sharing with everyone
    QUERY      — I need a specific peer to answer something
    CONTEXT    — sharing enrichment data (not a final result)
    REQUEST    — directed one-to-one question

No LLM or ML dependency — all similarity is word-overlap Jaccard.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL TYPES
# ─────────────────────────────────────────────────────────────────────────────

class SignalType(str, Enum):
    EXCITATORY = "EXCITATORY"   # amplify peer — raise scheduling priority
    INHIBITORY = "INHIBITORY"   # stand down   — suppress competing agent
    RESULT     = "RESULT"       # I have produced output
    QUERY      = "QUERY"        # I need a peer to answer something
    CONTEXT    = "CONTEXT"      # enrichment data for peers
    REQUEST    = "REQUEST"      # directed one-to-one request


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NeuralSignal:
    signal_type:  SignalType
    source_agent: str
    topic:        str                        # semantic domain / category
    payload:      dict                       # structured data (text, confidence, …)
    confidence:   float        = 0.5        # 0.0 – 1.0
    target_agent: Optional[str]= None       # None = broadcast
    wave:         int          = 1          # 1 = fast wave, 2 = reasoning wave
    ttl:          float        = 30.0       # seconds until expiry
    signal_id:    str          = field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp:    float        = field(default_factory=time.time)
    synapse_wt:   float        = 0.5        # weight looked up at emit time

    # ── helpers ───────────────────────────────────────────────────────────────

    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > self.ttl

    def effective_strength(self) -> float:
        """confidence × synapse_weight — used for priority sorting."""
        return min(1.0, self.confidence * self.synapse_wt)

    def to_dict(self) -> dict:
        return {
            "id":         self.signal_id,
            "type":       self.signal_type.value,
            "from":       self.source_agent,
            "to":         self.target_agent or "broadcast",
            "topic":      self.topic,
            "confidence": round(self.confidence, 3),
            "strength":   round(self.effective_strength(), 3),
            "wave":       self.wave,
            "age_ms":     int((time.time() - self.timestamp) * 1000),
        }


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL BUS
# ─────────────────────────────────────────────────────────────────────────────

class NeuralBus:
    """
    Neuromorphic pub/sub medium.

    Key behaviours:
    • Typed signals (see SignalType) — each type triggers different reactions.
    • INHIBITORY signals mark the target as suppressed for signal.ttl seconds.
    • EXCITATORY signals accumulate per-agent excitation boost.
    • Rolling signal buffer (MAX_SIGNAL_BUFFER most recent) for peer context.
    • Synapse weights per (source, target) pair — strengthened by co-firing,
      decayed by background decay thread.
    • build_peer_context() — assembles what other agents have found so far.
    """

    MAX_SIGNAL_BUFFER: int   = 2000
    DECAY_INTERVAL_S:  float = 60.0
    MIN_SYNAPSE_WT:    float = 0.05
    BASELINE_WT:       float = 0.5
    MAX_SYNAPSE_WT:    float = 1.0

    def __init__(
        self,
        synaptic_state: "SynapticState | None" = None,
        decay_rate: float = 0.01,
    ):
        # Optional reference to shared workspace (set by NeuralOrchestrator)
        self._state = synaptic_state
        self._decay_rate = decay_rate

        # Subscribers: topic → list[(agent_name, callback, filter_types)]
        self._subs: dict[str, list[tuple[str, Callable, Optional[list[SignalType]]]]] = {}
        # Global subscribers (receive every signal regardless of topic)
        self._global_subs: list[tuple[str, Callable]] = []

        # Rolling signal buffer
        self._buffer: deque[NeuralSignal] = deque(maxlen=self.MAX_SIGNAL_BUFFER)

        # Suppression state: agent_name → expiry timestamp
        self._suppressed: dict[str, float] = {}

        # Excitation accumulation: agent_name → cumulative boost
        self._excitation: dict[str, float] = {}

        # Synapse weights: "source→target" → float
        self._synapses: dict[str, float] = {}

        self._lock = threading.RLock()
        self._start_decay_thread()

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(
        self,
        topic: str,
        handler: Callable[[NeuralSignal], None],
        agent_name: str,
        signal_types: Optional[list[SignalType]] = None,
    ) -> None:
        """Register agent_name to receive signals on topic (optionally filtered by type)."""
        with self._lock:
            if topic not in self._subs:
                self._subs[topic] = []
            self._subs[topic].append((agent_name, handler, signal_types))

    def subscribe_all(
        self,
        handler: Callable[[NeuralSignal], None],
        agent_name: str = "monitor",
    ) -> None:
        """Receive every signal on every topic (orchestrator / monitor use)."""
        with self._lock:
            self._global_subs.append((agent_name, handler))

    def unsubscribe(self, topic: str, agent_name: str) -> None:
        with self._lock:
            if topic in self._subs:
                self._subs[topic] = [
                    (n, h, f) for n, h, f in self._subs[topic] if n != agent_name
                ]

    # ── Publishing ────────────────────────────────────────────────────────────

    def publish(self, signal: NeuralSignal) -> int:
        """
        Deliver signal to matching subscribers.
        • Drops expired signals.
        • INHIBITORY updates suppression state.
        • EXCITATORY updates excitation accumulation.
        • Records in rolling buffer.
        Returns count of handlers notified.
        """
        if signal.is_expired():
            return 0

        # Look up & attach synapse weight at delivery time
        key = f"{signal.source_agent}→{signal.target_agent or 'broadcast'}"
        with self._lock:
            signal.synapse_wt = self._synapses.get(key, self.BASELINE_WT)

        # Side effects by signal type
        self._apply_side_effects(signal)

        # Buffer it
        with self._lock:
            self._buffer.append(signal)

        # Collect handlers to notify (outside lock to prevent deadlock)
        handlers: list[Callable] = []
        with self._lock:
            if signal.target_agent:
                # Directed: only target's topic handlers
                for (name, h, ftypes) in self._subs.get(signal.topic, []):
                    if name == signal.target_agent:
                        if ftypes is None or signal.signal_type in ftypes:
                            handlers.append(h)
            else:
                # Broadcast: all subscribers on topic except the sender
                for (name, h, ftypes) in self._subs.get(signal.topic, []):
                    if name != signal.source_agent:
                        if ftypes is None or signal.signal_type in ftypes:
                            handlers.append(h)

            # Global subscribers always notified
            for (name, h) in self._global_subs:
                if name != signal.source_agent:
                    handlers.append(h)

        notified = 0
        for h in handlers:
            try:
                h(signal)
                notified += 1
            except Exception:
                pass
        return notified

    def emit(
        self,
        signal_type: SignalType,
        source_agent: str,
        topic: str,
        payload: dict,
        confidence: float,
        target_agent: Optional[str] = None,
        wave: int = 1,
        ttl: float = 30.0,
    ) -> NeuralSignal:
        """Convenience factory — builds signal and publishes it."""
        key = f"{source_agent}→{target_agent or 'broadcast'}"
        with self._lock:
            wt = self._synapses.get(key, self.BASELINE_WT)
        sig = NeuralSignal(
            signal_type=signal_type,
            source_agent=source_agent,
            topic=topic,
            payload=payload,
            confidence=confidence,
            target_agent=target_agent,
            wave=wave,
            ttl=ttl,
            synapse_wt=wt,
        )
        self.publish(sig)
        return sig

    # ── Side effects ──────────────────────────────────────────────────────────

    def _apply_side_effects(self, signal: NeuralSignal) -> None:
        with self._lock:
            if signal.signal_type == SignalType.INHIBITORY and signal.target_agent:
                expiry = time.time() + signal.ttl
                # Only suppress if stronger than current suppression
                cur = self._suppressed.get(signal.target_agent, 0.0)
                if expiry > cur or signal.confidence >= 0.85:
                    self._suppressed[signal.target_agent] = expiry

            elif signal.signal_type == SignalType.EXCITATORY and signal.target_agent:
                key = signal.target_agent
                boost = signal.confidence * signal.synapse_wt
                self._excitation[key] = min(1.0, self._excitation.get(key, 0.0) + boost)

    # ── Suppression & Excitation queries ─────────────────────────────────────

    def is_suppressed(self, agent_name: str) -> bool:
        """Return True if agent is under active INHIBITORY signal."""
        with self._lock:
            expiry = self._suppressed.get(agent_name, 0.0)
        return time.time() < expiry

    def get_inhibited_agents(self) -> set[str]:
        now = time.time()
        with self._lock:
            return {a for a, exp in self._suppressed.items() if now < exp}

    def clear_suppression(self, agent_name: str) -> None:
        with self._lock:
            self._suppressed.pop(agent_name, None)

    def get_excitation_boost(self, agent_name: str) -> float:
        with self._lock:
            return self._excitation.get(agent_name, 0.0)

    def drain_excitation(self, agent_name: str) -> float:
        """Read and reset excitation boost (consumed on use)."""
        with self._lock:
            boost = self._excitation.pop(agent_name, 0.0)
        return boost

    def reset_per_query(self) -> None:
        """Clear transient state between queries."""
        with self._lock:
            self._suppressed.clear()
            self._excitation.clear()

    # ── Peer context builder ──────────────────────────────────────────────────

    def build_peer_context(
        self,
        requesting_agent: str,
        topic: str,
        max_age_s: float = 12.0,
        signal_types: Optional[list[SignalType]] = None,
        max_entries: int = 6,
    ) -> list[dict]:
        """
        Return recent peer RESULT / CONTEXT signals for the given topic.
        Used by wave-2 agents to read what wave-1 agents found.
        """
        types = signal_types or [SignalType.RESULT, SignalType.CONTEXT]
        cutoff = time.time() - max_age_s
        with self._lock:
            buf = list(self._buffer)

        results = []
        for sig in reversed(buf):       # newest first
            if sig.timestamp < cutoff:
                break
            if sig.source_agent == requesting_agent:
                continue
            if sig.signal_type not in types:
                continue
            if topic and sig.topic != topic and not sig.topic.startswith(topic):
                continue
            results.append(sig)
            if len(results) >= max_entries:
                break

        # Sort by effective strength (confidence × synapse weight)
        results.sort(key=lambda s: s.effective_strength(), reverse=True)
        return [s.to_dict() | {"text": s.payload.get("text", s.payload.get("content", ""))[:300]}
                for s in results]

    def get_recent_results(self, max_age_s: float = 15.0) -> list[NeuralSignal]:
        """Return all RESULT signals that haven't expired."""
        cutoff = time.time() - max_age_s
        with self._lock:
            return [s for s in self._buffer
                    if s.signal_type == SignalType.RESULT and s.timestamp >= cutoff]

    # ── Synapse weight management ─────────────────────────────────────────────

    def get_synapse(self, source: str, target: str) -> float:
        key = f"{source}→{target}"
        with self._lock:
            return self._synapses.get(key, self.BASELINE_WT)

    def reinforce_synapse(self, source: str, target: str, alpha: float = 0.1) -> float:
        """Hebbian update: agents that co-fired → strengthen connection."""
        key = f"{source}→{target}"
        with self._lock:
            current = self._synapses.get(key, self.BASELINE_WT)
            new_val = min(self.MAX_SYNAPSE_WT, current + alpha * (self.MAX_SYNAPSE_WT - current))
            self._synapses[key] = new_val
        return new_val

    def load_synapses(self, weights: dict[str, float]) -> None:
        """Load pre-trained weights at startup."""
        with self._lock:
            for k, v in weights.items():
                self._synapses[k] = max(self.MIN_SYNAPSE_WT, min(self.MAX_SYNAPSE_WT, float(v)))

    def get_all_synapses(self) -> dict[str, float]:
        with self._lock:
            return dict(self._synapses)

    # ── Signal buffer queries ─────────────────────────────────────────────────

    def get_signal_buffer(
        self,
        signal_type: Optional[SignalType] = None,
        agent: Optional[str] = None,
        limit: int = 50,
    ) -> list[NeuralSignal]:
        with self._lock:
            buf = list(self._buffer)[-limit * 3:]
        if signal_type:
            buf = [s for s in buf if s.signal_type == signal_type]
        if agent:
            buf = [s for s in buf if s.source_agent == agent or s.target_agent == agent]
        return buf[-limit:]

    def get_active_agents(self, max_age_s: float = 10.0) -> list[str]:
        cutoff = time.time() - max_age_s
        with self._lock:
            return list({s.source_agent for s in self._buffer if s.timestamp >= cutoff})

    # ── Decay ─────────────────────────────────────────────────────────────────

    def _start_decay_thread(self) -> None:
        def _loop():
            while True:
                time.sleep(self.DECAY_INTERVAL_S)
                self._decay_sweep()
        t = threading.Thread(target=_loop, daemon=True, name="neural-bus-decay")
        t.start()

    def _decay_sweep(self) -> None:
        """Multiplicative decay on all synapse weights. Floor at MIN_SYNAPSE_WT."""
        with self._lock:
            for k in list(self._synapses.keys()):
                self._synapses[k] = max(
                    self.MIN_SYNAPSE_WT,
                    self._synapses[k] * (1.0 - self._decay_rate),
                )
            # Clean expired suppression
            now = time.time()
            self._suppressed = {a: exp for a, exp in self._suppressed.items() if exp > now}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            inhibited = self.get_inhibited_agents()
            top_synapses = sorted(self._synapses.items(), key=lambda x: -x[1])[:8]
            return {
                "subscribers":      sum(len(v) for v in self._subs.values()),
                "buffer_size":      len(self._buffer),
                "inhibited_agents": list(inhibited),
                "active_synapses":  len(self._synapses),
                "top_synapses":     [{"pair": k, "weight": round(v, 3)} for k, v in top_synapses],
                "active_agents":    self.get_active_agents(),
            }
