"""
ARIA NeuralAgent — Base mixin that turns any agent into a neural node (v1)
==========================================================================

Inherit alongside your existing agent class:

    from core.neural_agent import NeuralAgentMixin

    class FastReasoner(NeuralAgentMixin):
        agent_id   = "fast_reasoner"
        threshold  = 0.3
        excites    = ["chain_reasoner", "nova_reasoner"]
        inhibits_on_confidence = [("chain_reasoner", 0.85)]

Each agent then gets:

  fire()        — share result with all peers via NeuralBus + SynapticState
  excite()      — tell a peer to boost priority
  inhibit()     — tell a peer to stand down
  request()     — synchronous one-to-one peer query
  enrich_context() — read what peers have found (call BEFORE generating)
  update_weights() — Hebbian reinforcement after co-firing
  is_suppressed()  — check if an INHIBITORY signal silences this agent
  should_defer()   — skip LLM call if a peer already has a better answer

All methods are safe to call from sync and async contexts.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.neural_bus   import NeuralBus, NeuralSignal, SignalType
    from core.synaptic_state import SynapticState, WorkspaceEntry


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL AGENT MIXIN
# ─────────────────────────────────────────────────────────────────────────────

class NeuralAgentMixin:
    """
    Neuromorphic communication mixin.  Add to any ARIA agent class.

    Class-level topology (override per agent):
        agent_id                 — unique name in the network
        threshold                — activation threshold (0.0–1.0) to fire
        excites                  — list of agent_ids to excite on fire()
        inhibits_on_confidence   — [(agent_id, min_confidence)] pairs
    """

    # ── Topology (override per agent) ──────────────────────────────────────────
    agent_id:                str              = "neural_agent"
    threshold:               float            = 0.3
    excites:                 list             = []       # agents to boost on fire
    inhibits_on_confidence:  list             = []       # [(target, min_conf)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Injected by orchestrator after construction
        self._neural_bus:   Optional["NeuralBus"]      = None
        self._neural_state: Optional["SynapticState"]  = None
        # Per-request state (reset by set_active_query)
        self._active_query:     str   = ""
        self._active_hash:      str   = ""
        self._activation:       float = 0.0
        self._last_fired:       float = 0.0
        self._signal_callbacks: list[Callable] = []
        self._lock = threading.Lock()

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect_neural(
        self,
        bus:   "NeuralBus",
        state: "SynapticState",
    ) -> None:
        """Wire this agent into the neural network. Called by NeuralOrchestrator."""
        self._neural_bus   = bus
        self._neural_state = state
        # Listen for direct REQUEST signals targeting this agent
        bus.subscribe(
            topic="*",                  # global topic fallback
            handler=self._on_signal,
            agent_name=self.agent_id,
        )

    def set_active_query(self, query: str, query_hash: str) -> None:
        """Called by orchestrator at start of each request."""
        self._active_query = query
        self._active_hash  = query_hash
        with self._lock:
            self._activation = 0.0

    # ── Signal reception ───────────────────────────────────────────────────────

    def _on_signal(self, signal: "NeuralSignal") -> None:
        """Internal — called by NeuralBus when signal arrives."""
        from core.neural_bus import SignalType
        with self._lock:
            # Direct-targeted signals not for us are dropped
            if signal.target_agent and signal.target_agent != self.agent_id:
                return

            wt = signal.synapse_wt if signal.synapse_wt else 0.5

            if signal.signal_type == SignalType.EXCITATORY:
                self._activation = min(1.0, self._activation + signal.confidence * wt)
            elif signal.signal_type == SignalType.INHIBITORY:
                self._activation = max(0.0, self._activation - signal.confidence * wt)
            elif signal.signal_type == SignalType.CONTEXT:
                self._activation = min(1.0, self._activation + signal.confidence * wt * 0.3)
            elif signal.signal_type == SignalType.REQUEST:
                self._activation = 1.0  # maximum urgency on direct request

        for cb in self._signal_callbacks:
            try:
                cb(signal)
            except Exception:
                pass

    def on_signal(self, callback: Callable) -> None:
        self._signal_callbacks.append(callback)

    @property
    def activation(self) -> float:
        with self._lock:
            return self._activation

    def reset_activation(self) -> None:
        with self._lock:
            self._activation = 0.0

    # ── Signal emission ────────────────────────────────────────────────────────

    def fire(
        self,
        content: str,
        confidence: float   = 0.5,
        topic:     str      = "",
        wave:      int      = 1,
        ttl:       float    = 30.0,
        metadata:  dict     = None,
    ) -> Optional["NeuralSignal"]:
        """
        Share result with all peers.
        1. Writes to SynapticState workspace.
        2. Emits RESULT signal on NeuralBus.
        3. Emits EXCITATORY to agents listed in self.excites.
        4. Emits INHIBITORY to agents in self.inhibits_on_confidence if threshold met.
        """
        if not self._neural_bus or not self._neural_state:
            return None

        from core.neural_bus import SignalType
        effective_topic = topic or self._active_hash or "general"

        # Write to shared workspace
        self._neural_state.write(
            key=f"result:{self.agent_id}:{effective_topic}",
            value=content,
            author=self.agent_id,
            confidence=confidence,
            signal_type=SignalType.RESULT,
            ttl=ttl,
        )

        # Emit RESULT signal
        payload = {"text": content[:500], "confidence": confidence}
        if metadata:
            payload.update(metadata)

        sig = self._neural_bus.emit(
            signal_type=SignalType.RESULT,
            source_agent=self.agent_id,
            topic=effective_topic,
            payload=payload,
            confidence=confidence,
            wave=wave,
            ttl=ttl,
        )

        # Emit EXCITATORY to downstream peers
        for peer_id in self.excites:
            self.excite(
                target_agent=peer_id,
                topic=effective_topic,
                reason=f"{self.agent_id} has new data",
                strength=confidence * 0.8,
                wave=wave,
            )

        # Lateral inhibition for high-confidence results
        for (target, min_conf) in self.inhibits_on_confidence:
            if confidence >= min_conf:
                self.inhibit(
                    target_agent=target,
                    topic=effective_topic,
                    reason=f"{self.agent_id} achieved {confidence:.2f} confidence",
                    strength=0.9,
                )

        self._last_fired = time.time()
        return sig

    def excite(
        self,
        target_agent: str,
        topic:        str,
        reason:       str   = "",
        strength:     float = 0.7,
        wave:         int   = 1,
    ) -> None:
        """Tell another agent to boost its scheduling priority."""
        if not self._neural_bus:
            return
        from core.neural_bus import SignalType
        self._neural_bus.emit(
            signal_type=SignalType.EXCITATORY,
            source_agent=self.agent_id,
            topic=topic,
            payload={"reason": reason[:200]},
            confidence=strength,
            target_agent=target_agent,
            wave=wave,
            ttl=20.0,
        )

    def inhibit(
        self,
        target_agent: str,
        topic:        str,
        reason:       str   = "",
        strength:     float = 0.8,
    ) -> None:
        """Tell another agent to stand down — you have it covered."""
        if not self._neural_bus:
            return
        from core.neural_bus import SignalType
        self._neural_bus.emit(
            signal_type=SignalType.INHIBITORY,
            source_agent=self.agent_id,
            topic=topic,
            payload={"reason": reason[:200]},
            confidence=strength,
            target_agent=target_agent,
            ttl=25.0,
        )
        # Also suppress in workspace
        if self._neural_state:
            self._neural_state.apply_lateral_inhibition(
                self.agent_id, topic, confidence_threshold=0.0
            )

    def request(
        self,
        target_agent: str,
        topic:        str,
        payload:      dict,
        timeout_s:    float = 5.0,
    ) -> Optional[dict]:
        """
        Directed one-to-one request. Blocks until target replies or timeout.
        Returns target's result payload or None.
        """
        if not self._neural_bus:
            return None
        from core.neural_bus import SignalType

        result_holder: list[dict] = []
        event = threading.Event()

        def on_reply(signal: "NeuralSignal"):
            if (signal.source_agent == target_agent and
                    signal.signal_type == SignalType.RESULT and
                    signal.topic == topic):
                result_holder.append(signal.payload)
                event.set()

        self._neural_bus.subscribe_all(on_reply, agent_name=f"{self.agent_id}_req_listener")
        self._neural_bus.emit(
            signal_type=SignalType.REQUEST,
            source_agent=self.agent_id,
            topic=topic,
            payload=payload,
            confidence=1.0,
            target_agent=target_agent,
            ttl=timeout_s + 2.0,
        )
        event.wait(timeout=timeout_s)
        return result_holder[0] if result_holder else None

    def share_context(
        self,
        topic:      str,
        content:    str,
        confidence: float = 0.5,
        wave:       int   = 1,
    ) -> None:
        """Share partial/intermediate context (not a final result)."""
        if not self._neural_bus:
            return
        from core.neural_bus import SignalType
        self._neural_bus.emit(
            signal_type=SignalType.CONTEXT,
            source_agent=self.agent_id,
            topic=topic,
            payload={"text": content[:400]},
            confidence=confidence,
            wave=wave,
            ttl=20.0,
        )
        if self._neural_state:
            self._neural_state.write(
                key=f"context:{self.agent_id}:{topic}",
                value=content,
                author=self.agent_id,
                confidence=confidence,
                signal_type=SignalType.CONTEXT,
                ttl=25.0,
            )

    # ── Synaptic enrichment ────────────────────────────────────────────────────

    def enrich_context(
        self,
        topic:          str   = "",
        max_chars:      int   = 1200,
        min_confidence: float = 0.25,
    ) -> str:
        """
        Read what peer agents have found so far.
        Call this BEFORE generating your LLM response to incorporate peer knowledge.
        Returns a formatted string to prepend to your system prompt.
        """
        if not self._neural_state:
            return ""
        effective_topic = topic or self._active_hash or ""
        ctx = self._neural_state.build_context(
            exclude_author=self.agent_id,
            max_chars=max_chars,
            topic_filter=effective_topic if effective_topic else None,
        )
        return ctx

    def read_peer(self, peer_id: str) -> Optional[str]:
        """Read a specific peer agent's latest result."""
        if not self._neural_state:
            return None
        topic = self._active_hash or "general"
        entry = self._neural_state.read(f"result:{peer_id}:{topic}")
        return entry.to_str() if entry else None

    def write_workspace(
        self,
        key:        str,
        value:      str,
        confidence: float = 0.5,
        ttl:        float = 30.0,
    ) -> None:
        """Write any data to the shared workspace."""
        if self._neural_state:
            from core.neural_bus import SignalType
            self._neural_state.write(key, value, self.agent_id, confidence,
                                     SignalType.CONTEXT, ttl=ttl)

    # ── Suppression checks ─────────────────────────────────────────────────────

    def is_suppressed(self) -> bool:
        """Return True if an INHIBITORY signal has silenced this agent."""
        if not self._neural_bus:
            return False
        return self._neural_bus.is_suppressed(self.agent_id)

    def should_defer(
        self,
        topic:                str   = "",
        confidence_threshold: float = 0.85,
    ) -> bool:
        """
        Return True if a peer already has a high-confidence answer for this topic.
        Allows this agent to skip its LLM call and save compute.
        """
        if not self._neural_state:
            return False
        effective_topic = topic or self._active_hash or ""
        entries = self._neural_state.read_all_live()
        for e in entries:
            if (e.author != self.agent_id and
                    e.confidence >= confidence_threshold and
                    (not effective_topic or effective_topic in e.key)):
                return True
        return False

    # ── Hebbian learning ───────────────────────────────────────────────────────

    def update_weights(self, co_fired_peers: list[str]) -> None:
        """
        After firing, call this to reinforce connections with peers
        that also fired on the same topic (Hebbian: fire together → wire together).
        """
        if not self._neural_state or not self._neural_bus:
            return
        topic = self._active_hash or "general"
        for peer in co_fired_peers:
            # Check if peer actually has a live result for this topic
            entry = self._neural_state.read(f"result:{peer}:{topic}")
            if entry and entry.is_live():
                # Reinforce in both state and bus
                self._neural_state.reinforce(self.agent_id, peer)
                self._neural_bus.reinforce_synapse(self.agent_id, peer)

    def get_weight(self, peer_id: str) -> float:
        if self._neural_state:
            return self._neural_state.get_weight(self.agent_id, peer_id)
        return 0.5

    # ── Query hash utility ─────────────────────────────────────────────────────

    @staticmethod
    def make_query_hash(query: str) -> str:
        return hashlib.sha1(query.strip().lower().encode()).hexdigest()[:12]

    # ── Stats ──────────────────────────────────────────────────────────────────

    def neural_stats(self) -> dict:
        suppressed = self.is_suppressed() if self._neural_bus else False
        with self._lock:
            activation = self._activation
        return {
            "agent_id":    self.agent_id,
            "activation":  round(activation, 3),
            "threshold":   self.threshold,
            "suppressed":  suppressed,
            "last_fired":  round(self._last_fired, 1),
            "excites":     self.excites,
            "inhibits_on": [t for t, _ in self.inhibits_on_confidence],
        }
