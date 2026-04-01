"""
ARIA Synaptic State — Shared global workspace for all agents (v1)
=================================================================

Like the brain's short-term working memory / blackboard:
every agent reads and writes here.  Multiple agents can write the same key;
the highest-confidence non-expired entry wins for reads.

Also manages Hebbian weight persistence:
  • reinforce(source, target)  — strengthen connection when agents co-fire
  • decay_all(rate)            — weaken all weights gradually (called by NeuralBus)
  • Weights saved atomically to data/synaptic_weights.json

Consensus formation:
  • When >= 2 agents write similar content for the same topic+query,
    a 'consensus' entry is synthesised with averaged confidence.

Lateral inhibition:
  • apply_lateral_inhibition(winner, topic, threshold) marks all other
    live RESULT entries for that topic as suppressed (confidence → 0).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.neural_bus import SignalType


# ─────────────────────────────────────────────────────────────────────────────
# WORKSPACE ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceEntry:
    key:          str
    value:        Any
    author:       str
    confidence:   float
    signal_type:  SignalType
    ttl:          float        = 30.0
    created_at:   float        = field(default_factory=time.time)
    access_count: int          = 0
    tags:         list         = field(default_factory=list)
    suppressed:   bool         = False

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    def is_live(self) -> bool:
        return not self.is_expired() and not self.suppressed

    def touch(self) -> None:
        self.access_count += 1

    def to_str(self, max_len: int = 300) -> str:
        text = str(self.value)
        return text[:max_len] + "…" if len(text) > max_len else text

    def to_dict(self) -> dict:
        return {
            "key":        self.key,
            "author":     self.author,
            "confidence": round(self.confidence, 3),
            "type":       self.signal_type.value,
            "age_s":      round(time.time() - self.created_at, 1),
            "suppressed": self.suppressed,
            "value":      self.to_str(200),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SYNAPTIC STATE
# ─────────────────────────────────────────────────────────────────────────────

class SynapticState:
    """
    Global workspace: blackboard + Hebbian weight store.

    Blackboard:
    • Thread-safe read/write with TTL eviction.
    • Entries namespaced by key (agent_name, topic, etc.).
    • Consensus formation when 2+ agents agree.

    Hebbian weights (dict: "source→target" → float):
    • Loaded from data/synaptic_weights.json at startup.
    • Persisted atomically after every update.
    • decay_all() applies multiplicative decay.
    • reinforce() strengthens a pair: w += α*(MAX - w)
    """

    BASELINE_WEIGHT:  float = 0.5
    MIN_WEIGHT:       float = 0.05
    MAX_WEIGHT:       float = 1.0
    HEBBIAN_ALPHA:    float = 0.10
    CONSENSUS_N:      int   = 2
    CONSENSUS_SIM:    float = 0.35    # Jaccard word-overlap threshold

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        hebbian_alpha: float = 0.10,
        consensus_n: int = 2,
    ):
        self._hebbian_alpha = hebbian_alpha
        self._consensus_n   = consensus_n

        # Blackboard: key → list[WorkspaceEntry]
        self._workspace: dict[str, list[WorkspaceEntry]] = {}
        self._ws_lock   = threading.RLock()

        # Synapse weights: "source→target" → float
        self._weights:    dict[str, float] = {}
        self._wt_lock     = threading.RLock()

        # Weights persistence path
        self._weights_path = weights_path or self._default_weights_path()
        self._load_weights()

    # ── Weights path ──────────────────────────────────────────────────────────

    @staticmethod
    def _default_weights_path() -> Path:
        try:
            root = Path(__file__).resolve().parent.parent
        except Exception:
            root = Path(".")
        p = root / "data" / "synaptic_weights.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ── Blackboard write ──────────────────────────────────────────────────────

    def write(
        self,
        key: str,
        value: Any,
        author: str,
        confidence: float,
        signal_type: SignalType = SignalType.RESULT,
        ttl: float = 30.0,
        tags: Optional[list] = None,
    ) -> WorkspaceEntry:
        entry = WorkspaceEntry(
            key=key, value=value, author=author,
            confidence=confidence, signal_type=signal_type,
            ttl=ttl, tags=tags or [],
        )
        with self._ws_lock:
            if key not in self._workspace:
                self._workspace[key] = []
            # Remove old entry from same author for this key
            self._workspace[key] = [
                e for e in self._workspace[key]
                if e.author != author
            ]
            self._workspace[key].append(entry)
        return entry

    # ── Blackboard read ───────────────────────────────────────────────────────

    def read(self, key: str) -> Optional[WorkspaceEntry]:
        """Return highest-confidence live entry for key, or None."""
        with self._ws_lock:
            entries = [e for e in self._workspace.get(key, []) if e.is_live()]
        if not entries:
            return None
        best = max(entries, key=lambda e: e.confidence)
        best.touch()
        return best

    def read_by_author(
        self, author: str, topic: Optional[str] = None
    ) -> list[WorkspaceEntry]:
        with self._ws_lock:
            all_entries = [
                e for entries in self._workspace.values()
                for e in entries
                if e.author == author and e.is_live()
            ]
        if topic:
            all_entries = [e for e in all_entries if topic in e.key]
        return sorted(all_entries, key=lambda e: -e.confidence)

    def read_by_tag(
        self, tag: str, min_confidence: float = 0.0
    ) -> list[WorkspaceEntry]:
        with self._ws_lock:
            return [
                e for entries in self._workspace.values()
                for e in entries
                if tag in e.tags and e.is_live() and e.confidence >= min_confidence
            ]

    def read_all_live(self) -> list[WorkspaceEntry]:
        """All non-expired, non-suppressed entries sorted by confidence desc."""
        with self._ws_lock:
            entries = [
                e for elist in self._workspace.values()
                for e in elist if e.is_live()
            ]
        return sorted(entries, key=lambda e: -e.confidence)

    def has(self, key: str) -> bool:
        return self.read(key) is not None

    # ── Eviction ──────────────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        count = 0
        with self._ws_lock:
            for key in list(self._workspace.keys()):
                before = len(self._workspace[key])
                self._workspace[key] = [e for e in self._workspace[key] if not e.is_expired()]
                count += before - len(self._workspace[key])
                if not self._workspace[key]:
                    del self._workspace[key]
        return count

    def clear_author(self, author: str) -> int:
        """Remove all entries by author (called when agent is inhibited)."""
        count = 0
        with self._ws_lock:
            for key in list(self._workspace.keys()):
                before = len(self._workspace[key])
                self._workspace[key] = [e for e in self._workspace[key] if e.author != author]
                count += before - len(self._workspace[key])
        return count

    def clear(self) -> None:
        """Full reset (called at start of each query)."""
        with self._ws_lock:
            self._workspace.clear()

    # ── Context builder ───────────────────────────────────────────────────────

    def build_context(
        self,
        exclude_author: str = "",
        max_chars: int = 2000,
        topic_filter: Optional[str] = None,
    ) -> str:
        """
        Build a rich context string from the workspace.
        Called by wave-2 agents and synthesis to see all peer findings.
        """
        entries = self.read_all_live()
        if exclude_author:
            entries = [e for e in entries if e.author != exclude_author]
        if topic_filter:
            entries = [e for e in entries if topic_filter in e.key]

        lines = []
        total = 0
        for e in entries[:12]:
            line = f"• [{e.author} conf={e.confidence:.1f}]: {e.to_str(250)}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line)

        return "Peer agent findings:\n" + "\n".join(lines) if lines else ""

    # ── Consensus formation ───────────────────────────────────────────────────

    def try_form_consensus(
        self, topic: str, query_hash: str = ""
    ) -> Optional[WorkspaceEntry]:
        """
        If >= CONSENSUS_N agents wrote similar RESULT entries for this topic,
        synthesise and return a consensus entry.
        """
        entries = self.read_all_live()
        result_entries = [
            e for e in entries
            if e.signal_type == SignalType.RESULT and topic in e.key
        ]
        if len(result_entries) < self._consensus_n:
            return None

        # Find a cluster with Jaccard similarity >= threshold
        texts = [e.to_str(400) for e in result_entries]
        agreeing: list[WorkspaceEntry] = []

        for i, e in enumerate(result_entries):
            group = [e]
            for j, other in enumerate(result_entries):
                if i == j:
                    continue
                if self._text_similarity(texts[i], texts[j]) >= self.CONSENSUS_SIM:
                    group.append(other)
            if len(group) >= self._consensus_n and len(group) > len(agreeing):
                agreeing = group

        if len(agreeing) < self._consensus_n:
            return None

        # Pick the highest-confidence entry as consensus base
        best = max(agreeing, key=lambda e: e.confidence)
        avg_conf = sum(e.confidence for e in agreeing) / len(agreeing)
        consensus_key = f"consensus:{topic}:{query_hash}"
        return self.write(
            key=consensus_key,
            value=best.value,
            author="consensus",
            confidence=min(1.0, avg_conf + 0.05),  # slight boost for agreement
            signal_type=SignalType.RESULT,
            ttl=60.0,
            tags=["consensus", topic],
        )

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Jaccard word-overlap similarity (no model needed)."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    # ── Lateral inhibition ────────────────────────────────────────────────────

    def apply_lateral_inhibition(
        self,
        winner_agent: str,
        topic: str,
        confidence_threshold: float = 0.85,
    ) -> list[str]:
        """
        If winner_agent has a high-confidence RESULT on this topic,
        mark all other RESULT entries for the same topic as suppressed.
        Returns list of suppressed agent names.
        """
        winner_entry = None
        with self._ws_lock:
            for entries in self._workspace.values():
                for e in entries:
                    if e.author == winner_agent and topic in e.key and e.is_live():
                        if e.confidence >= confidence_threshold:
                            winner_entry = e
                            break
                if winner_entry:
                    break

        if not winner_entry:
            return []

        suppressed = []
        with self._ws_lock:
            for key, entries in self._workspace.items():
                if topic not in key:
                    continue
                for e in entries:
                    if e.author != winner_agent and e.signal_type == SignalType.RESULT and e.is_live():
                        e.suppressed = True
                        if "inhibited" not in e.tags:
                            e.tags.append("inhibited")
                        if e.author not in suppressed:
                            suppressed.append(e.author)
        return suppressed

    # ── Hebbian weights ───────────────────────────────────────────────────────

    def get_weight(self, source: str, target: str) -> float:
        key = f"{source}→{target}"
        with self._wt_lock:
            return self._weights.get(key, self.BASELINE_WEIGHT)

    def reinforce(self, source: str, target: str, alpha: Optional[float] = None) -> float:
        """Hebbian update: w += α * (MAX - w). Returns new weight."""
        a = alpha if alpha is not None else self._hebbian_alpha
        key = f"{source}→{target}"
        with self._wt_lock:
            current = self._weights.get(key, self.BASELINE_WEIGHT)
            new_val = min(self.MAX_WEIGHT, current + a * (self.MAX_WEIGHT - current))
            self._weights[key] = new_val
        self._save_weights()
        return new_val

    def decay_all(self, rate: float) -> None:
        """Multiplicative decay: w *= (1 - rate). Floor at MIN_WEIGHT."""
        with self._wt_lock:
            for k in list(self._weights.keys()):
                self._weights[k] = max(
                    self.MIN_WEIGHT,
                    self._weights[k] * (1.0 - rate),
                )
        self._save_weights()

    def get_all_weights(self) -> dict[str, float]:
        with self._wt_lock:
            return dict(self._weights)

    def weight_matrix(self) -> dict:
        """Nested dict: {source: {target: weight}}"""
        matrix: dict[str, dict[str, float]] = {}
        with self._wt_lock:
            for key, val in self._weights.items():
                if "→" in key:
                    src, tgt = key.split("→", 1)
                    matrix.setdefault(src, {})[tgt] = round(val, 4)
        return matrix

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_weights(self) -> None:
        try:
            if self._weights_path.exists():
                data = json.loads(self._weights_path.read_text(encoding="utf-8"))
                with self._wt_lock:
                    for k, v in data.items():
                        self._weights[k] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, float(v)))
        except Exception:
            pass

    def _save_weights(self) -> None:
        """Atomic write via temp file + rename to prevent corruption."""
        try:
            with self._wt_lock:
                data = dict(self._weights)
            tmp = self._weights_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._weights_path)
        except Exception:
            pass

    # ── Introspection ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        entries = self.read_all_live()
        authors = list({e.author for e in entries})
        return {
            "live_entries":   len(entries),
            "active_authors": authors,
            "keys":           [e.key for e in entries[:20]],
        }

    def stats(self) -> dict:
        with self._ws_lock:
            total = sum(len(v) for v in self._workspace.values())
            live  = sum(1 for v in self._workspace.values() for e in v if e.is_live())
        with self._wt_lock:
            wt_count = len(self._weights)
        return {
            "total_entries": total,
            "live_entries":  live,
            "weight_pairs":  wt_count,
            "top_weights": sorted(
                [(k, round(v, 3)) for k, v in self._weights.items()],
                key=lambda x: -x[1],
            )[:5],
        }
