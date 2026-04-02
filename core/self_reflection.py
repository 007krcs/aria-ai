"""
ARIA Self-Reflection Engine
============================
After every response, ARIA evaluates itself:

  - Was the task completed?
  - Was the answer confident or uncertain?
  - Should something be stored in memory?
  - Did the user seem satisfied?
  - Was there a pattern worth learning?

This is what makes ARIA improve over time without retraining.
Runs asynchronously — never blocks the response.
"""

from __future__ import annotations

import re
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
REFLECT_LOG   = PROJECT_ROOT / "data" / "reflection_log.json"
REFLECT_LOG.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class QualityEvaluator:
    """
    Rule-based quality scoring — no model needed.
    Scores 0.0 to 1.0.
    """

    # Signs of uncertainty in ARIA's own answers
    UNCERTAINTY_MARKERS = [
        "i'm not sure", "i am not sure", "i don't know", "i do not know",
        "i'm uncertain", "cannot confirm", "not certain", "i think", "i believe",
        "possibly", "might be", "could be", "i'm unable", "i cannot",
        "no information", "can't find", "couldn't find",
    ]

    # Signs of a good, complete answer
    QUALITY_MARKERS = [
        "here is", "here are", "the answer is", "to summarize",
        "in conclusion", "the result", "i found", "according to",
        "based on", "the solution", "step 1", "step 2",
    ]

    # Signs the task was actually done
    COMPLETION_MARKERS = [
        "done", "completed", "finished", "here you go", "here's",
        "i've created", "i've written", "i've found", "i've analyzed",
        "successfully", "result:", "output:",
    ]

    def score(self, query: str, answer: str, intent: Dict) -> Dict[str, Any]:
        a_lower = answer.lower()
        q_lower = query.lower()

        # Uncertainty score (0 = very uncertain, 1 = very confident)
        uncertainty_hits = sum(1 for m in self.UNCERTAINTY_MARKERS if m in a_lower)
        confidence = max(0.1, 1.0 - uncertainty_hits * 0.2)

        # Quality score (0 = poor, 1 = good)
        quality_hits = sum(1 for m in self.QUALITY_MARKERS if m in a_lower)
        completion_hits = sum(1 for m in self.COMPLETION_MARKERS if m in a_lower)
        quality = min(1.0, 0.5 + quality_hits * 0.1 + completion_hits * 0.15)

        # Length appropriateness
        words = len(answer.split())
        complexity = intent.get("complexity", "medium")
        ideal_range = {"low": (10, 100), "medium": (50, 300), "high": (100, 800)}
        lo, hi = ideal_range.get(complexity, (50, 300))
        length_score = 1.0 if lo <= words <= hi else (0.7 if words > hi else 0.6)

        # Task completion
        task_done = completion_hits > 0 or (
            intent.get("intent") == "question" and confidence > 0.7
        )

        # Overall
        overall = (confidence * 0.4 + quality * 0.4 + length_score * 0.2)

        return {
            "confidence":   round(confidence, 2),
            "quality":      round(quality, 2),
            "length_score": round(length_score, 2),
            "overall":      round(overall, 2),
            "task_done":    task_done,
            "word_count":   words,
        }

    def should_retry(self, scores: Dict) -> bool:
        return scores["overall"] < 0.35 or scores["confidence"] < 0.2

    def should_remember(self, scores: Dict, intent: Dict) -> bool:
        """Decide if this Q&A pair is worth storing long-term."""
        if intent.get("intent") in ("preference", "correction"):
            return True
        if scores["overall"] > 0.7 and scores["task_done"]:
            return True
        return False

    def should_store_fact(self, answer: str) -> bool:
        """Does the answer contain facts worth storing in semantic memory?"""
        fact_markers = ["is a", "are a", "means", "refers to", "defined as",
                        "stands for", "invented by", "founded by", "was born"]
        return any(m in answer.lower() for m in fact_markers)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class PatternDetector:
    """
    Detects repeated behaviors worth learning as workflows.
    """
    def __init__(self):
        self._history: List[Dict] = []

    def record(self, intent: str, query: str):
        self._history.append({
            "intent": intent,
            "query_hash": hash(query[:50]),
            "ts": time.time()
        })
        # Keep last 50
        self._history = self._history[-50:]

    def detect_workflow(self) -> Optional[str]:
        """Return a workflow description if a pattern is found."""
        if len(self._history) < 3:
            return None

        # Check for repeated intent sequences
        recent_intents = [h["intent"] for h in self._history[-6:]]
        from collections import Counter
        counts = Counter(recent_intents)
        dominant = counts.most_common(1)[0]

        if dominant[1] >= 3:
            return f"User frequently performs '{dominant[0]}' tasks"

        return None


# ─────────────────────────────────────────────────────────────────────────────
# SELF REFLECTOR
# ─────────────────────────────────────────────────────────────────────────────

class SelfReflector:
    """
    Post-response self-evaluation engine.
    Always runs in background — never delays response.
    """

    def __init__(self):
        self.evaluator = QualityEvaluator()
        self.patterns  = PatternDetector()
        self._log: List[Dict] = []
        self._load_log()

    def _load_log(self):
        if REFLECT_LOG.exists():
            try:
                self._log = json.loads(REFLECT_LOG.read_text())[-200:]
            except Exception:
                self._log = []

    def _save_log(self):
        try:
            REFLECT_LOG.write_text(json.dumps(self._log[-200:], indent=2))
        except Exception:
            pass

    def reflect_async(self, query: str, answer: str, intent: Dict):
        """Non-blocking reflection — runs in background thread."""
        t = threading.Thread(
            target=self._reflect,
            args=(query, answer, intent),
            daemon=True,
            name="aria-reflect"
        )
        t.start()

    def _reflect(self, query: str, answer: str, intent: Dict):
        try:
            scores = self.evaluator.score(query, answer, intent)
            self.patterns.record(intent.get("intent", "unknown"), query)

            entry = {
                "ts":     datetime.now().isoformat(),
                "intent": intent.get("intent"),
                "scores": scores,
                "query_len": len(query.split()),
            }

            # Log the reflection
            self._log.append(entry)
            self._save_log()

            # Write to memory if worthy
            if self.evaluator.should_remember(scores, intent):
                try:
                    from core.memory_hierarchy import MemoryHierarchy
                    mem = MemoryHierarchy()
                    if self.evaluator.should_store_fact(answer):
                        mem.write_fact(answer[:500], trust=scores["confidence"])
                    else:
                        mem.write_episodic(
                            f"Q: {query[:200]}\nA: {answer[:400]}",
                            metadata={"quality": scores["overall"], "intent": intent.get("intent")}
                        )
                except Exception:
                    pass

            # Detect and store workflow patterns
            workflow = self.patterns.detect_workflow()
            if workflow:
                try:
                    from core.memory_hierarchy import MemoryHierarchy
                    mem = MemoryHierarchy()
                    existing = mem.get_workflows()
                    if not any(workflow in w.get("text", "") for w in existing):
                        mem.write_procedural(
                            workflow,
                            metadata={"type": "detected_pattern", "auto": True}
                        )
                except Exception:
                    pass

        except Exception:
            pass

    # ── Analytics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        if not self._log:
            return {"total": 0}

        scores = [e["scores"]["overall"] for e in self._log if "scores" in e]
        intents = [e.get("intent") for e in self._log if e.get("intent")]

        from collections import Counter
        intent_dist = dict(Counter(intents).most_common(5))

        return {
            "total_reflections": len(self._log),
            "avg_quality":       round(sum(scores) / len(scores), 2) if scores else 0,
            "min_quality":       round(min(scores), 2) if scores else 0,
            "max_quality":       round(max(scores), 2) if scores else 0,
            "intent_distribution": intent_dist,
        }

    def get_recent(self, n: int = 5) -> List[Dict]:
        return self._log[-n:]
