"""
ARIA Goal Engine
================
Tracks what ARIA is trying to accomplish at three levels:

  Immediate  — current task (cleared after completion)
  Session    — this conversation's objective (cleared on session end)
  Long-term  — persistent user goals (survived across sessions)

The brain checks active goals before answering so responses stay aligned.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOALS_PATH   = PROJECT_ROOT / "data" / "aria_goals.json"
GOALS_PATH.parent.mkdir(exist_ok=True)

_lock = threading.Lock()


class GoalEngine:
    """
    Manages ARIA's three-tier goal hierarchy.
    """

    LEVELS = ("immediate", "session", "long_term")

    def __init__(self):
        self._persistent: List[Dict] = []   # long-term (saved to disk)
        self._session:    List[Dict] = []   # session goals (in-memory)
        self._immediate:  List[Dict] = []   # current task (in-memory)
        self._load()

    def _load(self):
        if GOALS_PATH.exists():
            try:
                data = json.loads(GOALS_PATH.read_text())
                self._persistent = data.get("long_term", [])
            except Exception:
                self._persistent = []

    def _save(self):
        try:
            GOALS_PATH.write_text(json.dumps({
                "long_term": self._persistent,
                "saved_at":  datetime.now().isoformat(),
            }, indent=2))
        except Exception:
            pass

    # ── Add ──────────────────────────────────────────────────────────────────

    def add(self, goal: str, level: str = "session", metadata: Optional[Dict] = None) -> str:
        if level not in self.LEVELS:
            level = "session"

        entry = {
            "id":         f"{level}_{datetime.now().strftime('%H%M%S%f')[:12]}",
            "goal":       goal,
            "level":      level,
            "status":     "active",
            "created_at": datetime.now().isoformat(),
            "metadata":   metadata or {},
        }

        with _lock:
            if level == "immediate":
                self._immediate.append(entry)
            elif level == "session":
                self._session.append(entry)
            else:
                self._persistent.append(entry)
                self._save()

        return entry["id"]

    # ── Complete / fail ───────────────────────────────────────────────────────

    def complete(self, goal_id: str):
        self._update_status(goal_id, "completed")

    def fail(self, goal_id: str, reason: str = ""):
        self._update_status(goal_id, "failed", note=reason)

    def _update_status(self, goal_id: str, status: str, note: str = ""):
        for store in [self._immediate, self._session, self._persistent]:
            for g in store:
                if g["id"] == goal_id:
                    g["status"] = status
                    g["completed_at"] = datetime.now().isoformat()
                    if note:
                        g["note"] = note
                    if store is self._persistent:
                        self._save()
                    return

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_active(self, level: Optional[str] = None) -> List[Dict]:
        """Return all active goals. Optionally filter by level."""
        all_goals = self._immediate + self._session + self._persistent
        active = [g for g in all_goals if g["status"] == "active"]
        if level:
            active = [g for g in active if g["level"] == level]
        return active

    def get_immediate(self) -> Optional[Dict]:
        active = [g for g in self._immediate if g["status"] == "active"]
        return active[-1] if active else None

    def get_context_string(self) -> str:
        """Format goals for injection into LLM context."""
        active = self.get_active()
        if not active:
            return ""
        parts = []
        for g in active[:5]:
            parts.append(f"[{g['level'].upper()}] {g['goal']}")
        return "Current goals:\n" + "\n".join(parts)

    def clear_immediate(self):
        self._immediate = [g for g in self._immediate if g["status"] != "active"]

    def clear_session(self):
        self._session = [g for g in self._session if g["status"] != "active"]

    # ── Smart goal inference ──────────────────────────────────────────────────

    def infer_from_query(self, query: str) -> Optional[str]:
        """
        Try to infer an immediate goal from the query.
        Returns goal_id if inferred, else None.
        """
        q = query.lower()

        # Task patterns
        task_phrases = [
            ("find", "Research: "),
            ("create", "Create: "),
            ("write", "Write: "),
            ("analyze", "Analyze: "),
            ("fix", "Fix: "),
            ("build", "Build: "),
            ("plan", "Plan: "),
            ("summarize", "Summarize: "),
        ]

        for trigger, prefix in task_phrases:
            if q.startswith(trigger) or f" {trigger} " in q:
                goal_text = prefix + query[:100]
                return self.add(goal_text, level="immediate")

        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "immediate": len([g for g in self._immediate if g["status"] == "active"]),
            "session":   len([g for g in self._session   if g["status"] == "active"]),
            "long_term": len([g for g in self._persistent if g["status"] == "active"]),
            "total_active": len(self.get_active()),
        }
