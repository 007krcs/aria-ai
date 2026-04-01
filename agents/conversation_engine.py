"""
ARIA ConversationEngine — Persistent Multi-Turn Conversation Manager
=====================================================================

Tracks full conversation state across turns:
  - Context resolution: "do that again" / "undo" / "yes" / "it" / "that"
  - Entity memory:  app, file, url, time, email, symbol accumulate per session
  - Undo stack:     last 20 reversible action tokens
  - Proactive suggestions after every action
  - Persists to SQLite (data/conversations.db)

All methods are thread-safe. No LLM calls in hot path — context resolution
uses pure string matching for <1ms latency.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH  = DATA_DIR / "conversations.db"

# ── Phrase banks (no LLM needed) ──────────────────────────────────────────────
CONFIRM_PHRASES = frozenset({
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it",
    "go ahead", "proceed", "confirmed", "confirm", "continue",
    "that's fine", "sounds good", "absolutely", "please do",
    "go on", "do that", "make it happen", "alright", "aye",
})

UNDO_PHRASES = frozenset({
    "undo", "undo that", "revert", "go back", "cancel that",
    "undo last", "take that back", "reverse that", "undo it",
    "put it back", "restore", "revert that",
})

REPEAT_PHRASES = frozenset({
    "do that again", "again", "repeat", "repeat that",
    "do it again", "one more time", "same thing", "same again",
    "run that again", "try again",
})

# Entity extraction patterns
_APP_NAMES = {
    "chrome", "firefox", "edge", "safari", "opera",
    "notepad", "notepad++", "vscode", "visual studio code",
    "excel", "word", "powerpoint", "outlook", "teams",
    "explorer", "file explorer", "cmd", "powershell", "terminal",
    "spotify", "vlc", "discord", "slack", "zoom", "skype",
    "calculator", "paint", "snipping tool", "task manager",
    "control panel", "settings", "brave", "vivaldi",
}

_CITY_NAMES = {
    "london", "new york", "paris", "tokyo", "berlin", "sydney",
    "mumbai", "delhi", "bangalore", "chennai", "hyderabad",
    "singapore", "dubai", "toronto", "chicago", "los angeles",
    "beijing", "shanghai", "moscow", "rome", "amsterdam",
}

# Proactive suggestions after each action type
_PROACTIVE = {
    "open_app:chrome":      "Want me to search for something in Chrome?",
    "open_app:firefox":     "Want me to search for something in Firefox?",
    "open_app":             "App is open. Need me to do something with it?",
    "screenshot":           "Got the screenshot. Want me to read the text from it?",
    "write_file":           "File saved. Want me to open it?",
    "read_file":            "File loaded. Want me to summarise it?",
    "list_dir":             "Want me to open any of these files?",
    "run_command:error":    "The command failed. Want me to diagnose and fix it?",
    "run_command:ok":       "Done. Want me to run another command?",
    "web_search":           "Should I save these results to memory?",
    "draft_email":          "Email drafted. Just say 'send it' when ready.",
    "get_sysinfo:cpu_high": "CPU is high — want me to find what's using it?",
    "get_sysinfo":          "Anything else you'd like to check?",
    "network_check:slow":   "Connection looks slow. Want me to run a full speed test?",
    "network_check":        "Network looks good. Anything else?",
    "browser_nav":          "Page loaded. Want me to read or search it?",
    "set_volume":           "Volume adjusted. Anything else?",
    "media_control":        None,
    "set_reminder":         "Reminder set. I'll alert you when the time comes.",
    "delete_file":          None,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    id:          str
    session_id:  str
    role:        str          # "user" | "aria"
    text:        str
    intent:      Dict[str, Any] = field(default_factory=dict)
    action_plan: Optional[str]  = None
    entities:    Dict[str, Any] = field(default_factory=dict)
    ts:          float          = field(default_factory=time.time)


@dataclass
class ConversationSession:
    id:           str
    created_ts:   float        = field(default_factory=time.time)
    last_active:  float        = field(default_factory=time.time)
    entities:     Dict[str, Any] = field(default_factory=dict)
    pending_plan: Optional[str] = None
    last_action:  Optional[str] = None
    last_args:    Dict[str, Any] = field(default_factory=dict)
    undo_stack:   List[str]     = field(default_factory=list)   # LIFO undo tokens
    context:      str           = ""                             # rolling summary


# ─────────────────────────────────────────────────────────────────────────────
# ConversationEngine
# ─────────────────────────────────────────────────────────────────────────────

class ConversationEngine:
    """
    Persistent multi-turn conversation manager.
    Thread-safe via a per-engine RLock.
    """

    def __init__(self, engine=None, db_path: Path = DB_PATH):
        self._engine   = engine
        self._db_path  = db_path
        self._lock     = threading.RLock()
        self._cache: Dict[str, ConversationSession] = {}   # in-memory session cache
        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conv_sessions (
                    id               TEXT PRIMARY KEY,
                    created_ts       REAL NOT NULL,
                    last_active      REAL NOT NULL,
                    entities_json    TEXT DEFAULT '{}',
                    pending_plan     TEXT,
                    last_action      TEXT,
                    last_args_json   TEXT DEFAULT '{}',
                    undo_stack_json  TEXT DEFAULT '[]',
                    context_text     TEXT DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS conv_turns (
                    id            TEXT PRIMARY KEY,
                    session_id    TEXT NOT NULL,
                    role          TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    intent_json   TEXT DEFAULT '{}',
                    action_plan   TEXT,
                    entities_json TEXT DEFAULT '{}',
                    ts            REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON conv_turns(session_id, ts DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_active
                    ON conv_sessions(last_active DESC);
            """)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=5, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ── Session management ────────────────────────────────────────────────────

    def get_or_create_session(self, session_id: str = "default") -> ConversationSession:
        with self._lock:
            if session_id in self._cache:
                s = self._cache[session_id]
                s.last_active = time.time()
                return s

            with self._connect() as db:
                row = db.execute(
                    "SELECT * FROM conv_sessions WHERE id=?", (session_id,)
                ).fetchone()

            if row:
                s = ConversationSession(
                    id=row["id"],
                    created_ts=row["created_ts"],
                    last_active=time.time(),
                    entities=json.loads(row["entities_json"] or "{}"),
                    pending_plan=row["pending_plan"],
                    last_action=row["last_action"],
                    last_args=json.loads(row["last_args_json"] or "{}"),
                    undo_stack=json.loads(row["undo_stack_json"] or "[]"),
                    context=row["context_text"] or "",
                )
            else:
                s = ConversationSession(id=session_id)
                self._save_session(s)

            self._cache[session_id] = s
            return s

    def _save_session(self, s: ConversationSession) -> None:
        with self._connect() as db:
            db.execute("""
                INSERT OR REPLACE INTO conv_sessions
                    (id, created_ts, last_active, entities_json,
                     pending_plan, last_action, last_args_json,
                     undo_stack_json, context_text)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                s.id, s.created_ts, s.last_active,
                json.dumps(s.entities),
                s.pending_plan, s.last_action,
                json.dumps(s.last_args),
                json.dumps(s.undo_stack[-20:]),   # keep last 20
                s.context[:2000],
            ))

    def save_turn(self, session_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            with self._connect() as db:
                db.execute("""
                    INSERT OR REPLACE INTO conv_turns
                        (id, session_id, role, text,
                         intent_json, action_plan, entities_json, ts)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    turn.id, turn.session_id, turn.role, turn.text,
                    json.dumps(turn.intent), turn.action_plan,
                    json.dumps(turn.entities), turn.ts,
                ))
            # Merge entities into session
            s = self.get_or_create_session(session_id)
            s.entities.update({k: v for k, v in turn.entities.items() if v})
            self._save_session(s)

    def get_recent_turns(self, session_id: str, n: int = 8) -> List[ConversationTurn]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM conv_turns WHERE session_id=? ORDER BY ts DESC LIMIT ?",
                (session_id, n),
            ).fetchall()
        return [
            ConversationTurn(
                id=r["id"], session_id=r["session_id"],
                role=r["role"], text=r["text"],
                intent=json.loads(r["intent_json"] or "{}"),
                action_plan=r["action_plan"],
                entities=json.loads(r["entities_json"] or "{}"),
                ts=r["ts"],
            )
            for r in reversed(rows)   # chronological order
        ]

    # ── Intent parsing ────────────────────────────────────────────────────────

    def parse_intent(self, raw_query: str, session: ConversationSession) -> dict:
        """
        Parse user query into structured intent dict.
        Zero-LLM for common patterns. LLM fallback for ambiguous cases.
        """
        q_stripped = raw_query.strip()
        q_lower    = q_stripped.lower()

        # ── 1. Detect meta-commands ───────────────────────────────────────────
        is_confirm = q_lower in CONFIRM_PHRASES or any(
            q_lower.startswith(p) for p in ("yes ", "ok ", "sure ", "go ahead")
        )
        is_undo    = q_lower in UNDO_PHRASES or any(
            p in q_lower for p in ("undo", "revert", "go back", "take that back")
        )
        is_repeat  = q_lower in REPEAT_PHRASES or any(
            p in q_lower for p in ("do that again", "do it again", "again", "repeat")
        )

        # ── 2. Resolve context references ────────────────────────────────────
        resolved = self.resolve_context(q_stripped, session)

        # ── 3. Extract entities ───────────────────────────────────────────────
        entities = self.extract_entities(resolved)

        # ── 4. Detect question vs action ─────────────────────────────────────
        is_question = self._is_question(resolved)

        # ── 5. Match ACTION_MAP (imported lazily to avoid circular import) ───
        action_type = "question"
        risk_hint   = "SAFE"
        try:
            from agents.auto_executor import classify_action_fast
            action_type, risk_hint = classify_action_fast(resolved)
        except Exception:
            if not is_question:
                action_type = "unknown_action"

        return {
            "resolved_query": resolved,
            "raw_query":      raw_query,
            "action_type":    action_type,
            "entities":       entities,
            "is_confirm":     is_confirm,
            "is_undo":        is_undo,
            "is_repeat":      is_repeat,
            "is_question":    is_question,
            "risk_hint":      risk_hint,
            "pending_plan":   session.pending_plan,
        }

    def resolve_context(self, query: str, session: ConversationSession) -> str:
        """
        Replace pronouns and references with known entities.
        Pure string matching — no LLM, <0.5ms.
        """
        q = query.strip()
        q_lower = q.lower()

        # "do that again" → reconstruct last action query
        if q_lower in REPEAT_PHRASES or "do that again" in q_lower or "repeat" in q_lower:
            if session.last_action and session.last_args:
                target = session.last_args.get("app") or session.last_args.get("file") or \
                         session.last_args.get("url") or session.last_args.get("command") or ""
                return f"{session.last_action.replace('_', ' ')} {target}".strip()

        # Replace "it" / "that" with last entity
        ent = session.entities
        replacements = {
            "open it":       f"open {ent.get('app', ent.get('file', 'it'))}",
            "close it":      f"close {ent.get('app', 'it')}",
            "read it":       f"read {ent.get('file', ent.get('url', 'it'))}",
            "search it":     f"search {ent.get('symbol', ent.get('app', 'it'))}",
            "delete it":     f"delete {ent.get('file', 'it')}",
            "send it":       f"send {ent.get('email', 'the email')}",
            "go there":      f"navigate to {ent.get('url', 'the website')}",
            "open that":     f"open {ent.get('app', ent.get('file', 'that'))}",
            "do that":       session.last_action.replace("_", " ") if session.last_action else q,
            "run that":      f"run {ent.get('command', 'that')}",
            "show me that":  f"show {ent.get('file', ent.get('url', 'that'))}",
        }
        for phrase, replacement in replacements.items():
            if q_lower == phrase or q_lower.startswith(phrase + " "):
                return replacement

        # Replace "the app", "the file", "the website" inline
        if "the app" in q_lower and ent.get("app"):
            q = re.sub(r"\bthe app\b", ent["app"], q, flags=re.IGNORECASE)
        if "the file" in q_lower and ent.get("file"):
            q = re.sub(r"\bthe file\b", ent["file"], q, flags=re.IGNORECASE)
        if any(w in q_lower for w in ("the website", "that site", "that url")) and ent.get("url"):
            q = re.sub(r"\b(the website|that site|that url)\b", ent["url"], q, flags=re.IGNORECASE)

        return q

    def extract_entities(self, query: str) -> dict:
        """
        Rule-based entity extraction. Updates session entity memory.
        """
        q = query.lower()
        entities: dict = {}

        # App names
        for app in _APP_NAMES:
            if app in q:
                entities["app"] = app
                break

        # File paths and extensions
        file_m = re.search(
            r"([A-Za-z]:[\\\/][\w\\\/\.\-]+\.\w+|[\w\-\.]+\.(txt|pdf|docx|xlsx|csv|py|js|json|html|png|jpg|mp4|mp3|zip))",
            query, re.IGNORECASE
        )
        if file_m:
            entities["file"] = file_m.group(0)

        # URLs
        url_m = re.search(r"https?://[^\s]+|(?:www\.)[a-z0-9\-]+\.[a-z]{2,}[^\s]*", query, re.IGNORECASE)
        if url_m:
            entities["url"] = url_m.group(0)

        # Email addresses
        email_m = re.search(r"[\w.\-+]+@[\w\-]+\.[a-z]{2,}", query)
        if email_m:
            entities["email"] = email_m.group(0)

        # Stock ticker symbols (2-5 uppercase letters)
        ticker_m = re.search(r"\b([A-Z]{2,5})(?:\.NS|\.BSE|\.L|\.TO)?\b", query)
        if ticker_m and ticker_m.group(1) not in {"I", "A", "OK", "AM", "PM", "US", "AI", "MY"}:
            entities["symbol"] = ticker_m.group(1)

        # Time expressions
        time_m = re.search(
            r"at (\d{1,2}(?::\d{2})?\s*(?:am|pm))|in (\d+)\s*(minute|hour|second)s?|tomorrow|tonight|today at",
            query, re.IGNORECASE
        )
        if time_m:
            entities["time"] = time_m.group(0)

        # Numbers (volume, percentage, count)
        num_m = re.search(r"\b(\d{1,3})\s*(?:%|percent|volume)?\b", query)
        if num_m and any(w in q for w in ("volume", "percent", "%", "set", "increase", "decrease")):
            entities["number"] = int(num_m.group(1))

        # City names
        for city in _CITY_NAMES:
            if city in q:
                entities["city"] = city
                break

        # Directory / folder
        dir_m = re.search(
            r"(downloads|documents|desktop|pictures|videos|music|temp|appdata|system32)",
            q, re.IGNORECASE
        )
        if dir_m:
            entities["folder"] = dir_m.group(1)

        # Shell command (quoted or after "run"/"execute"/"command")
        cmd_m = re.search(r"(?:run|execute|command|cmd|bash|shell)\s+[\"']?(.+?)[\"']?\s*$", query, re.IGNORECASE)
        if cmd_m:
            entities["command"] = cmd_m.group(1).strip('"\'')

        return entities

    def _is_question(self, query: str) -> bool:
        """Heuristic: is the query asking for information vs requesting action?"""
        q = query.strip().lower()
        question_starters = (
            "what", "who", "where", "when", "why", "how", "which",
            "can you explain", "tell me", "is it", "does", "do you",
            "what is", "what are", "give me info", "describe",
        )
        if q.endswith("?"):
            return True
        if any(q.startswith(s) for s in question_starters):
            # But "how do I open..." is still an action
            if any(w in q for w in ("open", "run", "execute", "create", "delete", "install")):
                return False
            return True
        return False

    # ── Context for LLM prompts ───────────────────────────────────────────────

    def build_context_prompt(self, session: ConversationSession, max_chars: int = 600) -> str:
        """
        Build compact conversation history for injection into LLM system prompts.
        """
        turns = self.get_recent_turns(session.id, n=6)
        if not turns:
            return ""

        lines = ["[Recent conversation:]"]
        budget = max_chars - 30
        for t in turns:
            prefix = "User" if t.role == "user" else "ARIA"
            line   = f"{prefix}: {t.text[:120]}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line) + 1

        if session.entities:
            ent_str = ", ".join(f"{k}={v}" for k, v in list(session.entities.items())[:5])
            lines.append(f"[Known entities: {ent_str}]")

        return "\n".join(lines)

    # ── Session update ────────────────────────────────────────────────────────

    def update_session_context(
        self,
        session_id:  str,
        user_text:   str,
        aria_text:   str,
        action_type: Optional[str] = None,
        action_args: Optional[dict] = None,
        undo_token:  Optional[str]  = None,
        pending_plan: Optional[str] = None,
    ) -> None:
        """Call after every turn to update rolling context + entities."""
        with self._lock:
            s = self.get_or_create_session(session_id)

            # Rolling 800-char context summary
            new_ctx = f"User: {user_text[:200]}\nARIA: {aria_text[:200]}"
            s.context = (s.context + "\n" + new_ctx)[-800:]

            # Track last action for "do that again"
            if action_type:
                s.last_action = action_type
            if action_args:
                s.last_args = action_args

            # Push undo token
            if undo_token:
                s.undo_stack.append(undo_token)
                s.undo_stack = s.undo_stack[-20:]

            # Set or clear pending plan
            s.pending_plan = pending_plan

            s.last_active = time.time()
            self._save_session(s)

            # Save turns
            uid = str(uuid.uuid4())[:12]
            for role, text in (("user", user_text), ("aria", aria_text)):
                turn = ConversationTurn(
                    id=f"{uid}_{role}", session_id=session_id,
                    role=role, text=text, ts=time.time(),
                )
                self.save_turn(session_id, turn)

    # ── Undo stack helpers ────────────────────────────────────────────────────

    def pop_undo_token(self, session_id: str) -> Optional[str]:
        """Pop and return the latest undo token."""
        with self._lock:
            s = self.get_or_create_session(session_id)
            if not s.undo_stack:
                return None
            token = s.undo_stack.pop()
            self._save_session(s)
            return token

    def push_undo_token(self, session_id: str, token: str) -> None:
        with self._lock:
            s = self.get_or_create_session(session_id)
            s.undo_stack.append(token)
            s.undo_stack = s.undo_stack[-20:]
            self._save_session(s)

    def set_pending_plan(self, session_id: str, plan_id: Optional[str]) -> None:
        with self._lock:
            s = self.get_or_create_session(session_id)
            s.pending_plan = plan_id
            self._save_session(s)

    # ── Proactive suggestions ─────────────────────────────────────────────────

    def suggest_next_step(self, action_type: str, result: dict) -> Optional[str]:
        """
        Return a proactive suggestion after an action completes.
        Checks specific outcome first (e.g. run_command:error), then generic.
        """
        # Build specific key
        status = result.get("status", "ok")
        extra  = ""
        if action_type == "run_command":
            extra = ":error" if not result.get("ok", True) else ":ok"
        elif action_type == "get_sysinfo":
            cpu = result.get("cpu_pct", 0)
            extra = ":cpu_high" if isinstance(cpu, (int, float)) and cpu > 80 else ""
        elif action_type == "network_check":
            speed = result.get("download_mbps", 100)
            extra = ":slow" if isinstance(speed, (int, float)) and speed < 5 else ""

        specific = _PROACTIVE.get(f"{action_type}{extra}")
        if specific is not None:
            return specific

        generic = _PROACTIVE.get(action_type)
        return generic

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> dict:
        s = self.get_or_create_session(session_id)
        return {
            "session_id":      s.id,
            "entities":        s.entities,
            "pending_plan":    s.pending_plan,
            "last_action":     s.last_action,
            "undo_available":  bool(s.undo_stack),
            "undo_depth":      len(s.undo_stack),
            "context_chars":   len(s.context),
            "last_active":     s.last_active,
        }

    def clear_session(self, session_id: str) -> None:
        """Reset session state (keep turns, clear entities/context)."""
        with self._lock:
            s = self.get_or_create_session(session_id)
            s.entities     = {}
            s.context      = ""
            s.pending_plan = None
            s.last_action  = None
            s.last_args    = {}
            s.undo_stack   = []
            self._save_session(s)
            if session_id in self._cache:
                del self._cache[session_id]
