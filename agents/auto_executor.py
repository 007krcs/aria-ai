"""
ARIA AutoExecutor — Autonomous Action Engine
============================================

Executes any natural language command with minimal human intervention.

Risk Classification:
    SAFE      → Execute immediately, report after (file reads, info, search, media)
    CAUTION   → Stream "On it — <action>…" then execute (write files, run commands)
    DANGEROUS → Show plan, block execution, wait for explicit confirmation

Execution pipeline:
    1. classify_action_fast()  — regex ACTION_MAP, <1ms, no LLM
    2. classify_risk()         — hard rules, LLM fallback for ambiguous
    3. _parse_intent_to_plan() — builds ActionPlan from intent
    4. execute()               — streams SSE events throughout
    5. _jarvis()               — natural language result message

Undo:
    Every reversible action stores an undo token.
    AutoExecutor.undo(token) reverses the action.

Self-healing:
    On first failure: retry with alternative strategy.
    On second failure: report with plain-English error + suggestion.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Risk levels
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    SAFE      = "SAFE"
    CAUTION   = "CAUTION"
    DANGEROUS = "DANGEROUS"


# ─────────────────────────────────────────────────────────────────────────────
# ACTION_MAP — phrase patterns → (action_type, RiskLevel)
# Checked top-to-bottom; first match wins.
# ─────────────────────────────────────────────────────────────────────────────

ACTION_MAP: List[Tuple[str, str, RiskLevel]] = [
    # OS control — DANGEROUS
    (r"\b(shutdown|restart|reboot|hibernate)\b",                           "os_control",       RiskLevel.DANGEROUS),
    (r"\b(lock screen|sign out|log ?out)\b",                               "os_control",       RiskLevel.DANGEROUS),
    (r"\b(install|pip install|npm install|apt install|brew install)\b",    "install_pkg",      RiskLevel.DANGEROUS),
    (r"\bsend\s+(an?\s+)?email\b",                                         "send_email",       RiskLevel.DANGEROUS),
    (r"\b(format|wipe|erase)\s+(disk|drive|partition)\b",                  "os_control",       RiskLevel.DANGEROUS),

    # File delete — DANGEROUS
    (r"\b(delete|remove|erase|rm|trash)\b.+(file|folder|directory)",       "delete_file",      RiskLevel.DANGEROUS),

    # Port scan — CAUTION
    (r"\b(port\s+scan|scan\s+(network|ports))\b",                          "port_scan",        RiskLevel.CAUTION),

    # Terminal / command — CAUTION
    (r"\b(run|execute|terminal|bash|powershell|cmd|shell|command)\b",      "run_command",      RiskLevel.CAUTION),

    # Browser actions — CAUTION
    (r"\b(fill|submit|click\s+(?:the\s+)?button|automate\s+form)\b",       "browser_action",   RiskLevel.CAUTION),

    # Email draft — SAFE (draft only, not send)
    (r"\b(draft|compose|write)\s+(an?\s+)?email\b",                        "draft_email",      RiskLevel.SAFE),

    # Write / create file — CAUTION
    (r"\b(write|save|create|make)\s+.*(file|document|note|report)\b",      "write_file",       RiskLevel.CAUTION),
    (r"\b(write|save|create)\s+to\s+.+\.\w+",                              "write_file",       RiskLevel.CAUTION),

    # App control — SAFE
    (r"\b(open|launch|start)\b.*(chrome|firefox|edge|notepad|excel|word|vscode|"
     r"terminal|powershell|cmd|explorer|spotify|vlc|discord|slack|zoom|"
     r"calculator|paint|brave|task\s*manager|control\s*panel)\b",          "open_app",         RiskLevel.SAFE),
    (r"\b(close|quit|kill)\b.*(app|application|window|chrome|firefox|notepad)\b",
                                                                            "close_app",        RiskLevel.CAUTION),

    # System info — SAFE
    (r"\b(cpu|ram|memory\s+usage|disk\s+usage|battery|uptime|system\s+info|"
     r"how\s+much\s+(ram|cpu)|temperature\s+sensor)\b",                    "get_sysinfo",      RiskLevel.SAFE),
    (r"\b(what.?s\s+running|running\s+apps|active\s+windows|processes|task\s*list)\b",
                                                                            "list_processes",   RiskLevel.SAFE),

    # Screenshot — SAFE
    (r"\b(screenshot|screen\s*shot|capture\s+screen|snap\s+screen)\b",     "screenshot",       RiskLevel.SAFE),

    # Volume / audio — SAFE
    (r"\b(volume|mute|unmute|louder|quieter|turn\s+(up|down)\s+volume|set\s+volume)\b",
                                                                            "set_volume",       RiskLevel.SAFE),

    # Clipboard — SAFE
    (r"\b(clipboard|copy|paste|what.?s\s+in\s+clipboard)\b",               "clipboard",        RiskLevel.SAFE),

    # File read / list — SAFE
    (r"\b(read|open|show|display)\b.*(file|document|pdf|txt|csv|spreadsheet)\b",
                                                                            "read_file",        RiskLevel.SAFE),
    (r"\b(list|show|what.?s\s+in)\b.*(folder|directory|desktop|downloads|documents)\b",
                                                                            "list_dir",         RiskLevel.SAFE),

    # Web search / browse — SAFE
    (r"\b(search(\s+for|\s+the\s+web|\s+online)?|look\s+up|google|find\s+online)\b",
                                                                            "web_search",       RiskLevel.SAFE),
    (r"\b(browse|navigate|go\s+to|open\s+(website|url|webpage)|visit)\b",  "browser_nav",      RiskLevel.SAFE),

    # Media control — SAFE
    (r"\b(play|pause|stop\s+music|next\s+track|previous\s+track|skip|resume)\b",
                                                                            "media_control",    RiskLevel.SAFE),

    # Network — SAFE
    (r"\b(ping|check\s+internet|check\s+connection|wifi\s+status|ip\s+address)\b",
                                                                            "network_check",    RiskLevel.SAFE),
    (r"\b(speed\s*test|bandwidth|internet\s+speed|download\s+speed)\b",    "speed_test",       RiskLevel.SAFE),

    # Reminder / notification — SAFE
    (r"\b(remind\s+me|set\s+(a\s+)?reminder|alert\s+me|notify\s+me)\b",   "set_reminder",     RiskLevel.SAFE),

    # Stock price lookup — SAFE (single company share/stock price today)
    (r"\b(what\s+(is|are|'s|was)\s+[\w\s]{1,20}(share|stock)\s*(price|today|rate|now)?|"
     r"[\w.]{1,10}\s+(share|stock)\s*(price|today|rate|now)|"
     r"price\s+of\s+\w+(\.\w{2,3})?\s*(share|stock|today)?|"
     r"(share|stock)\s+price\s+of\s+\w+|"
     r"how\s+much\s+is\s+[\w\s]{1,20}(share|stock))\b",
                                                                            "stock_predict",    RiskLevel.SAFE),

    # Stock — SAFE (broad pattern: any stock/share/market analysis query)
    (r"\b(top\s*\d*\s*stocks?|best\s+shares?|stock\s+analysis|market\s+ranking|"
     r"stock\s+performing|shares?\s+to\s+buy|which\s+stocks?|stocks?\s+today|"
     r"buy\s+for\s+(more\s+)?gain|nse\s+stocks?|bse\s+stocks?|nifty\s+stocks?|"
     r"sensex\s+stocks?|stock\s+market\s+today|good\s+stocks?|recommend\s+stocks?|"
     r"invest\s+in\s+stocks?)\b",
                                                                            "stock_analysis",   RiskLevel.SAFE),

    # Stock Prediction — SAFE (single-ticker deep prediction + intraday)
    (r"\b(predict|forecast|projection|intraday|price\s+target|entry\s+price|stop\s+loss|"
     r"target\s+price|will\s+\w+\s+(go|rise|fall|drop|move)|when\s+to\s+(buy|sell)|"
     r"should\s+i\s+(buy|sell)|kalman|hmm\s+model|bayesian|wave\s+analysis)\b.*\b\w+\b",
                                                                            "stock_predict",    RiskLevel.SAFE),

    # Stock Monitor — SAFE (focus/watch/alert on a specific stock)
    (r"\b(focus\s+on|monitor|watch|alert\s+me|notify\s+me|track)\b.*\b(stock|share|ticker|"
     r"nse|bse|nifty|sensex|\.[A-Z]{2})\b",
                                                                            "stock_monitor",    RiskLevel.SAFE),

    # Story tell / narrate URL or topic — SAFE
    (r"\b(tell|narrate|read|retell|story|story\s*tell)\b.*(https?://|url|link|article|blog|page)\b",
                                                                            "story_tell",       RiskLevel.SAFE),
    (r"https?://\S+.*(tell|narrate|read|retell|humor|funny|laugh|tone|style|way)",
                                                                            "story_tell",       RiskLevel.SAFE),
    (r"\b(narrate|retell|read\s+(this|it|aloud)|story\s*tell)\b.*\b(humor|funny|laugh|tone|comedy|dramatic|children|thriller)\b",
                                                                            "story_tell",       RiskLevel.SAFE),
    (r"\b(tell|narrate|read|retell)\b.*(story|tale|book|article|page)\b.*(humor|funny|laugh|tone|comedy|dramatic|children|thriller|epic|sarcastic|casual|engaging)\b",
                                                                            "story_tell",       RiskLevel.SAFE),

    # Translate — SAFE
    (r"\btranslate\b",                                                      "translate",        RiskLevel.SAFE),

    # Grammar correct — SAFE
    (r"\b(correct\s+(my\s+)?(grammar|spelling)|fix\s+(my\s+)?(grammar|spelling))\b",
                                                                            "grammar_correct",  RiskLevel.SAFE),
]

# Compile all patterns once
_COMPILED_MAP: List[Tuple[re.Pattern, str, RiskLevel]] = [
    (re.compile(pat, re.IGNORECASE), act, risk)
    for pat, act, risk in ACTION_MAP
]

# Dangerous substring override — if these appear, always bump to DANGEROUS
_DANGER_OVERRIDES = [
    r"rm\s+-rf", r"del\s+/f\s+/s", r"format\s+c:", r"mkfs\.",
    r"drop\s+table", r"truncate\s+table", r"DROP\s+DATABASE",
]
_DANGER_COMPILED = [re.compile(p, re.IGNORECASE) for p in _DANGER_OVERRIDES]


def classify_action_fast(query: str) -> Tuple[str, str]:
    """
    Module-level function — importable without instantiating AutoExecutor.
    Returns (action_type, risk_level_str). Pure regex, <1ms.
    """
    # Danger override check first
    for pat in _DANGER_COMPILED:
        if pat.search(query):
            return "os_control", RiskLevel.DANGEROUS.value

    for pat, action, risk in _COMPILED_MAP:
        if pat.search(query):
            return action, risk.value

    return "question", RiskLevel.SAFE.value


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionPlan:
    id:          str             = field(default_factory=lambda: str(uuid.uuid4())[:10])
    intent:      str             = ""
    action_type: str             = "question"
    args:        Dict[str, Any]  = field(default_factory=dict)
    risk:        RiskLevel       = RiskLevel.SAFE
    reversible:  bool            = True
    steps:       List[dict]      = field(default_factory=list)   # for multi-step
    created_ts:  float           = field(default_factory=time.time)
    status:      str             = "pending"                      # pending|running|done|failed|blocked


@dataclass
class ExecutionResult:
    plan_id:    str
    success:    bool
    output:     str
    jarvis_msg: str
    undo_token: Optional[str]
    ms:         int


# ─────────────────────────────────────────────────────────────────────────────
# JARVIS response templates
# ─────────────────────────────────────────────────────────────────────────────

_JARVIS_TEMPLATES: Dict[str, List[str]] = {
    "open_app":       ["Opening {target}.", "{target} is launching.", "Done — {target} is ready."],
    "close_app":      ["{target} has been closed.", "Done — {target} shut down."],
    "set_volume":     ["Volume set to {value}%.", "Done — audio adjusted to {value}%."],
    "screenshot":     ["Screenshot captured. Saved to {path}.", "Done — screenshot taken."],
    "get_sysinfo":    ["Here's your system status:"],
    "list_processes": ["Here are the running processes:"],
    "run_command":    ["Command executed. Output:"],
    "web_search":     ["Here's what I found:"],
    "read_file":      ["Here's the content of {target}:"],
    "write_file":     ["File written to {path}.", "Done — {path} saved."],
    "list_dir":       ["Here's what's in {target}:"],
    "media_control":  ["Done.", "Media updated."],
    "network_check":  ["Network check complete:"],
    "speed_test":     ["Speed test complete:"],
    "set_reminder":   ["Reminder set — I'll alert you.", "Got it — reminder queued."],
    "draft_email":    ["Here's a draft email for your review:"],
    "stock_analysis": ["Here are the top stocks:"],
    "translate":      ["Translation complete:"],
    "grammar_correct":["Here's the corrected text:"],
    "browser_nav":    ["Page loaded.", "Done — navigated to {target}."],
    "clipboard":      ["Clipboard content:"],
    "browser_action": ["Done — action completed on the page."],
    "port_scan":      ["Port scan results:"],
    "delete_file":    ["{target} deleted."],
    "os_control":     ["Done."],
}

_CAUTION_PREFIX: Dict[str, str] = {
    "run_command":    "On it — running your command…",
    "write_file":     "Writing to file…",
    "close_app":      "Closing {target}…",
    "browser_action": "Automating the page…",
    "port_scan":      "Scanning ports…",
    "delete_file":    "Deleting {target}…",
    "install_pkg":    "Installing {target}…",
}

_DANGEROUS_PREFIX: Dict[str, str] = {
    "os_control":  "This will {action} the system. That can't be undone. Confirm?",
    "delete_file": "This will permanently delete {target}. That can't be undone. Confirm?",
    "send_email":  "About to send an email to {target}. Confirm?",
    "install_pkg": "About to install {target} system-wide. Confirm?",
}


# ─────────────────────────────────────────────────────────────────────────────
# Blocked plan store (in-memory + file backup)
# ─────────────────────────────────────────────────────────────────────────────

_ROOT         = Path(__file__).resolve().parent.parent
_BLOCKED_FILE = _ROOT / "data" / "blocked_plans.json"
_blocked_plans: Dict[str, dict] = {}
_blocked_lock  = threading.Lock()


def _save_blocked_plan(plan: ActionPlan) -> None:
    with _blocked_lock:
        _blocked_plans[plan.id] = {
            "id":          plan.id,
            "intent":      plan.intent,
            "action_type": plan.action_type,
            "args":        plan.args,
            "risk":        plan.risk.value,
            "steps":       plan.steps,
            "created_ts":  plan.created_ts,
        }
        try:
            _BLOCKED_FILE.parent.mkdir(exist_ok=True)
            _BLOCKED_FILE.write_text(json.dumps(_blocked_plans, indent=2))
        except Exception:
            pass


def _load_blocked_plan(plan_id: str) -> Optional[dict]:
    with _blocked_lock:
        if plan_id in _blocked_plans:
            return _blocked_plans[plan_id]
        try:
            data = json.loads(_BLOCKED_FILE.read_text())
            _blocked_plans.update(data)
            return data.get(plan_id)
        except Exception:
            return None


def _remove_blocked_plan(plan_id: str) -> None:
    with _blocked_lock:
        _blocked_plans.pop(plan_id, None)
        try:
            _BLOCKED_FILE.write_text(json.dumps(_blocked_plans, indent=2))
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Undo store
# ─────────────────────────────────────────────────────────────────────────────

_undo_store: Dict[str, dict] = {}   # token → {"action": ..., "data": ...}
_undo_lock  = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# AutoExecutor
# ─────────────────────────────────────────────────────────────────────────────

class AutoExecutor:
    """
    Autonomous action engine. Thread-safe. Designed for async SSE streaming.
    """

    def __init__(
        self,
        aria_components:    dict,
        engine=None,
        conversation_engine=None,
        task_queue=None,
    ):
        self._aria     = aria_components
        self._engine   = engine or aria_components.get("engine")
        self._conv     = conversation_engine
        self._tq       = task_queue
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    async def execute(
        self,
        query:      str,
        intent:     dict,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """
        Main execution entry. Yields SSE strings.
        Intent dict comes from ConversationEngine.parse_intent().
        """
        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        t_start     = time.time()
        action_type = intent.get("action_type", "question")
        risk_str    = intent.get("risk_hint", RiskLevel.SAFE.value)
        risk        = RiskLevel(risk_str) if risk_str in RiskLevel._value2member_map_ else RiskLevel.SAFE
        resolved    = intent.get("resolved_query", query)
        entities    = intent.get("entities", {})

        # ── Build plan ────────────────────────────────────────────────────────
        plan = self._build_plan(action_type, risk, resolved, entities)

        yield sse({
            "type":        "risk_check",
            "level":       plan.risk.value,
            "action":      plan.action_type,
            "plan_id":     plan.id,
        })

        # ── DANGEROUS: block and ask ──────────────────────────────────────────
        if plan.risk == RiskLevel.DANGEROUS:
            _save_blocked_plan(plan)
            if self._conv:
                self._conv.set_pending_plan(session_id, plan.id)

            block_msg = self._dangerous_message(plan)
            yield sse({
                "type":     "blocked",
                "plan_id":  plan.id,
                "action":   plan.action_type,
                "message":  block_msg,
            })
            yield sse({"type": "token", "text": block_msg})
            yield sse({"type": "done", "mode": "blocked", "plan_id": plan.id,
                       "ms": int((time.time() - t_start) * 1000)})
            return

        # ── CAUTION: announce then execute ────────────────────────────────────
        if plan.risk == RiskLevel.CAUTION:
            caution_msg = self._caution_message(plan)
            yield sse({"type": "action_start", "action": plan.action_type,
                       "message": caution_msg})
            yield sse({"type": "token", "text": caution_msg + "\n\n"})
            await asyncio.sleep(0)

        # ── Execute ───────────────────────────────────────────────────────────
        yield sse({"type": "action_start", "action": plan.action_type,
                   "args": {k: str(v)[:60] for k, v in plan.args.items()}})

        loop = asyncio.get_event_loop()
        # Stock analysis can take up to 120s on first run (yfinance + AI layer for 20 stocks)
        _timeout = 120.0 if plan.action_type in ("stock_analysis", "stock_predict", "stock_monitor", "story_tell") else 60.0
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._dispatch(plan)),
                timeout=_timeout,
            )
        except asyncio.TimeoutError:
            result = {"ok": False, "error": f"Execution timed out after {_timeout:.0f}s", "output": ""}
        except Exception as e:
            result = {"ok": False, "error": str(e), "output": ""}

        # ── Self-heal on failure ──────────────────────────────────────────────
        if not result.get("ok") and plan.action_type not in ("question",):
            yield sse({"type": "status", "text": "Trying alternative approach…"})
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._fallback(plan, result)),
                timeout=30.0,
            )

        ms     = int((time.time() - t_start) * 1000)
        output = result.get("output", result.get("result", ""))

        # ── Register undo token ───────────────────────────────────────────────
        undo_token = None
        if plan.reversible and result.get("ok") and plan.action_type in (
            "open_app", "write_file", "set_volume", "browser_nav", "set_reminder"
        ):
            undo_token = self._register_undo(plan, result)
            if self._conv:
                self._conv.push_undo_token(session_id, undo_token)

        # ── JARVIS response ───────────────────────────────────────────────────
        jarvis = self._jarvis(plan, result)

        yield sse({"type": "action_done", "action": plan.action_type,
                   "ok": result.get("ok", False), "ms": ms})

        full_text = jarvis
        if output and output != jarvis and len(output) < 3000:
            full_text = f"{jarvis}\n\n{output}"

        yield sse({"type": "replace", "text": ""})
        # Stream in chunks for smooth UI
        for chunk in _chunk(full_text, size=120):
            yield sse({"type": "token", "text": chunk})
            await asyncio.sleep(0)

        # ── Proactive suggestion ──────────────────────────────────────────────
        if self._conv:
            suggestion = self._conv.suggest_next_step(plan.action_type, result)
            if suggestion:
                yield sse({"type": "suggestion", "text": suggestion})

            # Update conversation state
            self._conv.update_session_context(
                session_id,
                user_text=query,
                aria_text=full_text,
                action_type=plan.action_type,
                action_args=plan.args,
                undo_token=undo_token,
                pending_plan=None,
            )

        yield sse({
            "type":       "done",
            "mode":       "auto_exec",
            "action":     plan.action_type,
            "risk":       plan.risk.value,
            "ok":         result.get("ok", False),
            "ms":         ms,
            "undo_token": undo_token,
            "text":       full_text,
        })

    async def confirm_and_execute(
        self,
        plan_id:    str,
        session_id: str = "default",
    ) -> AsyncGenerator[str, None]:
        """Unblock and execute a previously blocked DANGEROUS plan."""
        def sse(obj: dict) -> str:
            return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

        plan_data = _load_blocked_plan(plan_id)
        if not plan_data:
            yield sse({"type": "error", "text": "Plan not found or already executed."})
            yield sse({"type": "done", "mode": "confirm_error"})
            return

        _remove_blocked_plan(plan_id)
        if self._conv:
            self._conv.set_pending_plan(session_id, None)

        # Rebuild plan from stored data
        plan = ActionPlan(
            id=plan_data["id"],
            intent=plan_data["intent"],
            action_type=plan_data["action_type"],
            args=plan_data["args"],
            risk=RiskLevel.CAUTION,   # downgrade: user confirmed
            steps=plan_data.get("steps", []),
        )

        yield sse({"type": "status", "text": f"Confirmed — executing {plan.intent}…"})

        t_start = time.time()
        loop    = asyncio.get_event_loop()
        result  = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: self._dispatch(plan)),
            timeout=60.0,
        )

        jarvis = self._jarvis(plan, result)
        output = result.get("output", result.get("result", ""))
        full   = jarvis + ("\n\n" + output if output and output != jarvis else "")

        for chunk in _chunk(full, size=120):
            yield sse({"type": "token", "text": chunk})
            await asyncio.sleep(0)

        yield sse({
            "type":   "done",
            "mode":   "confirmed_exec",
            "action": plan.action_type,
            "ok":     result.get("ok", False),
            "ms":     int((time.time() - t_start) * 1000),
            "text":   full,
        })

    def undo(self, undo_token: str) -> dict:
        """Reverse last reversible action."""
        with _undo_lock:
            entry = _undo_store.get(undo_token)
        if not entry:
            return {"ok": False, "message": "No undo record found for that action."}

        action = entry.get("action")
        data   = entry.get("data", {})

        try:
            if action == "open_app":
                agent = self._aria.get("desktop")
                if agent and hasattr(agent, "close_application"):
                    agent.close_application(data.get("app", ""))
                return {"ok": True, "message": f"{data.get('app', 'App')} closed."}

            elif action == "write_file":
                path = Path(data.get("path", ""))
                prev = data.get("previous_content")
                if prev is None and path.exists():
                    path.unlink()
                    return {"ok": True, "message": f"{path.name} deleted (was just created)."}
                elif prev is not None:
                    path.write_text(prev, encoding="utf-8")
                    return {"ok": True, "message": f"{path.name} restored to previous content."}

            elif action == "set_volume":
                agent = self._aria.get("sys_agent")
                prev  = data.get("previous_volume", 50)
                if agent and hasattr(agent, "set_volume"):
                    agent.set_volume(prev)
                return {"ok": True, "message": f"Volume restored to {prev}%."}

            elif action == "browser_nav":
                agent = self._aria.get("browser")
                if agent and hasattr(agent, "go_back"):
                    agent.go_back()
                return {"ok": True, "message": "Navigated back."}

        except Exception as e:
            return {"ok": False, "message": f"Undo failed: {e}"}

        return {"ok": False, "message": "Undo not supported for this action."}

    # ── Plan building ─────────────────────────────────────────────────────────

    def _build_plan(
        self,
        action_type: str,
        risk:        RiskLevel,
        query:       str,
        entities:    dict,
    ) -> ActionPlan:
        """Convert action_type + entities into an ActionPlan with task steps."""
        args: Dict[str, Any] = {}

        if action_type == "open_app":
            args["app"] = (
                entities.get("app") or
                self._extract_app_from_query(query) or "unknown"
            )
        elif action_type in ("close_app",):
            args["app"] = entities.get("app") or self._extract_app_from_query(query) or "unknown"
        elif action_type in ("read_file", "write_file", "delete_file"):
            args["file"] = entities.get("file") or self._extract_path(query) or "output.txt"
            if action_type == "write_file":
                args["content"] = self._extract_content(query)
        elif action_type == "list_dir":
            args["folder"] = entities.get("folder") or self._extract_folder(query) or "."
        elif action_type == "set_volume":
            args["value"] = entities.get("number") or self._extract_number(query) or 50
        elif action_type == "run_command":
            args["command"] = entities.get("command") or self._extract_command(query) or query
        elif action_type in ("web_search", "browser_nav"):
            args["query"] = query
            args["url"]   = entities.get("url") or ""
        elif action_type == "browser_nav":
            args["url"] = entities.get("url") or f"https://duckduckgo.com/?q={query}"
        elif action_type == "draft_email":
            args["to"]      = entities.get("email", "")
            args["subject"] = self._extract_subject(query)
            args["body"]    = query
        elif action_type == "send_email":
            args["to"] = entities.get("email", "")
        elif action_type == "set_reminder":
            args["time"] = entities.get("time", "soon")
            args["text"] = query
        elif action_type == "stock_analysis":
            args["market"] = self._extract_market(query)
        elif action_type == "stock_predict":
            args["ticker"] = self._extract_ticker(query)
            args["query"]  = query
        elif action_type == "stock_monitor":
            args["ticker"]    = self._extract_ticker(query)
            args["query"]     = query
            args["threshold"] = 1.0  # default 1% move alert
        elif action_type == "story_tell":
            # Extract URL if present
            _url_m = re.search(r"https?://\S+", query)
            args["url"]   = _url_m.group(0).rstrip(".,!?)") if _url_m else ""
            args["query"] = query
            # Detect tone keyword from query
            args["tone"]  = self._extract_story_tone(query)
        elif action_type == "translate":
            args["text"]     = query
            args["language"] = self._extract_language(query)
        elif action_type == "grammar_correct":
            args["text"] = query
        elif action_type == "os_control":
            args["action"] = self._extract_os_action(query)
        elif action_type == "screenshot":
            args["path"] = str(Path.home() / "Pictures" / f"aria_screenshot_{int(time.time())}.png")

        # Build TaskQueue-compatible steps
        steps = self._build_steps(action_type, args)

        return ActionPlan(
            intent=f"{action_type.replace('_', ' ')}: {query[:60]}",
            action_type=action_type,
            args=args,
            risk=risk,
            reversible=action_type in ("open_app", "write_file", "set_volume", "browser_nav", "set_reminder"),
            steps=steps,
        )

    def _build_steps(self, action_type: str, args: dict) -> List[dict]:
        """Map action_type to TaskQueue step format."""
        tool_map = {
            "open_app":       ("desktop.open_app",   args),
            "close_app":      ("desktop.close_app",  args),
            "screenshot":     ("desktop.screenshot", {}),
            "read_file":      ("desktop.read_file",  args),
            "write_file":     ("desktop.write_file", args),
            "list_dir":       ("desktop.list_dir",   args),
            "run_command":    ("terminal.run",        {"command": args.get("command", "")}),
            "set_volume":     ("system.set_volume",   args),
            "get_sysinfo":    ("system.sysinfo",      {}),
            "list_processes": ("terminal.run",        {"command": "tasklist" if os.name == "nt" else "ps aux"}),
            "media_control":  ("media.control",       args),
            "network_check":  ("network.ping",        {"host": "8.8.8.8"}),
            "speed_test":     ("network.ping",        {"host": "8.8.8.8"}),
            "browser_nav":    ("browser.navigate",    {"url": args.get("url", "")}),
            "web_search":     ("browser.search",      {"query": args.get("query", "")}),
            "stock_analysis": ("stock.top10",         {"market": args.get("market", "us")}),
            "stock_predict":  ("stock.predict",        {"ticker": args.get("ticker", ""), "query": args.get("query", "")}),
            "stock_monitor":  ("stock.monitor",        {"ticker": args.get("ticker", ""), "threshold": args.get("threshold", 1.0)}),
            "story_tell":     ("story.narrate",        {"query": args.get("query", ""), "url": args.get("url", ""), "tone": args.get("tone", "engaging")}),
            "translate":      ("translate",           args),
        }
        if action_type in tool_map:
            tool, step_args = tool_map[action_type]
            return [{"tool": tool, "args": step_args, "on_fail": "abort"}]
        # Default: LLM answer for unmatched types
        return [{"tool": "llm", "args": {"prompt": args.get("query", ""), "system": "Answer concisely."}}]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def _dispatch(self, plan: ActionPlan) -> dict:
        """Execute plan synchronously. Called in thread executor."""
        if self._tq:
            # Use TaskQueue for execution (single step)
            return self._tq._execute_step(plan.id, plan.steps[0] if plan.steps else {})

        # Inline fallback if TaskQueue not available
        return self._inline_dispatch(plan)

    def _inline_dispatch(self, plan: ActionPlan) -> dict:
        """Direct dispatch without TaskQueue."""
        aria = self._aria
        args = plan.args
        at   = plan.action_type

        try:
            if at == "open_app":
                agent = aria.get("desktop")
                if agent and hasattr(agent, "open_application"):
                    res = agent.open_application(args.get("app", ""))
                    return {"ok": True, "output": str(res)}

            elif at == "screenshot":
                agent = aria.get("desktop")
                if agent and hasattr(agent, "take_screenshot"):
                    path = agent.take_screenshot()
                    return {"ok": True, "output": str(path), "path": str(path)}

            elif at == "get_sysinfo":
                agent = aria.get("sys_agent")
                if agent and hasattr(agent, "get_system_info"):
                    res = agent.get_system_info()
                    return {"ok": True, "output": str(res)}

            elif at == "run_command":
                import subprocess
                cmd = args.get("command", "")
                r   = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
                return {
                    "ok":     r.returncode == 0,
                    "output": (r.stdout or r.stderr)[:2000],
                    "status": "ok" if r.returncode == 0 else "error",
                }

            elif at == "set_volume":
                agent = aria.get("sys_agent")
                if agent and hasattr(agent, "set_volume"):
                    agent.set_volume(int(args.get("value", 50)))
                    return {"ok": True, "output": f"Volume set to {args.get('value', 50)}%"}

            elif at in ("web_search", "browser_nav"):
                agent = aria.get("browser")
                url   = args.get("url") or f"https://duckduckgo.com/?q={args.get('query','').replace(' ','+')}"
                if agent and hasattr(agent, "open_url"):
                    agent.open_url(url)
                    return {"ok": True, "output": f"Opened: {url}"}

            elif at == "stock_analysis":
                agent = aria.get("quantum_stock")
                if agent and hasattr(agent, "find_top10"):
                    report = agent.find_top10(market=args.get("market", "us"))
                    return {"ok": True, "output": report.render() if hasattr(report, "render") else str(report)}

            elif at == "stock_predict":
                agent = aria.get("stock_predictor")
                ticker = args.get("ticker", "")
                if agent and hasattr(agent, "predict_nl"):
                    if ticker:
                        output = agent.predict_nl(ticker)
                    else:
                        output = agent.predict_nl(args.get("query", ""))
                    return {"ok": True, "output": output}
                return {"ok": False, "output": "StockPredictionAgent not available", "error": "not_loaded"}

            elif at == "stock_monitor":
                agent = aria.get("stock_predictor")
                ticker = args.get("ticker", "")
                if agent and hasattr(agent, "start_monitor") and ticker:
                    # Fire-and-forget monitor; alerts delivered via SSE notifications
                    def _on_alert(alert):
                        try:
                            from system.notifications import notification_manager
                            notification_manager.notify(
                                f"ARIA Stock Alert — {alert.ticker}",
                                alert.message,
                                "stock_alert",
                            )
                        except Exception:
                            pass
                    agent.start_monitor(ticker, callback=_on_alert, threshold_pct=args.get("threshold", 1.0))
                    return {"ok": True, "output": f"Now monitoring **{ticker.upper()}** — you'll be alerted on moves ≥{args.get('threshold',1.0):.1f}%."}
                return {"ok": False, "output": "StockPredictionAgent not available or no ticker found.", "error": "not_loaded"}

            elif at == "story_tell":
                agent = aria.get("story_agent")
                if agent:
                    query_text = args.get("query", plan.intent)
                    url        = args.get("url", "")
                    tone       = args.get("tone", "engaging")
                    # Extract URL from query if not pre-parsed
                    if not url:
                        url_m = re.search(r"https?://\S+", plan.intent)
                        if url_m:
                            url = url_m.group(0).rstrip(".,!?)")
                    chunks: list[str] = []
                    try:
                        import asyncio as _asyncio
                        _loop = _asyncio.new_event_loop()
                        async def _collect_story():
                            if url:
                                async for c in agent.narrate_url(url, tone=tone):
                                    chunks.append(c)
                            else:
                                async for c in agent.run_nl(query_text):
                                    chunks.append(c)
                        _loop.run_until_complete(_collect_story())
                        _loop.close()
                    except Exception as _se:
                        return {"ok": False, "output": f"Story narration error: {_se}"}
                    return {"ok": True, "output": "".join(chunks)}
                return {"ok": False, "output": "StoryAgent not available.", "error": "not_loaded"}

            elif at == "translate":
                agent = aria.get("trust_language")
                if agent and hasattr(agent, "lang_agent"):
                    res = agent.lang_agent.translate(args.get("text", ""), args.get("language", "english"))
                    return {"ok": True, "output": res}

            elif at == "draft_email":
                engine = self._engine
                if engine:
                    prompt = f"Write a professional email to {args.get('to','[recipient]')} about: {args.get('body','')}"
                    draft  = engine.generate(prompt, system="Write a concise professional email.", temperature=0.5)
                    return {"ok": True, "output": draft}

            elif at == "grammar_correct":
                agent = aria.get("trust_language")
                if agent and hasattr(agent, "lang_agent"):
                    res = agent.lang_agent.correct_grammar(args.get("text", ""))
                    return {"ok": True, "output": str(res)}

            elif at == "set_reminder":
                if self._tq:
                    self._tq.enqueue_notify(
                        title="ARIA Reminder",
                        message=args.get("text", "Reminder"),
                    )
                return {"ok": True, "output": f"Reminder set for {args.get('time', 'soon')}"}

            # LLM fallback for question / unknown
            if self._engine:
                result = self._engine.generate(
                    plan.intent,
                    system="I am ARIA. Answer the user's query directly and concisely.",
                    temperature=0.4,
                )
                return {"ok": True, "output": result or ""}

        except Exception as e:
            return {"ok": False, "error": str(e), "output": f"Error: {e}"}

        return {"ok": False, "error": "No handler found", "output": ""}

    def _fallback(self, plan: ActionPlan, first_result: dict) -> dict:
        """Try alternative strategy when primary execution fails."""
        try:
            # For run_command failures: try with LLM error-fix
            if plan.action_type == "run_command" and self._engine:
                fix_prompt = (
                    f"The command '{plan.args.get('command','')}' failed with error: "
                    f"{first_result.get('error','unknown')}. "
                    "Suggest the corrected command and explain the fix."
                )
                fixed = self._engine.generate(fix_prompt, temperature=0.2)
                return {"ok": True, "output": fixed or first_result.get("output", ""), "self_healed": True}

            # For browser failures: try DDG search instead
            if plan.action_type in ("browser_nav", "web_search"):
                try:
                    from ddgs import DDGS
                    with DDGS() as ddg:
                        results = list(ddg.text(plan.args.get("query", plan.intent), max_results=3))
                    snippets = [f"**{r.get('title','')}**: {r.get('body',r.get('snippet',''))[:150]}" for r in results]
                    return {"ok": True, "output": "\n\n".join(snippets)}
                except Exception:
                    pass

            # Generic: LLM answer
            if self._engine:
                ans = self._engine.generate(plan.intent, temperature=0.4)
                return {"ok": True, "output": ans or "", "self_healed": True}
        except Exception:
            pass

        return first_result

    # ── Undo registration ──────────────────────────────────────────────────────

    def _register_undo(self, plan: ActionPlan, result: dict) -> str:
        token = f"{plan.action_type}:{str(uuid.uuid4())[:8]}"
        data  = dict(plan.args)
        data.update(result)
        with _undo_lock:
            _undo_store[token] = {"action": plan.action_type, "data": data}
        return token

    # ── JARVIS messaging ──────────────────────────────────────────────────────

    def _jarvis(self, plan: ActionPlan, result: dict) -> str:
        templates = _JARVIS_TEMPLATES.get(plan.action_type, ["Done."])
        import random
        tpl = random.choice(templates)
        target = (
            plan.args.get("app") or plan.args.get("file") or plan.args.get("url") or
            plan.args.get("command", "")[:30] or plan.intent[:40]
        )
        try:
            return tpl.format(
                target=target,
                path=plan.args.get("path", plan.args.get("file", "file")),
                value=plan.args.get("value", ""),
                action=plan.args.get("action", plan.action_type),
            )
        except Exception:
            return templates[0]

    def _caution_message(self, plan: ActionPlan) -> str:
        tpl    = _CAUTION_PREFIX.get(plan.action_type, "On it…")
        target = plan.args.get("app") or plan.args.get("file") or plan.args.get("command", "")[:30]
        try:
            return tpl.format(target=target)
        except Exception:
            return tpl

    def _dangerous_message(self, plan: ActionPlan) -> str:
        tpl    = _DANGEROUS_PREFIX.get(plan.action_type, "This action requires confirmation. Proceed?")
        target = plan.args.get("app") or plan.args.get("file") or plan.args.get("command", "")[:30]
        action = plan.args.get("action", plan.action_type.replace("_", " "))
        try:
            return tpl.format(target=target or "the item", action=action)
        except Exception:
            return tpl

    # ── Entity extraction helpers ─────────────────────────────────────────────

    def _extract_app_from_query(self, query: str) -> str:
        q = query.lower()
        known = [
            "chrome", "firefox", "edge", "notepad", "excel", "word",
            "vscode", "terminal", "powershell", "cmd", "explorer",
            "spotify", "vlc", "discord", "slack", "zoom", "calculator",
            "paint", "brave", "outlook", "teams",
        ]
        for app in known:
            if app in q:
                return app
        # Grab last word after open/launch/start
        m = re.search(r"\b(?:open|launch|start)\s+(\w[\w\s]*?)(?:\s+(?:and|then|,)|\s*$)", query, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_path(self, query: str) -> str:
        m = re.search(r"([A-Za-z]:[\\\/][\w\\\/\.\-]+|\b[\w\-\.]+\.\w{1,5})", query)
        return m.group(0) if m else ""

    def _extract_folder(self, query: str) -> str:
        q = query.lower()
        for folder in ("downloads", "documents", "desktop", "pictures", "videos", "music", "temp"):
            if folder in q:
                return str(Path.home() / folder.capitalize())
        return "."

    def _extract_content(self, query: str) -> str:
        m = re.search(r'(?:with|containing|content[:\s]+)["\']?(.+)["\']?\s*$', query, re.IGNORECASE)
        return m.group(1).strip() if m else "Created by ARIA."

    def _extract_number(self, query: str) -> int:
        m = re.search(r"\b(\d{1,3})\b", query)
        return int(m.group(1)) if m else 50

    def _extract_command(self, query: str) -> str:
        m = re.search(r'(?:run|execute|command|bash|shell)[:\s]+["\']?(.+?)["\']?\s*$', query, re.IGNORECASE)
        return m.group(1).strip('"\'') if m else query

    def _extract_subject(self, query: str) -> str:
        m = re.search(r"about\s+(.+?)(?:\s+to\s+|\s*$)", query, re.IGNORECASE)
        return m.group(1).strip() if m else "Message"

    def _extract_ticker(self, query: str) -> str:
        """
        Pull a stock ticker or company name from a natural-language query.
        Delegates to StockPredictionAgent._extract_ticker when available,
        falls back to regex heuristics.
        """
        try:
            agent = self._aria.get("stock_predictor") if self._aria else None
            if agent and hasattr(agent, "_extract_ticker"):
                t = agent._extract_ticker(query)
                if t:
                    return t
        except Exception:
            pass
        # Regex fallback: look for ALL-CAPS 1-5 letter ticker
        m = re.search(r"\b([A-Z]{1,5})\b", query)
        if m:
            return m.group(1)
        return ""

    def _extract_market(self, query: str) -> str:
        q = query.lower()
        markets = {
            "india": ["india", "nifty", "sensex", "bse", "nse"],
            "us":    ["us", "usa", "american", "nasdaq", "s&p", "dow"],
            "uk":    ["uk", "ftse", "london", "british"],
            "germany": ["germany", "dax", "german"],
            "japan": ["japan", "nikkei", "japanese"],
        }
        for market, keywords in markets.items():
            if any(k in q for k in keywords):
                return market
        return "us"

    def _extract_language(self, query: str) -> str:
        langs = {
            "hindi": ["hindi", "हिंदी"], "french": ["french", "français"],
            "german": ["german", "deutsch"], "spanish": ["spanish", "español"],
            "japanese": ["japanese", "日本語"], "arabic": ["arabic", "عربي"],
            "portuguese": ["portuguese"], "russian": ["russian"],
            "chinese": ["chinese", "mandarin"], "korean": ["korean"],
        }
        q = query.lower()
        for lang, keywords in langs.items():
            if any(k in q for k in keywords):
                return lang
        return "english"

    def _extract_os_action(self, query: str) -> str:
        q = query.lower()
        for action in ("shutdown", "restart", "reboot", "hibernate", "sleep", "lock", "logout"):
            if action in q:
                return action
        return "shutdown"

    def _extract_story_tone(self, query: str) -> str:
        """Detect story narration tone from user query."""
        q = query.lower()
        tone_map = [
            (["humor", "humour", "funny", "laugh", "hilarious", "comedy"],     "humor"),
            (["sarcastic", "sarcasm", "dry", "ironic", "deadpan"],             "sarcastic"),
            (["epic", "legendary", "grand", "fantasy", "heroic"],              "epic"),
            (["thriller", "horror", "scary", "suspense", "dark", "mystery"],   "thriller"),
            (["romance", "romantic", "love"],                                  "romance"),
            (["children", "kids", "child", "bedtime", "fairy"],                "children"),
            (["educational", "learn", "history", "science"],                   "educational"),
            (["dramatic", "powerful", "intense", "emotional"],                 "dramatic"),
            (["casual", "chill", "simple", "friendly", "normal"],              "casual"),
        ]
        for keywords, tone in tone_map:
            if any(k in q for k in keywords):
                return tone
        return "engaging"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _chunk(text: str, size: int = 120):
    """Split text into chunks for smooth SSE streaming."""
    for i in range(0, len(text), size):
        yield text[i:i + size]
