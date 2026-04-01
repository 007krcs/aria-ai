"""
ARIA TaskQueue — Background Task Execution Engine
==================================================

Manages multi-step background tasks with:
  - SQLite persistence (survives restarts)
  - Priority scheduling (1=highest, 10=lowest)
  - Multi-step plan execution (sequential steps with variable passing)
  - SSE progress streaming per session
  - Retry logic (configurable max_retries per task)
  - Recurring / scheduled tasks (cron-like via interval_s)
  - JARVIS-style natural language status messages

Worker runs in a background daemon thread.
SSE push bridges thread→asyncio via run_coroutine_threadsafe.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
TQ_DB    = DATA_DIR / "task_queue.db"

# ── JARVIS templates ──────────────────────────────────────────────────────────
_DONE_TEMPLATES = {
    "single":     "Done — {title}.",
    "multi_step": "All {n} steps complete — {title} finished in {duration}s.",
    "scheduled":  "Scheduled task '{title}' completed. Next run in {next_in}.",
    "watch":      "Alert: {title} — {detail}.",
}

_STEP_LABELS = {
    "desktop.open_app":    "Opening {app}",
    "desktop.close_app":   "Closing {app}",
    "desktop.screenshot":  "Capturing screenshot",
    "desktop.read_file":   "Reading {file}",
    "desktop.write_file":  "Writing to {file}",
    "desktop.list_dir":    "Listing {folder}",
    "terminal.run":        "Running terminal command",
    "browser.navigate":    "Navigating to {url}",
    "browser.search":      "Searching for {query}",
    "system.set_volume":   "Setting volume to {value}%",
    "system.sysinfo":      "Reading system info",
    "media.control":       "Controlling media",
    "network.ping":        "Pinging {host}",
    "network.speedtest":   "Running speed test",
    "notify":              "Sending notification",
    "speak":               "Speaking result",
    "llm":                 "Thinking",
    "wait":                "Waiting {seconds}s",
    "stock.top10":         "Fetching top 10 stocks for {market}",
    "translate":           "Translating to {language}",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    id:           str
    session_id:   str
    plan_id:      str
    type:         str       # single | multi_step | scheduled | watch
    status:       str       # pending|running|done|failed|blocked|cancelled
    priority:     int
    title:        str
    steps:        List[dict]
    current_step: int       = 0
    result:       dict      = field(default_factory=dict)
    error_text:   str       = ""
    retry_count:  int       = 0
    max_retries:  int       = 2
    created_ts:   float     = field(default_factory=time.time)
    started_ts:   Optional[float] = None
    completed_ts: Optional[float] = None
    next_run_ts:  Optional[float] = None
    interval_s:   float     = 0
    metadata:     dict      = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# TaskQueue
# ─────────────────────────────────────────────────────────────────────────────

class TaskQueue:
    """
    Persistent background task executor.
    Call start() once at server init. All methods are thread-safe.
    """

    POLL_INTERVAL_S = 0.5

    def __init__(self, aria_components: dict, db_path: Path = TQ_DB):
        self._aria       = aria_components
        self._db_path    = db_path
        self._lock       = threading.Lock()
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sse_queues: Dict[str, asyncio.Queue] = {}   # session_id → queue
        self._step_vars: Dict[str, Dict[str, Any]] = {}   # task_id → var_store
        self._init_db()

    # ── DB setup ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id            TEXT PRIMARY KEY,
                    session_id    TEXT DEFAULT 'default',
                    plan_id       TEXT DEFAULT '',
                    type          TEXT NOT NULL DEFAULT 'single',
                    status        TEXT NOT NULL DEFAULT 'pending',
                    priority      INTEGER DEFAULT 5,
                    title         TEXT NOT NULL,
                    steps_json    TEXT DEFAULT '[]',
                    current_step  INTEGER DEFAULT 0,
                    result_json   TEXT DEFAULT '{}',
                    error_text    TEXT DEFAULT '',
                    retry_count   INTEGER DEFAULT 0,
                    max_retries   INTEGER DEFAULT 2,
                    created_ts    REAL NOT NULL,
                    started_ts    REAL,
                    completed_ts  REAL,
                    next_run_ts   REAL,
                    interval_s    REAL DEFAULT 0,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_tq_status
                    ON tasks(status, priority, created_ts);
                CREATE INDEX IF NOT EXISTS idx_tq_session
                    ON tasks(session_id, created_ts DESC);
                CREATE INDEX IF NOT EXISTS idx_tq_next_run
                    ON tasks(next_run_ts);
            """)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self._db_path), timeout=10, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _row_to_task(self, row) -> Task:
        return Task(
            id=row["id"],
            session_id=row["session_id"] or "default",
            plan_id=row["plan_id"] or "",
            type=row["type"],
            status=row["status"],
            priority=row["priority"],
            title=row["title"],
            steps=json.loads(row["steps_json"] or "[]"),
            current_step=row["current_step"],
            result=json.loads(row["result_json"] or "{}"),
            error_text=row["error_text"] or "",
            retry_count=row["retry_count"],
            max_retries=row["max_retries"],
            created_ts=row["created_ts"],
            started_ts=row["started_ts"],
            completed_ts=row["completed_ts"],
            next_run_ts=row["next_run_ts"],
            interval_s=row["interval_s"] or 0,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _save_task(self, t: Task, db=None) -> None:
        sql = """
            INSERT OR REPLACE INTO tasks
                (id, session_id, plan_id, type, status, priority, title,
                 steps_json, current_step, result_json, error_text,
                 retry_count, max_retries, created_ts, started_ts,
                 completed_ts, next_run_ts, interval_s, metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        vals = (
            t.id, t.session_id, t.plan_id, t.type, t.status, t.priority, t.title,
            json.dumps(t.steps), t.current_step, json.dumps(t.result), t.error_text,
            t.retry_count, t.max_retries, t.created_ts, t.started_ts,
            t.completed_ts, t.next_run_ts, t.interval_s, json.dumps(t.metadata),
        )
        if db:
            db.execute(sql, vals)
        else:
            with self._connect() as db2:
                db2.execute(sql, vals)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        # Grab event loop from calling context (server startup)
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()

        self._running = True
        self._thread  = threading.Thread(
            target=self._worker_loop, daemon=True, name="aria-taskqueue"
        )
        self._thread.start()
        print("  [OK] TaskQueue background worker started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(
        self,
        title:      str,
        steps:      List[dict],
        session_id: str   = "default",
        plan_id:    str   = "",
        priority:   int   = 5,
        task_type:  str   = "single",
        interval_s: float = 0,
        metadata:   dict  = None,
    ) -> str:
        """Add task to queue. Returns task_id."""
        task_id = str(uuid.uuid4())[:12]
        now     = time.time()
        t = Task(
            id=task_id, session_id=session_id, plan_id=plan_id,
            type=task_type, status="pending", priority=priority, title=title,
            steps=steps, created_ts=now,
            next_run_ts=now if task_type == "scheduled" else None,
            interval_s=interval_s,
            metadata=metadata or {},
        )
        self._save_task(t)
        return task_id

    def cancel(self, task_id: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                "UPDATE tasks SET status='cancelled' WHERE id=? AND status IN ('pending','running')",
                (task_id,)
            )
            return cur.rowcount > 0

    def get_status(self, task_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if not row:
            return {}
        t = self._row_to_task(row)
        return {
            "id":           t.id,
            "title":        t.title,
            "status":       t.status,
            "current_step": t.current_step,
            "total_steps":  len(t.steps),
            "pct":          int(t.current_step / max(len(t.steps), 1) * 100),
            "result":       t.result,
            "error":        t.error_text,
            "started_ts":   t.started_ts,
            "completed_ts": t.completed_ts,
        }

    def list_tasks(self, session_id: str = "", status: str = "") -> List[dict]:
        with self._connect() as db:
            if session_id and status:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE session_id=? AND status=? ORDER BY priority,created_ts DESC LIMIT 50",
                    (session_id, status)
                ).fetchall()
            elif session_id:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE session_id=? ORDER BY created_ts DESC LIMIT 50",
                    (session_id,)
                ).fetchall()
            elif status:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE status=? ORDER BY priority,created_ts DESC LIMIT 50",
                    (status,)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM tasks ORDER BY created_ts DESC LIMIT 50"
                ).fetchall()
        return [self.get_status(r["id"]) for r in rows]

    # ── SSE streaming ─────────────────────────────────────────────────────────

    def subscribe_session(self, session_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._sse_queues[session_id] = q
        return q

    def unsubscribe_session(self, session_id: str) -> None:
        self._sse_queues.pop(session_id, None)

    async def stream_task(self, task_id: str) -> AsyncGenerator[str, None]:
        """Stream SSE events for a single task until completion."""
        deadline = time.time() + 300   # 5-min max

        while time.time() < deadline:
            status = self.get_status(task_id)
            if not status:
                yield f"data: {json.dumps({'type':'error','error':'Task not found'})}\n\n"
                return

            yield f"data: {json.dumps({'type':'task_progress', **status})}\n\n"

            if status["status"] in ("done", "failed", "cancelled"):
                break
            await asyncio.sleep(0.4)

        yield f"data: {json.dumps({'type':'task_stream_end','task_id':task_id})}\n\n"

    def _push_sse(self, session_id: str, event: dict) -> None:
        """Thread-safe SSE push to session queue."""
        q = self._sse_queues.get(session_id)
        if not q or not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(q.put(event), self._loop)
        except Exception:
            pass

    # ── Worker loop ───────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        while self._running:
            try:
                self._process_one()
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL_S)

    def _process_one(self) -> None:
        now = time.time()
        with self._connect() as db:
            # Pick highest-priority pending task (or due scheduled task)
            row = db.execute("""
                SELECT * FROM tasks
                WHERE status = 'pending'
                  AND (next_run_ts IS NULL OR next_run_ts <= ?)
                ORDER BY priority ASC, created_ts ASC
                LIMIT 1
            """, (now,)).fetchone()

            if not row:
                return

            t = self._row_to_task(row)
            t.status     = "running"
            t.started_ts = now
            self._save_task(t, db)

        self._push_sse(t.session_id, {
            "type":    "task_started",
            "task_id": t.id,
            "title":   t.title,
            "steps":   len(t.steps),
        })
        self._step_vars[t.id] = {}

        all_ok    = True
        last_result = {}
        t_start   = now

        for i, step in enumerate(t.steps):
            if not self._running:
                break

            # Check cancelled
            with self._connect() as db:
                row = db.execute("SELECT status FROM tasks WHERE id=?", (t.id,)).fetchone()
                if row and row["status"] == "cancelled":
                    return

            desc = self._render_step_label(step)
            self._push_sse(t.session_id, {
                "type":    "step_start",
                "task_id": t.id,
                "step":    i + 1,
                "total":   len(t.steps),
                "desc":    desc,
            })

            result = self._execute_step(t.id, step)
            last_result = result

            if result.get("ok"):
                # Store output variable if specified
                if step.get("output_var"):
                    self._step_vars[t.id][step["output_var"]] = result.get("result", "")

                self._push_sse(t.session_id, {
                    "type":    "step_done",
                    "task_id": t.id,
                    "step":    i + 1,
                    "result":  str(result.get("result", ""))[:200],
                    "pct":     int((i + 1) / len(t.steps) * 100),
                })

                # Update step pointer
                with self._connect() as db:
                    db.execute("UPDATE tasks SET current_step=? WHERE id=?", (i + 1, t.id))
            else:
                on_fail = step.get("on_fail", "abort")
                if on_fail == "skip":
                    continue
                elif on_fail == "retry" and t.retry_count < t.max_retries:
                    t.retry_count += 1
                    result2 = self._execute_step(t.id, step)
                    if result2.get("ok"):
                        last_result = result2
                        continue
                all_ok = False
                t.error_text = result.get("error", "Unknown error")
                break

        # ── Finalise ──────────────────────────────────────────────────────────
        duration = round(time.time() - t_start, 1)
        t.completed_ts = time.time()
        t.result       = last_result

        if all_ok:
            t.status  = "done"
            jarvis    = self._jarvis_done(t, duration)
            self._push_sse(t.session_id, {
                "type":    "task_done",
                "task_id": t.id,
                "title":   t.title,
                "summary": jarvis,
                "result":  last_result,
                "ms":      int(duration * 1000),
            })
        else:
            t.status = "failed"
            self._push_sse(t.session_id, {
                "type":    "task_failed",
                "task_id": t.id,
                "title":   t.title,
                "error":   t.error_text,
            })

        # Handle recurring tasks
        if t.interval_s > 0 and t.status == "done":
            t.status     = "pending"
            t.next_run_ts = time.time() + t.interval_s
            t.current_step = 0
            t.result      = {}
            t.started_ts  = None
            t.completed_ts = None

        self._step_vars.pop(t.id, None)
        self._save_task(t)

    # ── Step execution ────────────────────────────────────────────────────────

    def _execute_step(self, task_id: str, step: dict) -> dict:
        """Dispatch one step to the correct aria agent."""
        tool    = step.get("tool", "")
        args    = dict(step.get("args", {}))
        timeout = step.get("timeout", 30)

        # Substitute {variable} references from previous step outputs
        vars_store = self._step_vars.get(task_id, {})
        for k, v in args.items():
            if isinstance(v, str):
                for var_name, var_val in vars_store.items():
                    args[k] = args[k].replace(f"{{{var_name}}}", str(var_val))

        try:
            return self._dispatch(tool, args, timeout)
        except Exception as e:
            return {"ok": False, "error": str(e), "result": ""}

    def _dispatch(self, tool: str, args: dict, timeout: int) -> dict:
        aria = self._aria

        # ── Desktop agent ──────────────────────────────────────────────────
        if tool == "desktop.open_app":
            agent = aria.get("desktop")
            if agent and hasattr(agent, "open_application"):
                res = agent.open_application(args.get("app", ""))
                return {"ok": True, "result": res}
            return {"ok": False, "error": "Desktop agent unavailable"}

        if tool == "desktop.close_app":
            agent = aria.get("desktop")
            if agent and hasattr(agent, "close_application"):
                res = agent.close_application(args.get("app", ""))
                return {"ok": True, "result": res}
            return {"ok": False, "error": "Desktop agent unavailable"}

        if tool == "desktop.screenshot":
            agent = aria.get("desktop")
            if agent and hasattr(agent, "take_screenshot"):
                path = agent.take_screenshot()
                return {"ok": True, "result": path}
            return {"ok": False, "error": "Desktop agent unavailable"}

        if tool == "desktop.read_file":
            agent = aria.get("desktop")
            if agent and hasattr(agent, "read_file"):
                content = agent.read_file(args.get("file", ""))
                return {"ok": True, "result": content}
            # Fallback: direct read
            try:
                p = Path(args.get("file", ""))
                if p.exists():
                    return {"ok": True, "result": p.read_text(errors="replace")[:5000]}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        if tool == "desktop.write_file":
            try:
                p = Path(args.get("file", "output.txt"))
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args.get("content", ""), encoding="utf-8")
                return {"ok": True, "result": str(p)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        if tool == "desktop.list_dir":
            try:
                folder = Path(args.get("folder", "."))
                items  = [f.name for f in folder.iterdir()][:50]
                return {"ok": True, "result": "\n".join(items)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # ── Terminal ───────────────────────────────────────────────────────
        if tool == "terminal.run":
            agent = aria.get("terminal")
            cmd   = args.get("command", args.get("cmd", ""))
            if agent and hasattr(agent, "run_command"):
                res = agent.run_command(cmd)
                ok  = res.get("exit_code", 0) == 0 if isinstance(res, dict) else True
                out = res.get("output", str(res)) if isinstance(res, dict) else str(res)
                return {"ok": ok, "result": out, "status": "ok" if ok else "error"}
            # Fallback: subprocess
            import subprocess
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
                return {
                    "ok":     r.returncode == 0,
                    "result": (r.stdout or r.stderr)[:2000],
                    "status": "ok" if r.returncode == 0 else "error",
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # ── Browser ────────────────────────────────────────────────────────
        if tool == "browser.navigate":
            agent = aria.get("browser")
            if agent and hasattr(agent, "open_url"):
                res = agent.open_url(args.get("url", ""))
                return {"ok": True, "result": str(res)}
            return {"ok": False, "error": "Browser agent unavailable"}

        if tool == "browser.search":
            agent = aria.get("browser")
            q     = args.get("query", "")
            url   = f"https://duckduckgo.com/?q={q.replace(' ', '+')}"
            if agent and hasattr(agent, "open_url"):
                agent.open_url(url)
                return {"ok": True, "result": f"Searched: {q}"}
            return {"ok": False, "error": "Browser agent unavailable"}

        # ── System ─────────────────────────────────────────────────────────
        if tool == "system.set_volume":
            agent = aria.get("sys_agent")
            if agent and hasattr(agent, "set_volume"):
                res = agent.set_volume(int(args.get("value", 50)))
                return {"ok": True, "result": str(res)}
            return {"ok": False, "error": "System agent unavailable"}

        if tool == "system.sysinfo":
            agent = aria.get("sys_agent")
            if agent and hasattr(agent, "get_system_info"):
                res = agent.get_system_info()
                return {"ok": True, "result": str(res)}
            return {"ok": False, "error": "System agent unavailable"}

        # ── Media ──────────────────────────────────────────────────────────
        if tool == "media.control":
            agent  = aria.get("media")
            action = args.get("action", "play")
            if agent and hasattr(agent, "control"):
                res = agent.control(action)
                return {"ok": True, "result": str(res)}
            return {"ok": False, "error": "Media agent unavailable"}

        # ── Network ────────────────────────────────────────────────────────
        if tool == "network.ping":
            agent = aria.get("network")
            if agent and hasattr(agent, "ping"):
                res = agent.ping(args.get("host", "8.8.8.8"))
                return {"ok": True, "result": str(res)}
            import subprocess
            host = args.get("host", "8.8.8.8")
            r = subprocess.run(f"ping -n 2 {host}", shell=True, capture_output=True, text=True, timeout=10)
            return {"ok": r.returncode == 0, "result": r.stdout[:500]}

        # ── Notify ─────────────────────────────────────────────────────────
        if tool == "notify":
            try:
                from system.notifications import notification_manager
                notification_manager.send(args.get("title", "ARIA"), args.get("message", ""))
                return {"ok": True, "result": "Notification sent"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # ── Speak ──────────────────────────────────────────────────────────
        if tool == "speak":
            agent = aria.get("voice")
            text  = args.get("text", "")
            if agent and hasattr(agent, "speak"):
                agent.speak(text)
                return {"ok": True, "result": f"Spoke: {text[:50]}"}
            return {"ok": True, "result": "Voice unavailable (text delivered)"}

        # ── Stock ──────────────────────────────────────────────────────────
        if tool == "stock.top10":
            agent  = aria.get("quantum_stock")
            market = args.get("market", "us")
            if agent and hasattr(agent, "find_top10"):
                report = agent.find_top10(market=market)
                return {"ok": True, "result": report.render() if hasattr(report, "render") else str(report)}
            return {"ok": False, "error": "QuantumStockAgent unavailable"}

        # ── Translate ──────────────────────────────────────────────────────
        if tool == "translate":
            agent = aria.get("trust_language")
            text  = args.get("text", "")
            lang  = args.get("language", "english")
            if agent and hasattr(agent, "lang_agent"):
                res = agent.lang_agent.translate(text, lang)
                return {"ok": True, "result": res}
            return {"ok": False, "error": "Translation agent unavailable"}

        # ── LLM call ───────────────────────────────────────────────────────
        if tool == "llm":
            engine = self._aria.get("engine")
            prompt = args.get("prompt", "")
            system = args.get("system", "I am ARIA. Answer concisely.")
            if engine and hasattr(engine, "generate"):
                result = engine.generate(prompt, system=system, temperature=0.4)
                return {"ok": True, "result": result or ""}
            return {"ok": False, "error": "Engine unavailable"}

        # ── Wait ───────────────────────────────────────────────────────────
        if tool == "wait":
            seconds = float(args.get("seconds", 1))
            time.sleep(min(seconds, 60))
            return {"ok": True, "result": f"Waited {seconds}s"}

        return {"ok": False, "error": f"Unknown tool: {tool}"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _render_step_label(self, step: dict) -> str:
        tool = step.get("tool", "")
        args = step.get("args", {})
        desc = step.get("desc", "")
        if desc:
            return desc
        template = _STEP_LABELS.get(tool, f"Running {tool}")
        try:
            return template.format(**args)
        except Exception:
            return template

    def _jarvis_done(self, t: Task, duration: float) -> str:
        tpl = _DONE_TEMPLATES.get(t.type, "Done — {title}.")
        try:
            return tpl.format(
                title=t.title,
                n=len(t.steps),
                duration=duration,
                next_in=f"{t.interval_s:.0f}s" if t.interval_s else "N/A",
                detail=str(t.result.get("result", ""))[:80],
            )
        except Exception:
            return f"Done — {t.title}."

    # ── Convenience builders ─────────────────────────────────────────────────

    def enqueue_open_app(self, app: str, session_id: str = "default") -> str:
        return self.enqueue(
            title=f"Open {app}",
            steps=[{"tool": "desktop.open_app", "args": {"app": app}, "desc": f"Opening {app}"}],
            session_id=session_id,
            priority=2,
        )

    def enqueue_run_command(self, cmd: str, session_id: str = "default") -> str:
        return self.enqueue(
            title=f"Run: {cmd[:40]}",
            steps=[{"tool": "terminal.run", "args": {"command": cmd}, "on_fail": "abort"}],
            session_id=session_id,
            priority=3,
        )

    def enqueue_notify(self, title: str, message: str, session_id: str = "default") -> str:
        return self.enqueue(
            title=f"Notify: {title}",
            steps=[{"tool": "notify", "args": {"title": title, "message": message}}],
            session_id=session_id,
            priority=1,
        )

    def enqueue_speak(self, text: str, session_id: str = "default") -> str:
        return self.enqueue(
            title=f"Speak: {text[:30]}",
            steps=[{"tool": "speak", "args": {"text": text}}],
            session_id=session_id,
            priority=1,
        )
