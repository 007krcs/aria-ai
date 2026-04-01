"""
ARIA — Performance logger
Every query, every response, every agent run is logged here.
This SQLite database is the source of truth for self-adaptation.
Failures here become training data. Patterns here trigger new agents.
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from core.config import LOG_DB_PATH


class Logger:
    """
    SQLite-based logger. Tracks every interaction and agent performance.
    The self-adaptation engine reads this to detect failure patterns.
    """

    def __init__(self):
        self.db_path = LOG_DB_PATH
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT    NOT NULL,
                    query       TEXT    NOT NULL,
                    response    TEXT    NOT NULL,
                    agent_used  TEXT    NOT NULL,
                    intent      TEXT,
                    confidence  REAL,
                    latency_ms  INTEGER,
                    success     INTEGER DEFAULT 1,
                    feedback    TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT    NOT NULL,
                    agent_name  TEXT    NOT NULL,
                    task        TEXT    NOT NULL,
                    result      TEXT,
                    score       REAL,
                    latency_ms  INTEGER,
                    error       TEXT
                );

                CREATE TABLE IF NOT EXISTS ingested_sources (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT    NOT NULL,
                    source      TEXT    UNIQUE NOT NULL,
                    source_type TEXT,
                    chunks      INTEGER,
                    domain      TEXT
                );

                CREATE TABLE IF NOT EXISTS agent_registry (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name  TEXT    UNIQUE NOT NULL,
                    domain      TEXT    NOT NULL,
                    model       TEXT    NOT NULL,
                    adapter_path TEXT,
                    created_at  TEXT,
                    avg_score   REAL    DEFAULT 0.5,
                    run_count   INTEGER DEFAULT 0,
                    active      INTEGER DEFAULT 1
                );
            """)

    # ── Log an interaction ────────────────────────────────────────────────────

    def log_interaction(
        self,
        query: str,
        response: str,
        agent_used: str,
        intent: str = "general",
        confidence: float = 0.5,
        latency_ms: int = 0,
        success: bool = True,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO interactions
                   (ts, query, response, agent_used, intent, confidence, latency_ms, success)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), query, response,
                 agent_used, intent, confidence, latency_ms, int(success))
            )
            return cur.lastrowid

    # ── Log an agent run ──────────────────────────────────────────────────────

    def log_agent_run(
        self,
        agent_name: str,
        task: str,
        result: str = "",
        score: float = 0.5,
        latency_ms: int = 0,
        error: str = "",
    ):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO agent_runs
                   (ts, agent_name, task, result, score, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), agent_name,
                 task, result, score, latency_ms, error)
            )

    # ── Log ingested source ────────────────────────────────────────────────────

    def log_ingestion(self, source: str, source_type: str, chunks: int, domain: str):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ingested_sources
                   (ts, source, source_type, chunks, domain)
                   VALUES (?, ?, ?, ?, ?)""",
                (datetime.now().isoformat(), source, source_type, chunks, domain)
            )

    # ── Register an agent ──────────────────────────────────────────────────────

    def register_agent(self, agent_name: str, domain: str, model: str, adapter_path: str = ""):
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO agent_registry
                   (agent_name, domain, model, adapter_path, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (agent_name, domain, model, adapter_path, datetime.now().isoformat())
            )

    # ── Get failure patterns ───────────────────────────────────────────────────

    def get_failure_patterns(self, hours: int = 48, min_count: int = 10) -> list[dict]:
        """
        Find intents/domains with consistently low confidence in recent hours.
        This is what triggers new agent spawning.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT intent,
                          AVG(confidence) as avg_conf,
                          COUNT(*) as total,
                          SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures
                   FROM interactions
                   WHERE ts > datetime('now', ?)
                   GROUP BY intent
                   HAVING AVG(confidence) < 0.55 AND COUNT(*) >= ?
                   ORDER BY avg_conf ASC""",
                (f"-{hours} hours", min_count)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Get failures for a domain ─────────────────────────────────────────────

    def get_failures_for_domain(self, intent: str, hours: int = 48) -> list[dict]:
        """Get actual failed queries for a domain — used to build training data."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT query, response, confidence FROM interactions
                   WHERE intent = ? AND ts > datetime('now', ?)
                   AND (confidence < 0.55 OR success = 0)
                   ORDER BY ts DESC LIMIT 200""",
                (intent, f"-{hours} hours")
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Stats dashboard ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._connect() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM interactions WHERE success=1").fetchone()[0]
            avg_c   = conn.execute("SELECT AVG(confidence) FROM interactions").fetchone()[0]
            sources = conn.execute("SELECT COUNT(*) FROM ingested_sources").fetchone()[0]
            agents  = conn.execute("SELECT COUNT(*) FROM agent_registry WHERE active=1").fetchone()[0]
        return {
            "total_interactions": total,
            "success_rate":       round((success / total * 100) if total else 0, 1),
            "avg_confidence":     round(avg_c or 0, 3),
            "ingested_sources":   sources,
            "active_agents":      agents,
        }
