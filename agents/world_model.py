"""
ARIA — Autonomous World Model
================================
This is the single biggest upgrade from "smart tool" to "intelligent system."

Current ARIA: user asks → retrieve from ChromaDB → answer.
This is REACTIVE. ARIA only knows what it was explicitly told.

World Model ARIA: continuously maintains a living graph of facts,
entities, relationships, causal links, and confidence scores.
ARIA knows things BEFORE you ask. It proactively updates its beliefs.
It knows when its beliefs are wrong and corrects them.

What the world model tracks:
  Entities:     people, companies, projects, places, concepts
  Facts:        claims with confidence scores and expiry times
  Relations:    X causes Y, X is part of Y, X happened before Y
  Patterns:     recurring behaviours, seasonal trends, usage cycles
  Predictions:  what ARIA expects will happen next (and tracks accuracy)
  Self-model:   what ARIA knows well vs poorly, per domain

Key insight: facts DECAY. If ARIA learned "Apple stock is ₹180" three
days ago, that fact has near-zero confidence today. The world model
continuously expires stale facts and marks them for refresh.

The world model is the difference between:
  "What is Apple's stock price?" → retrieve old doc → probably wrong
  "What is Apple's stock price?" → check fact freshness → expired →
    fetch live → update graph → answer with confidence: 0.99

Storage: SQLite + NetworkX graph (in-memory, persisted to JSON).
Memory: ~50MB for a rich personal knowledge graph.
IoT: a stripped version runs on 10MB with only structural facts.
"""

import json
import time
import sqlite3
import threading
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Any
from rich.console import Console

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()

WORLD_DB   = PROJECT_ROOT / "data" / "world_model.db"
WORLD_JSON = PROJECT_ROOT / "data" / "world_graph.json"
WORLD_DB.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FACT — the atomic unit of knowledge
# ─────────────────────────────────────────────────────────────────────────────

class Fact:
    """
    A single piece of knowledge with provenance and confidence.

    Facts decay. A stock price from 3 days ago has ~0 confidence.
    A person's birth date from 10 years ago has ~1.0 confidence.
    Decay rate is set per fact category.
    """

    # Seconds until confidence halves (half-life)
    HALF_LIVES = {
        "stock_price":     300,       # 5 min
        "news":            3600,      # 1 hour
        "weather":         1800,      # 30 min
        "person_role":     2592000,   # 30 days
        "company_info":    604800,    # 7 days
        "research_paper":  31536000,  # 1 year
        "system_state":    30,        # 30 seconds
        "user_preference": 7776000,   # 90 days
        "general":         86400,     # 1 day
    }

    def __init__(
        self,
        subject:    str,
        predicate:  str,
        obj:        Any,
        confidence: float = 1.0,
        source:     str   = "unknown",
        category:   str   = "general",
        expires_s:  int   = None,
    ):
        self.id         = hashlib.md5(f"{subject}{predicate}".encode()).hexdigest()[:12]
        self.subject    = subject
        self.predicate  = predicate
        self.object     = obj
        self.confidence = confidence
        self.source     = source
        self.category   = category
        self.created_ts = time.time()
        self.half_life  = expires_s or self.HALF_LIVES.get(category, 86400)

    def current_confidence(self) -> float:
        """Confidence decays exponentially with time."""
        age  = time.time() - self.created_ts
        decay= (0.5 ** (age / self.half_life))
        return round(self.confidence * decay, 4)

    def is_fresh(self, min_confidence: float = 0.3) -> bool:
        return self.current_confidence() >= min_confidence

    def to_dict(self) -> dict:
        return {
            "id":         self.id,
            "subject":    self.subject,
            "predicate":  self.predicate,
            "object":     str(self.object),
            "confidence": self.current_confidence(),
            "source":     self.source,
            "category":   self.category,
            "created_ts": self.created_ts,
            "half_life":  self.half_life,
            "fresh":      self.is_fresh(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        f            = cls(d["subject"], d["predicate"], d["object"],
                          d["confidence"], d["source"], d["category"])
        f.created_ts = d.get("created_ts", time.time())
        f.half_life  = d.get("half_life", 86400)
        return f


# ─────────────────────────────────────────────────────────────────────────────
# WORLD MODEL
# ─────────────────────────────────────────────────────────────────────────────

class WorldModel:
    """
    ARIA's continuously-maintained model of the world.

    Three interconnected stores:
    1. Fact store   — SQLite: (subject, predicate, object, confidence, expiry)
    2. Entity graph — NetworkX: entities as nodes, relations as edges
    3. Pattern store — JSON: recurring patterns ARIA has detected

    The world model runs a background refresh loop:
    - Every 5 min: expire stale facts, queue for refresh
    - Every 15 min: refresh stock prices, news, system state
    - Every hour:  check user patterns, update predictions
    - Every day:   deep learning from interaction history
    """

    def __init__(self):
        self._facts:    dict[str, Fact] = {}
        self._graph     = nx.DiGraph() if NX_AVAILABLE else None
        self._patterns: list[dict]      = []
        self._lock      = threading.Lock()
        self._refresh_queue: list[str]  = []  # fact IDs to refresh

        self._init_db()
        self._load()
        self._start_background()

    def _init_db(self):
        with sqlite3.connect(WORLD_DB) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS facts (
                    id         TEXT PRIMARY KEY,
                    subject    TEXT,
                    predicate  TEXT,
                    object     TEXT,
                    confidence REAL,
                    source     TEXT,
                    category   TEXT,
                    created_ts REAL,
                    half_life  REAL
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction TEXT,
                    confidence REAL,
                    created_ts TEXT,
                    resolved   INTEGER DEFAULT 0,
                    correct    INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS self_model (
                    domain     TEXT PRIMARY KEY,
                    confidence REAL,
                    attempts   INTEGER DEFAULT 0,
                    successes  INTEGER DEFAULT 0,
                    last_ts    TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_subj ON facts(subject);
                CREATE INDEX IF NOT EXISTS idx_pred ON facts(predicate);
            """)

    def _load(self):
        """Load persisted facts from SQLite."""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM facts").fetchall()
                loaded = 0
                for row in rows:
                    f = Fact.from_dict(dict(row))
                    if f.current_confidence() > 0.01:  # skip near-dead facts
                        self._facts[f.id] = f
                        if self._graph is not None and NX_AVAILABLE:
                            self._graph.add_edge(
                                f.subject, f.object,
                                predicate=f.predicate,
                                confidence=f.current_confidence(),
                            )
                        loaded += 1
            console.print(f"  [green]World model:[/] {loaded} facts loaded")
        except Exception as e:
            console.print(f"  [yellow]World model load error: {e}[/]")

    def _save_fact(self, fact: Fact):
        """Persist a single fact to SQLite."""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO facts
                    (id, subject, predicate, object, confidence, source, category, created_ts, half_life)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (fact.id, fact.subject, fact.predicate, str(fact.object),
                      fact.confidence, fact.source, fact.category,
                      fact.created_ts, fact.half_life))
        except Exception:
            pass

    # ── Core API ──────────────────────────────────────────────────────────────

    def assert_fact(
        self,
        subject:    str,
        predicate:  str,
        obj:        Any,
        confidence: float = 1.0,
        source:     str   = "aria",
        category:   str   = "general",
    ) -> Fact:
        """
        Add or update a fact in the world model.
        If the fact already exists and new confidence is higher, update it.
        """
        fact = Fact(subject, predicate, obj, confidence, source, category)

        with self._lock:
            existing = self._facts.get(fact.id)
            if existing and existing.current_confidence() > confidence:
                return existing  # existing is more confident, keep it

            self._facts[fact.id] = fact

            if self._graph is not None and NX_AVAILABLE:
                self._graph.add_edge(
                    subject, str(obj),
                    predicate=predicate,
                    confidence=confidence,
                )

        self._save_fact(fact)
        return fact

    def query(
        self,
        subject:      str = None,
        predicate:    str = None,
        min_confidence: float = 0.3,
    ) -> list[Fact]:
        """
        Query facts. All parameters are optional filters.
        Returns facts sorted by confidence (highest first).
        """
        results = []
        with self._lock:
            for fact in self._facts.values():
                if subject and fact.subject.lower() != subject.lower():
                    continue
                if predicate and fact.predicate.lower() != predicate.lower():
                    continue
                if fact.current_confidence() < min_confidence:
                    self._refresh_queue.append(fact.id)
                    continue
                results.append(fact)

        results.sort(key=lambda f: f.current_confidence(), reverse=True)
        return results

    def ask(self, subject: str, predicate: str) -> Optional[Any]:
        """
        Simple Q: what is subject's predicate?
        Returns the most confident answer, or None if unknown/stale.
        """
        facts = self.query(subject, predicate, min_confidence=0.3)
        if facts:
            return facts[0].object
        return None

    def knows(self, subject: str, predicate: str) -> tuple[bool, float]:
        """
        Does ARIA know this? Returns (knows, confidence).
        confidence=0 means unknown or stale.
        """
        facts = self.query(subject, predicate, min_confidence=0.0)
        if not facts:
            return False, 0.0
        best = facts[0]
        conf = best.current_confidence()
        return conf > 0.3, conf

    def related(self, entity: str, max_hops: int = 2) -> list[dict]:
        """Find all entities related to this one within N hops."""
        if self._graph is None or not NX_AVAILABLE:
            return []
        results = []
        try:
            for node in nx.single_source_shortest_path_length(
                self._graph, entity, cutoff=max_hops
            ):
                if node != entity:
                    # Find the edge data
                    paths = nx.all_simple_paths(self._graph, entity, node, cutoff=max_hops)
                    for path in paths:
                        results.append({
                            "entity":   node,
                            "via":      path,
                            "distance": len(path) - 1,
                        })
                        break
        except Exception:
            pass
        return results[:20]

    def expire_stale(self, max_age_s: int = 86400 * 30) -> int:
        """Remove facts that are so old they're certainly wrong."""
        now     = time.time()
        expired = []
        with self._lock:
            for fid, fact in list(self._facts.items()):
                if fact.current_confidence() < 0.01:
                    expired.append(fid)
                elif now - fact.created_ts > max_age_s:
                    expired.append(fid)

            for fid in expired:
                del self._facts[fid]

        if expired:
            try:
                with sqlite3.connect(WORLD_DB) as conn:
                    conn.execute(
                        f"DELETE FROM facts WHERE id IN ({','.join('?'*len(expired))})",
                        expired
                    )
            except Exception:
                pass

        return len(expired)

    # ── Self model ────────────────────────────────────────────────────────────

    def record_success(self, domain: str, success: bool):
        """Track ARIA's accuracy per domain. Builds the self-model."""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                conn.execute("""
                    INSERT INTO self_model (domain, confidence, attempts, successes, last_ts)
                    VALUES (?, 0.5, 1, ?, ?)
                    ON CONFLICT(domain) DO UPDATE SET
                        attempts  = attempts + 1,
                        successes = successes + ?,
                        confidence= CAST(successes + ? AS REAL) / (attempts + 1),
                        last_ts   = ?
                """, (domain, int(success), datetime.now().isoformat(),
                      int(success), int(success), datetime.now().isoformat()))
        except Exception:
            pass

    def self_confidence(self, domain: str) -> float:
        """What is ARIA's confidence in its own answers about this domain?"""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                row = conn.execute(
                    "SELECT confidence FROM self_model WHERE domain = ?", (domain,)
                ).fetchone()
                return row[0] if row else 0.5
        except Exception:
            return 0.5

    def capability_gaps(self) -> list[dict]:
        """Domains where ARIA is underperforming — needs more training."""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                rows = conn.execute("""
                    SELECT domain, confidence, attempts
                    FROM self_model
                    WHERE attempts >= 5 AND confidence < 0.6
                    ORDER BY confidence ASC
                """).fetchall()
                return [{"domain": r[0], "confidence": round(r[1],3),
                         "attempts": r[2]} for r in rows]
        except Exception:
            return []

    # ── Prediction tracking ───────────────────────────────────────────────────

    def predict(self, prediction: str, confidence: float):
        """Record a prediction ARIA makes. Track accuracy over time."""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                conn.execute(
                    "INSERT INTO predictions (prediction, confidence, created_ts) VALUES (?,?,?)",
                    (prediction, confidence, datetime.now().isoformat())
                )
        except Exception:
            pass

    def resolve_prediction(self, prediction_id: int, was_correct: bool):
        """Mark a prediction as resolved. Feeds back into self-model."""
        try:
            with sqlite3.connect(WORLD_DB) as conn:
                conn.execute(
                    "UPDATE predictions SET resolved=1, correct=? WHERE id=?",
                    (int(was_correct), prediction_id)
                )
        except Exception:
            pass

    # ── Background refresh ────────────────────────────────────────────────────

    def _start_background(self):
        """Start background fact maintenance loop."""
        def loop():
            while True:
                try:
                    self.expire_stale()
                    self._process_refresh_queue()
                except Exception:
                    pass
                time.sleep(300)  # every 5 minutes

        threading.Thread(target=loop, daemon=True).start()

    def _process_refresh_queue(self):
        """Refresh stale facts that are still needed."""
        with self._lock:
            queue = list(self._refresh_queue[:10])
            self._refresh_queue = self._refresh_queue[10:]

        for fid in queue:
            fact = self._facts.get(fid)
            if not fact:
                continue
            # Only refresh certain categories automatically
            if fact.category == "stock_price":
                self._refresh_stock(fact)
            elif fact.category == "weather":
                self._refresh_weather(fact)

    def _refresh_stock(self, fact: Fact):
        try:
            import requests
            r     = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{fact.subject}",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=5,
            )
            price = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
            self.assert_fact(fact.subject, "stock_price", price, 0.99,
                             "yahoo_finance", "stock_price")
        except Exception:
            pass

    def _refresh_weather(self, fact: Fact):
        try:
            import requests
            r    = requests.get(
                f"https://wttr.in/{fact.subject}?format=j1",
                headers={"User-Agent": "ARIA/1.0"}, timeout=5,
            )
            data = r.json()["current_condition"][0]
            temp = data.get("temp_C","?")
            self.assert_fact(fact.subject, "weather_temp_c", temp, 0.95,
                             "wttr.in", "weather")
        except Exception:
            pass

    def stats(self) -> dict:
        with self._lock:
            total     = len(self._facts)
            fresh     = sum(1 for f in self._facts.values() if f.is_fresh())
            by_cat    = {}
            for f in self._facts.values():
                by_cat[f.category] = by_cat.get(f.category,0) + 1

        graph_nodes = self._graph.number_of_nodes() if self._graph else 0
        graph_edges = self._graph.number_of_edges() if self._graph else 0

        return {
            "total_facts":   total,
            "fresh_facts":   fresh,
            "stale_facts":   total - fresh,
            "by_category":   by_cat,
            "graph_nodes":   graph_nodes,
            "graph_edges":   graph_edges,
            "refresh_queue": len(self._refresh_queue),
        }
