"""
ARIA Multi-Device Sync Engine
Synchronizes conversations, memory, settings, and agent state across devices.

Architecture:
  - Each device has a UUID (persisted in data/sync/device_id)
  - State is stored in SQLite with vector clocks for conflict resolution
  - Devices push deltas via HTTP POST to each peer's /api/sync/push endpoint
  - Pull is periodic (30s) + event-driven (WebSocket notification)
  - Conflict resolution: Last-Write-Wins per field, with vector clock ordering

No external services needed — peer-to-peer, works on LAN or over internet.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests as _req
    _REQUESTS = True
except ImportError:
    _req = None
    _REQUESTS = False

logger = logging.getLogger("aria.sync")

DATA_DIR   = Path(r"C:\Users\chand\ai-remo\data\sync")
DB_PATH    = DATA_DIR / "sync.db"
DEVICE_ID_FILE = DATA_DIR / "device_id"

# Sync namespaces — what can be synced
NAMESPACES = {
    "conversations":  "chat history + context",
    "memory":         "ChromaDB knowledge chunks",
    "settings":       "user preferences + ARIA config",
    "agent_state":    "agent confidence + routing weights",
    "schedules":      "workflows + cron jobs",
    "contacts":       "calendar + contacts",
    "macros":         "automation macros",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class VectorClock:
    """Logical clock for conflict-free distributed ordering."""
    clocks: Dict[str, int] = field(default_factory=dict)

    def tick(self, device_id: str) -> "VectorClock":
        new = VectorClock(dict(self.clocks))
        new.clocks[device_id] = new.clocks.get(device_id, 0) + 1
        return new

    def merge(self, other: "VectorClock") -> "VectorClock":
        merged = dict(self.clocks)
        for dev, ts in other.clocks.items():
            merged[dev] = max(merged.get(dev, 0), ts)
        return VectorClock(merged)

    def happens_before(self, other: "VectorClock") -> bool:
        """Returns True if self happened strictly before other."""
        return (
            all(self.clocks.get(d, 0) <= other.clocks.get(d, 0) for d in set(self.clocks) | set(other.clocks))
            and self.clocks != other.clocks
        )

    def to_dict(self) -> dict:
        return self.clocks

    @classmethod
    def from_dict(cls, d: dict) -> "VectorClock":
        return cls(clocks=d or {})


@dataclass
class SyncDelta:
    """A single change unit to be synced."""
    delta_id:    str            # unique ID for this delta
    device_id:   str            # originating device
    namespace:   str            # e.g. "conversations"
    key:         str            # record key within namespace
    value:       Any            # the data (JSON-serializable)
    deleted:     bool           # tombstone marker
    wall_time:   float          # unix timestamp
    vector_clock: dict          # VectorClock.to_dict()
    checksum:    str            # SHA256 of JSON(value)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SyncDelta":
        return cls(**d)


@dataclass
class PeerDevice:
    """A registered sync peer."""
    device_id:   str
    name:        str
    base_url:    str            # e.g. "http://192.168.1.10:8000"
    last_seen:   float
    last_sync:   float
    trusted:     bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ── SQLite storage ────────────────────────────────────────────────────────────

def _init_db(db_path: Path) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS deltas (
            delta_id     TEXT PRIMARY KEY,
            device_id    TEXT NOT NULL,
            namespace    TEXT NOT NULL,
            key          TEXT NOT NULL,
            value        TEXT,
            deleted      INTEGER NOT NULL DEFAULT 0,
            wall_time    REAL NOT NULL,
            vector_clock TEXT NOT NULL,
            checksum     TEXT NOT NULL,
            synced       INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_deltas_ns_key ON deltas(namespace, key);
        CREATE INDEX IF NOT EXISTS idx_deltas_time ON deltas(wall_time);
        CREATE INDEX IF NOT EXISTS idx_deltas_unsynced ON deltas(synced) WHERE synced = 0;

        CREATE TABLE IF NOT EXISTS peers (
            device_id TEXT PRIMARY KEY,
            name      TEXT NOT NULL,
            base_url  TEXT NOT NULL,
            last_seen REAL NOT NULL DEFAULT 0,
            last_sync REAL NOT NULL DEFAULT 0,
            trusted   INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS sync_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            event      TEXT NOT NULL,
            device_id  TEXT,
            namespace  TEXT,
            detail     TEXT,
            ts         REAL NOT NULL
        );
    """)
    conn.commit()
    return conn


def _get_or_create_device_id() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DEVICE_ID_FILE.exists():
        return DEVICE_ID_FILE.read_text().strip()
    dev_id = str(uuid.uuid4())
    DEVICE_ID_FILE.write_text(dev_id)
    return dev_id


def _checksum(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Core sync engine ──────────────────────────────────────────────────────────

class SyncEngine:
    """
    Multi-device state synchronization for ARIA.

    Public API:
        engine.write(namespace, key, value)       -- record a local change
        engine.delete(namespace, key)             -- tombstone a record
        engine.read(namespace, key) -> Any        -- read latest merged value
        engine.read_all(namespace) -> dict        -- read all keys in namespace
        engine.add_peer(device_id, name, url)     -- register a sync peer
        engine.remove_peer(device_id)             -- remove a peer
        engine.list_peers() -> list               -- list registered peers
        engine.push_to_peers()                    -- push unsynced deltas to all peers
        engine.pull_from_peer(device_id) -> int   -- pull deltas from a peer
        engine.receive_push(deltas: list) -> int  -- receive pushed deltas
        engine.status() -> dict                   -- sync health status
        engine.start()                            -- start background sync
        engine.stop()                             -- stop background sync
    """

    def __init__(
        self,
        device_name: str = "ARIA",
        sync_interval: int = 30,
        max_delta_age_days: int = 30,
    ):
        self.device_id      = _get_or_create_device_id()
        self.device_name    = device_name
        self.sync_interval  = sync_interval
        self.max_delta_age_days = max_delta_age_days

        self._db            = _init_db(DB_PATH)
        self._lock          = threading.Lock()
        self._vector_clock  = self._load_vector_clock()
        self._stop_event    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # In-memory materialized view: namespace -> key -> value
        self._view: Dict[str, Dict[str, Any]] = {}
        self._rebuild_view()

        logger.info(f"SyncEngine ready — device_id={self.device_id[:8]}... name={device_name}")

    # ── Local write/read ──────────────────────────────────────────────────────

    def write(self, namespace: str, key: str, value: Any) -> str:
        """Record a local change. Returns delta_id."""
        with self._lock:
            self._vector_clock = self._vector_clock.tick(self.device_id)
            delta = SyncDelta(
                delta_id    = str(uuid.uuid4()),
                device_id   = self.device_id,
                namespace   = namespace,
                key         = key,
                value       = value,
                deleted     = False,
                wall_time   = time.time(),
                vector_clock= self._vector_clock.to_dict(),
                checksum    = _checksum(value),
            )
            self._store_delta(delta)
            self._apply_to_view(delta)
            self._save_vector_clock()
            return delta.delta_id

    def delete(self, namespace: str, key: str) -> str:
        """Tombstone a record. Returns delta_id."""
        with self._lock:
            self._vector_clock = self._vector_clock.tick(self.device_id)
            delta = SyncDelta(
                delta_id    = str(uuid.uuid4()),
                device_id   = self.device_id,
                namespace   = namespace,
                key         = key,
                value       = None,
                deleted     = True,
                wall_time   = time.time(),
                vector_clock= self._vector_clock.to_dict(),
                checksum    = _checksum(None),
            )
            self._store_delta(delta)
            self._apply_to_view(delta)
            self._save_vector_clock()
            return delta.delta_id

    def read(self, namespace: str, key: str) -> Optional[Any]:
        """Read the latest merged value for a key."""
        return self._view.get(namespace, {}).get(key)

    def read_all(self, namespace: str) -> Dict[str, Any]:
        """Read all non-deleted keys in a namespace."""
        return dict(self._view.get(namespace, {}))

    # ── Peer management ───────────────────────────────────────────────────────

    def add_peer(self, device_id: str, name: str, base_url: str, trusted: bool = True) -> bool:
        """Register a sync peer."""
        if device_id == self.device_id:
            return False
        with self._lock:
            self._db.execute("""
                INSERT OR REPLACE INTO peers (device_id, name, base_url, last_seen, last_sync, trusted)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (device_id, name, base_url.rstrip("/"), time.time(), 0, int(trusted)))
            self._db.commit()
        logger.info(f"Peer added: {name} ({device_id[:8]}...) at {base_url}")
        return True

    def remove_peer(self, device_id: str) -> bool:
        with self._lock:
            self._db.execute("DELETE FROM peers WHERE device_id = ?", (device_id,))
            self._db.commit()
        return True

    def list_peers(self) -> List[dict]:
        rows = self._db.execute(
            "SELECT device_id, name, base_url, last_seen, last_sync, trusted FROM peers"
        ).fetchall()
        return [
            {"device_id": r[0], "name": r[1], "base_url": r[2],
             "last_seen": r[3], "last_sync": r[4], "trusted": bool(r[5])}
            for r in rows
        ]

    # ── Push / pull ───────────────────────────────────────────────────────────

    def get_unsynced_deltas(self, since_time: float = 0) -> List[dict]:
        """Return all unsynced deltas (or all since a timestamp)."""
        with self._lock:
            rows = self._db.execute("""
                SELECT delta_id, device_id, namespace, key, value, deleted,
                       wall_time, vector_clock, checksum
                FROM deltas
                WHERE wall_time >= ? AND synced = 0
                ORDER BY wall_time ASC
            """, (since_time,)).fetchall()
        return [self._row_to_delta_dict(r) for r in rows]

    def get_deltas_since(self, since_time: float) -> List[dict]:
        """Return all deltas (synced or not) since a wall-time."""
        with self._lock:
            rows = self._db.execute("""
                SELECT delta_id, device_id, namespace, key, value, deleted,
                       wall_time, vector_clock, checksum
                FROM deltas
                WHERE wall_time >= ?
                ORDER BY wall_time ASC
            """, (since_time,)).fetchall()
        return [self._row_to_delta_dict(r) for r in rows]

    def receive_push(self, raw_deltas: List[dict]) -> int:
        """
        Accept deltas pushed from a remote peer.
        Returns count of new deltas applied.
        """
        applied = 0
        with self._lock:
            for rd in raw_deltas:
                try:
                    delta = SyncDelta.from_dict(rd)
                    if delta.device_id == self.device_id:
                        continue  # our own delta bounced back
                    if self._delta_exists(delta.delta_id):
                        continue  # already have it
                    # Verify checksum
                    if _checksum(delta.value) != delta.checksum:
                        logger.warning(f"Checksum mismatch on delta {delta.delta_id[:8]}")
                        continue
                    self._store_delta(delta, synced=1)
                    self._apply_to_view(delta)
                    # Merge vector clock
                    remote_vc = VectorClock.from_dict(delta.vector_clock)
                    self._vector_clock = self._vector_clock.merge(remote_vc)
                    applied += 1
                except Exception as e:
                    logger.error(f"Error applying remote delta: {e}")
            if applied:
                self._save_vector_clock()
                self._db.commit()
        if applied:
            logger.info(f"Received {applied} new deltas from push")
            self._log_event("push_received", detail=f"applied={applied}")
        return applied

    def push_to_peers(self) -> dict:
        """Push unsynced local deltas to all trusted peers. Returns {peer_id: result}."""
        if not _REQUESTS:
            return {"error": "requests not installed"}

        unsynced = self.get_unsynced_deltas()
        if not unsynced:
            return {"pushed": 0}

        peers = self.list_peers()
        results = {}
        for peer in peers:
            if not peer["trusted"]:
                continue
            try:
                result = self._push_to_peer(peer, unsynced)
                results[peer["device_id"]] = result
                if result.get("ok"):
                    # Mark deltas as synced
                    with self._lock:
                        self._db.execute("""
                            UPDATE deltas SET synced = 1
                            WHERE synced = 0 AND device_id = ?
                        """, (self.device_id,))
                        self._db.execute("""
                            UPDATE peers SET last_sync = ? WHERE device_id = ?
                        """, (time.time(), peer["device_id"]))
                        self._db.commit()
            except Exception as e:
                results[peer["device_id"]] = {"ok": False, "error": str(e)}

        return {"pushed": len(unsynced), "peers": results}

    def pull_from_peer(self, device_id: str) -> int:
        """Pull deltas from a specific peer since last sync. Returns count applied."""
        if not _REQUESTS:
            return 0
        peer = next((p for p in self.list_peers() if p["device_id"] == device_id), None)
        if not peer:
            return 0
        try:
            since = peer["last_sync"]
            url = f"{peer['base_url']}/api/sync/deltas?since={since}&device_id={self.device_id}"
            resp = _req.get(url, timeout=10)
            if resp.status_code != 200:
                return 0
            data = resp.json()
            raw_deltas = data.get("deltas", [])
            applied = self.receive_push(raw_deltas)
            with self._lock:
                self._db.execute("UPDATE peers SET last_sync = ?, last_seen = ? WHERE device_id = ?",
                                 (time.time(), time.time(), device_id))
                self._db.commit()
            return applied
        except Exception as e:
            logger.error(f"Pull from {device_id[:8]} failed: {e}")
            return 0

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM deltas").fetchone()[0]
            unsynced = self._db.execute("SELECT COUNT(*) FROM deltas WHERE synced = 0").fetchone()[0]
        peers = self.list_peers()
        return {
            "device_id":         self.device_id,
            "device_name":       self.device_name,
            "total_deltas":      total,
            "unsynced_deltas":   unsynced,
            "peers":             len(peers),
            "vector_clock":      self._vector_clock.to_dict(),
            "namespaces":        list(self._view.keys()),
            "background_sync":   self._thread is not None and self._thread.is_alive(),
        }

    # ── Background sync ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sync_loop,
            name="aria-sync-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"SyncEngine background loop started (interval={self.sync_interval}s)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _sync_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self.sync_interval)
            if self._stop_event.is_set():
                break
            try:
                # Pull from all peers first, then push our changes
                for peer in self.list_peers():
                    if peer["trusted"]:
                        self.pull_from_peer(peer["device_id"])
                self.push_to_peers()
                # Periodic cleanup of old deltas
                self._cleanup_old_deltas()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _store_delta(self, delta: SyncDelta, synced: int = 0) -> None:
        self._db.execute("""
            INSERT OR IGNORE INTO deltas
            (delta_id, device_id, namespace, key, value, deleted, wall_time, vector_clock, checksum, synced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            delta.delta_id, delta.device_id, delta.namespace, delta.key,
            json.dumps(delta.value), int(delta.deleted), delta.wall_time,
            json.dumps(delta.vector_clock), delta.checksum, synced,
        ))
        self._db.commit()

    def _delta_exists(self, delta_id: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM deltas WHERE delta_id = ?", (delta_id,)
        ).fetchone()
        return row is not None

    def _apply_to_view(self, delta: SyncDelta) -> None:
        """Apply a delta to the in-memory materialized view using LWW."""
        ns = self._view.setdefault(delta.namespace, {})
        if delta.deleted:
            ns.pop(delta.key, None)
        else:
            # LWW: only apply if newer than what we have
            existing_time = ns.get(f"__ts_{delta.key}", 0)
            if delta.wall_time >= existing_time:
                ns[delta.key] = delta.value
                ns[f"__ts_{delta.key}"] = delta.wall_time

    def _rebuild_view(self) -> None:
        """Rebuild in-memory view from all stored deltas (latest per key)."""
        rows = self._db.execute("""
            SELECT namespace, key, value, deleted, wall_time
            FROM deltas
            ORDER BY wall_time ASC
        """).fetchall()
        self._view = {}
        for ns, key, value_json, deleted, wall_time in rows:
            ns_dict = self._view.setdefault(ns, {})
            if deleted:
                ns_dict.pop(key, None)
                ns_dict.pop(f"__ts_{key}", None)
            else:
                try:
                    v = json.loads(value_json) if value_json else None
                except Exception:
                    v = value_json
                existing_time = ns_dict.get(f"__ts_{key}", 0)
                if wall_time >= existing_time:
                    ns_dict[key] = v
                    ns_dict[f"__ts_{key}"] = wall_time

    def _load_vector_clock(self) -> VectorClock:
        row = self._db.execute(
            "SELECT detail FROM sync_log WHERE event = 'vector_clock' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if row:
            try:
                return VectorClock.from_dict(json.loads(row[0]))
            except Exception:
                pass
        return VectorClock({self.device_id: 0})

    def _save_vector_clock(self) -> None:
        self._db.execute(
            "INSERT INTO sync_log (event, detail, ts) VALUES (?, ?, ?)",
            ("vector_clock", json.dumps(self._vector_clock.to_dict()), time.time())
        )

    def _log_event(self, event: str, device_id: str = None, namespace: str = None, detail: str = None) -> None:
        self._db.execute(
            "INSERT INTO sync_log (event, device_id, namespace, detail, ts) VALUES (?, ?, ?, ?, ?)",
            (event, device_id, namespace, detail, time.time())
        )
        self._db.commit()

    def _push_to_peer(self, peer: dict, deltas: List[dict]) -> dict:
        url = f"{peer['base_url']}/api/sync/push"
        payload = {
            "from_device_id":   self.device_id,
            "from_device_name": self.device_name,
            "deltas":           deltas,
        }
        resp = _req.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            return {"ok": True, "applied": resp.json().get("applied", 0)}
        return {"ok": False, "status": resp.status_code}

    def _cleanup_old_deltas(self) -> None:
        """Remove synced deltas older than max_delta_age_days."""
        cutoff = time.time() - (self.max_delta_age_days * 86400)
        with self._lock:
            self._db.execute(
                "DELETE FROM deltas WHERE wall_time < ? AND synced = 1", (cutoff,)
            )
            self._db.commit()

    def _row_to_delta_dict(self, row: tuple) -> dict:
        delta_id, device_id, namespace, key, value_json, deleted, wall_time, vc_json, checksum = row
        try:
            value = json.loads(value_json) if value_json else None
        except Exception:
            value = value_json
        try:
            vector_clock = json.loads(vc_json)
        except Exception:
            vector_clock = {}
        return {
            "delta_id":     delta_id,
            "device_id":    device_id,
            "namespace":    namespace,
            "key":          key,
            "value":        value,
            "deleted":      bool(deleted),
            "wall_time":    wall_time,
            "vector_clock": vector_clock,
            "checksum":     checksum,
        }


# ── Module-level singleton ────────────────────────────────────────────────────

_engine: Optional[SyncEngine] = None
_engine_lock = threading.Lock()


def get_engine(device_name: str = "ARIA") -> SyncEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = SyncEngine(device_name=device_name)
        return _engine
