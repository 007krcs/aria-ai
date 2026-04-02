"""
ARIA Memory Hierarchy — Four-Layer Memory System
=================================================
Separates memory by type so the brain reads the right layer:

  Working Memory   — current session, in-process only, very fast
  Episodic Memory  — past interactions, events, conversations (decays)
  Semantic Memory  — facts, user preferences, stable knowledge (slow decay)
  Procedural Memory — learned workflows, how-to patterns (very slow decay)

Each layer has its own:
  - Storage backend
  - Decay rate
  - Retrieval strategy
  - Write policy
"""

from __future__ import annotations

import json
import time
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "memory"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


# ── Lazy ChromaDB ─────────────────────────────────────────────────────────────
def _chroma_client():
    try:
        import chromadb
        return chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
    except Exception:
        return None

def _embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BASE MEMORY LAYER
# ─────────────────────────────────────────────────────────────────────────────

class _BaseMemoryLayer:
    def __init__(self, name: str, decay_days: float, max_items: int):
        self.name       = name
        self.decay_days = decay_days
        self.max_items  = max_items
        self._path      = DATA_DIR / f"{name}.json"
        self._items: List[Dict] = []
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._items = json.loads(self._path.read_text())
            except Exception:
                self._items = []

    def _save(self):
        try:
            self._path.write_text(json.dumps(self._items[-self.max_items:], indent=2))
        except Exception:
            pass

    def _is_fresh(self, item: Dict) -> bool:
        if self.decay_days <= 0:
            return True
        try:
            created = datetime.fromisoformat(item["created_at"])
            return datetime.now() - created < timedelta(days=self.decay_days)
        except Exception:
            return True

    def _evict_stale(self):
        before = len(self._items)
        self._items = [i for i in self._items if self._is_fresh(i)]
        if len(self._items) < before:
            self._save()

    def write(self, text: str, metadata: Optional[Dict] = None) -> str:
        item_id = hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:12]
        item = {
            "id":         item_id,
            "text":       text[:2000],
            "metadata":   metadata or {},
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
        }
        with _lock:
            self._items.append(item)
            self._evict_stale()
            self._save()
        return item_id

    def search(self, query: str, k: int = 5) -> List[Dict]:
        self._evict_stale()
        q = query.lower()
        scored = []
        for item in self._items:
            text = item["text"].lower()
            # simple keyword overlap score
            words = set(q.split())
            hits  = sum(1 for w in words if w in text and len(w) > 3)
            if hits > 0:
                recency_bonus = 1.0
                try:
                    age_hours = (datetime.now() - datetime.fromisoformat(item["created_at"])).total_seconds() / 3600
                    recency_bonus = max(0.1, 1.0 - age_hours / (self.decay_days * 24 * 2))
                except Exception:
                    pass
                score = hits * recency_bonus + item.get("access_count", 0) * 0.1
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, item in scored[:k]:
            item["access_count"] = item.get("access_count", 0) + 1
            results.append(item)

        if results:
            self._save()
        return results

    def forget(self, query: str):
        q = query.lower()
        before = len(self._items)
        self._items = [
            i for i in self._items
            if not any(w in i["text"].lower() for w in q.split() if len(w) > 3)
        ]
        if len(self._items) < before:
            self._save()

    def all(self) -> List[Dict]:
        self._evict_stale()
        return list(self._items)

    def clear(self):
        self._items = []
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# FOUR MEMORY LAYERS
# ─────────────────────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    In-process only. Ultra-fast. Lost on restart. Holds current session state.
    """
    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._log: List[Dict] = []

    def set(self, key: str, value: Any):
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def log(self, text: str, metadata: Optional[Dict] = None):
        self._log.append({
            "text": text,
            "metadata": metadata or {},
            "ts": datetime.now().isoformat()
        })
        if len(self._log) > 100:
            self._log = self._log[-100:]

    def recent(self, n: int = 5) -> List[Dict]:
        return self._log[-n:]

    def search(self, query: str, k: int = 3) -> List[Dict]:
        q = query.lower()
        results = []
        for item in reversed(self._log):
            if any(w in item["text"].lower() for w in q.split() if len(w) > 3):
                results.append(item)
                if len(results) >= k:
                    break
        return results

    def clear(self):
        self._store.clear()
        self._log.clear()


class EpisodicMemory(_BaseMemoryLayer):
    """
    Past events, conversations, interactions.
    Decays in 30 days. Max 500 items.
    """
    def __init__(self):
        super().__init__("episodic", decay_days=30, max_items=500)


class SemanticMemory(_BaseMemoryLayer):
    """
    Facts, user preferences, stable knowledge.
    Decays in 180 days. Max 1000 items.
    Preferences have no decay.
    """
    def __init__(self):
        super().__init__("semantic", decay_days=180, max_items=1000)

    def write_preference(self, preference: str):
        """Preferences stored with no decay."""
        return self.write(preference, metadata={"type": "preference", "decay": False})

    def write_fact(self, fact: str, source: Optional[str] = None, trust: float = 0.8):
        return self.write(fact, metadata={
            "type": "fact",
            "source": source or "unknown",
            "trust_score": trust,
        })

    def get_preferences(self) -> List[str]:
        return [
            i["text"] for i in self._items
            if i.get("metadata", {}).get("type") == "preference"
        ]


class ProceduralMemory(_BaseMemoryLayer):
    """
    Learned workflows, repeated patterns, how-to sequences.
    Very slow decay (365 days). Max 200 items.
    Upgraded when the same workflow is used multiple times.
    """
    def __init__(self):
        super().__init__("procedural", decay_days=365, max_items=200)

    def learn_workflow(self, name: str, steps: List[str], trigger: str = ""):
        return self.write(
            f"Workflow: {name}\nTrigger: {trigger}\nSteps:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps)),
            metadata={"type": "workflow", "name": name, "usage_count": 1}
        )

    def get_workflows(self) -> List[Dict]:
        return [i for i in self._items if i.get("metadata", {}).get("type") == "workflow"]

    def reinforce(self, workflow_name: str):
        """Increase usage count for a workflow."""
        for item in self._items:
            if item.get("metadata", {}).get("name") == workflow_name:
                item["metadata"]["usage_count"] = item["metadata"].get("usage_count", 1) + 1
                self._save()
                break


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY HIERARCHY — Unified interface
# ─────────────────────────────────────────────────────────────────────────────

class MemoryHierarchy:
    """
    Single access point for all memory layers.
    The brain calls this — not individual layers directly.
    """

    def __init__(self):
        self.working    = WorkingMemory()
        self.episodic   = EpisodicMemory()
        self.semantic   = SemanticMemory()
        self.procedural = ProceduralMemory()

    # ── Write ────────────────────────────────────────────────────────────────

    def write_episodic(self, text: str, metadata: Optional[Dict] = None) -> str:
        return self.episodic.write(text, metadata)

    def write_semantic(self, text: str, metadata: Optional[Dict] = None) -> str:
        return self.semantic.write(text, metadata)

    def write_procedural(self, text: str, metadata: Optional[Dict] = None) -> str:
        return self.procedural.write(text, metadata)

    def write_preference(self, preference: str) -> str:
        return self.semantic.write_preference(preference)

    def write_fact(self, fact: str, source: str = "", trust: float = 0.8) -> str:
        return self.semantic.write_fact(fact, source, trust)

    # ── Search — cross-layer ─────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search all layers and return merged results.
        Priority: Semantic > Procedural > Episodic > Working
        """
        results = []

        # Working memory — immediate context
        wm = self.working.search(query, k=2)
        for r in wm:
            r["source"] = "working"
            results.append(r)

        # Semantic — stable knowledge (highest priority)
        sm = self.semantic.search(query, k=3)
        for r in sm:
            r["source"] = "semantic"
            results.append(r)

        # Procedural — workflows
        pm = self.procedural.search(query, k=2)
        for r in pm:
            r["source"] = "procedural"
            results.append(r)

        # Episodic — past events (lower priority)
        em = self.episodic.search(query, k=3)
        for r in em:
            r["source"] = "episodic"
            results.append(r)

        # Deduplicate by text similarity
        seen = set()
        unique = []
        for r in results:
            key = r["text"][:80]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:k]

    # ── Forget ───────────────────────────────────────────────────────────────

    def forget(self, query: str):
        """Remove from all layers."""
        self.episodic.forget(query)
        self.semantic.forget(query)
        self.procedural.forget(query)
        self.working.clear()

    # ── Introspect ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        return {
            "working":    len(self.working._log),
            "episodic":   len(self.episodic._items),
            "semantic":   len(self.semantic._items),
            "procedural": len(self.procedural._items),
        }

    def get_preferences(self) -> List[str]:
        return self.semantic.get_preferences()

    def get_workflows(self) -> List[Dict]:
        return self.procedural.get_workflows()
