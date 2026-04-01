"""
ARIA — Memory  (v3 — TTL eviction + batch embeddings + adaptive model)
========================================================================
ChromaDB-backed long-term memory with:
  • Adaptive embedding model (smaller on low-RAM devices)
  • Batch embed for ingestion (3–5× faster than one-by-one)
  • TTL-based eviction — old unused memories auto-expire
  • Access frequency tracking — popular memories score higher
  • Domain-aware search with metadata filters
  • Thread-safe concurrent access
  • Lazy init — doesn't block startup

Install: pip install chromadb sentence-transformers
"""

from __future__ import annotations

import uuid
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Optional, List
from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# LAZY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def _try_chromadb():
    try:
        import chromadb
        from chromadb.config import Settings
        return chromadb, Settings
    except ImportError:
        return None, None


def _try_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────────────────────────────────────

class Memory:
    """
    Long-term persistent memory.

    Key improvements:
      • embed() uses batch processing — ingest 100 chunks as fast as 20
      • TTL eviction: set ttl_days=30 to auto-remove stale knowledge
      • access_count tracked — boosted in ranking
      • Adaptive embedding model based on hardware profile
      • Lazy-loaded — startup is instant, first search triggers model load
    """

    def __init__(self, ttl_days: int = 90):
        self._ttl_days  = ttl_days
        self._embedder  = None
        self._client    = None
        self._collection = None
        self._lock      = threading.Lock()
        self._ready     = False
        self._init_error: Optional[str] = None
        self._embed_model_name: Optional[str] = None

        # Lazy init in background so startup is non-blocking
        threading.Thread(target=self._lazy_init, daemon=True,
                         name="aria-memory-init").start()

    def _lazy_init(self):
        """Load ChromaDB + embedding model in the background."""
        try:
            from core.config import (
                CHROMA_PATH, CHROMA_COLLECTION,
                EMBEDDING_MODEL, TOP_K_RETRIEVAL, MIN_SIMILARITY
            )

            chromadb, Settings = _try_chromadb()
            if chromadb is None:
                self._init_error = "chromadb not installed. Run: pip install chromadb"
                return

            SentenceTransformer = _try_sentence_transformers()
            if SentenceTransformer is None:
                self._init_error = "sentence-transformers not installed."
                return

            # Pick embedding model based on hardware
            emb_model = EMBEDDING_MODEL
            try:
                from core.adaptive import AdaptiveManager
                emb_model = AdaptiveManager.get().embedding_model()
            except Exception:
                pass
            self._embed_model_name = emb_model

            console.print(f"  [dim]Loading embedding model: {emb_model}[/]")
            self._embedder = SentenceTransformer(emb_model)
            console.print(f"  [green]Embeddings ready[/] — {emb_model}")

            self._client = chromadb.PersistentClient(
                path=CHROMA_PATH,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            count = self._collection.count()
            console.print(
                f"  [green]Memory ready[/] — {count} chunks in ChromaDB"
            )
            self._ready = True

        except Exception as e:
            self._init_error = str(e)
            console.print(f"  [red]Memory init error:[/] {e}")

    def _wait_ready(self, timeout_s: float = 30.0) -> bool:
        """Block until memory is ready (or timeout)."""
        deadline = datetime.now().timestamp() + timeout_s
        while not self._ready and not self._init_error:
            if datetime.now().timestamp() > deadline:
                return False
            threading.Event().wait(0.2)
        return self._ready

    # ── Embed ─────────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        if not self._wait_ready():
            return []
        return self._embedder.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in one forward pass (much faster than one-by-one).
        Handles adaptive batch size based on available RAM.
        """
        if not self._wait_ready() or not texts:
            return []
        try:
            # Adaptive batch size
            batch_size = 32
            try:
                from core.adaptive import AdaptiveManager
                if AdaptiveManager.get().is_low_memory():
                    batch_size = 8
            except Exception:
                pass

            all_vecs = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i: i + batch_size]
                vecs  = self._embedder.encode(
                    chunk,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                all_vecs.extend(v.tolist() for v in vecs)
            return all_vecs
        except Exception as e:
            console.print(f"  [yellow]Batch embed error: {e}[/]")
            return [self.embed(t) for t in texts]

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(
        self,
        text:     str,
        source:   str           = "unknown",
        domain:   str           = "general",
        metadata: Optional[dict] = None,
        ttl_days: Optional[int]  = None,
    ) -> str:
        """
        Store a chunk. Deduplicates by content hash.
        Returns the chunk ID.
        """
        if not self._wait_ready():
            return ""
        if not text or not text.strip():
            return ""

        content_hash = hashlib.md5(text.encode()).hexdigest()

        with self._lock:
            existing = self._collection.get(where={"content_hash": content_hash})
            if existing["ids"]:
                # Update access count
                self._increment_access(existing["ids"][0])
                return existing["ids"][0]

            chunk_id = str(uuid.uuid4())
            vector   = self.embed(text)
            if not vector:
                return ""

            expires_at = ""
            effective_ttl = ttl_days if ttl_days is not None else self._ttl_days
            if effective_ttl > 0:
                expires_at = (
                    datetime.now() + timedelta(days=effective_ttl)
                ).isoformat()

            meta = {
                "source":       source,
                "domain":       domain,
                "content_hash": content_hash,
                "stored_at":    datetime.now().isoformat(),
                "expires_at":   expires_at,
                "access_count": 0,
                **(metadata or {}),
            }
            self._collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[meta],
            )
        return chunk_id

    def store_many(self, chunks: list[dict]) -> list[str]:
        """
        Batch store. Each chunk: {"text": str, "source": str, "domain": str}.
        Uses batch embedding for 3–5× speed improvement.
        """
        if not self._wait_ready() or not chunks:
            return []

        # Filter duplicates first (avoid embedding them)
        new_chunks = []
        result_ids  = []

        with self._lock:
            for chunk in chunks:
                text         = chunk.get("text", "")
                content_hash = hashlib.md5(text.encode()).hexdigest()
                existing     = self._collection.get(where={"content_hash": content_hash})
                if existing["ids"]:
                    result_ids.append(existing["ids"][0])
                else:
                    new_chunks.append(chunk)

        if not new_chunks:
            console.print(f"[dim]All {len(chunks)} chunks already in memory[/]")
            return result_ids

        # Batch embed new chunks
        texts   = [c["text"] for c in new_chunks]
        vectors = self.embed_batch(texts)

        ids, docs, metas = [], [], []
        effective_ttl = self._ttl_days
        expires_at = ""
        if effective_ttl > 0:
            expires_at = (
                datetime.now() + timedelta(days=effective_ttl)
            ).isoformat()

        for chunk, vector in zip(new_chunks, vectors):
            if not vector:
                continue
            cid = str(uuid.uuid4())
            ids.append(cid)
            docs.append(chunk["text"])
            metas.append({
                "source":       chunk.get("source", "unknown"),
                "domain":       chunk.get("domain", "general"),
                "content_hash": hashlib.md5(chunk["text"].encode()).hexdigest(),
                "stored_at":    datetime.now().isoformat(),
                "expires_at":   expires_at,
                "access_count": 0,
            })
            result_ids.append(cid)

        if ids:
            with self._lock:
                self._collection.add(
                    ids=ids,
                    embeddings=vectors[:len(ids)],
                    documents=docs,
                    metadatas=metas,
                )
            console.print(
                f"[green]Stored {len(ids)} new chunks[/] "
                f"({len(chunks) - len(new_chunks)} duplicates skipped)"
            )

        return result_ids

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        top_k:          int            = None,
        domain:         Optional[str]  = None,
        min_similarity: float          = None,
    ) -> list[dict]:
        """
        Semantic search. Returns ranked list of relevant chunks.
        Expired entries are filtered out automatically.
        """
        if not self._wait_ready():
            return []
        if not query or not query.strip():
            return []

        from core.config import TOP_K_RETRIEVAL, MIN_SIMILARITY
        top_k          = top_k or TOP_K_RETRIEVAL
        min_similarity = min_similarity if min_similarity is not None else MIN_SIMILARITY

        query_vector = self.embed(query)
        if not query_vector:
            return []

        where_filter = {"domain": domain} if domain else None
        count        = self._collection.count()
        if count == 0:
            return []

        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k * 2, max(1, count)),  # fetch extra, filter expired
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            console.print(f"  [yellow]Search error: {e}[/]")
            return []

        hits = []
        now  = datetime.now()
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Filter expired
            exp = meta.get("expires_at", "")
            if exp:
                try:
                    if datetime.fromisoformat(exp) < now:
                        continue
                except Exception:
                    pass

            similarity = 1.0 - dist
            if similarity < min_similarity:
                continue

            # Boost frequently-accessed items slightly
            access_bonus = min(0.05, meta.get("access_count", 0) * 0.002)
            boosted_sim  = min(1.0, similarity + access_bonus)

            hits.append({
                "text":         doc,
                "source":       meta.get("source", "unknown"),
                "domain":       meta.get("domain", "general"),
                "similarity":   round(boosted_sim, 3),
                "stored_at":    meta.get("stored_at", ""),
                "access_count": meta.get("access_count", 0),
            })

        hits = sorted(hits, key=lambda x: x["similarity"], reverse=True)[:top_k]

        # Update access counts in background
        threading.Thread(
            target=self._update_access_counts,
            args=([h["text"] for h in hits],),
            daemon=True,
        ).start()

        return hits

    def build_context(self, query: str, domain: Optional[str] = None) -> tuple[str, bool]:
        """Format memory hits as a RAG context block."""
        hits = self.search(query, domain=domain)
        if not hits:
            return "", False
        lines = ["Relevant knowledge from memory:\n"]
        for i, h in enumerate(hits, 1):
            lines.append(
                f"[{i}] (source: {h['source']}, relevance: {h['similarity']:.2f})\n"
                f"{h['text']}\n"
            )
        return "\n".join(lines), True

    # ── Access tracking ───────────────────────────────────────────────────────

    def _increment_access(self, chunk_id: str):
        """Increment access count for a specific chunk."""
        try:
            result = self._collection.get(ids=[chunk_id], include=["metadatas"])
            if result["metadatas"]:
                meta = result["metadatas"][0]
                meta["access_count"] = int(meta.get("access_count", 0)) + 1
                self._collection.update(ids=[chunk_id], metadatas=[meta])
        except Exception:
            pass

    def _update_access_counts(self, texts: list[str]):
        """Background update of access counts for retrieved texts."""
        if not self._ready:
            return
        for text in texts:
            try:
                content_hash = hashlib.md5(text.encode()).hexdigest()
                result = self._collection.get(
                    where={"content_hash": content_hash}, include=["metadatas"]
                )
                if result["ids"]:
                    self._increment_access(result["ids"][0])
            except Exception:
                pass

    # ── Eviction ──────────────────────────────────────────────────────────────

    def evict_expired(self) -> int:
        """
        Remove all chunks past their TTL.
        Returns number of deleted chunks.
        Called automatically by the maintenance task, or manually.
        """
        if not self._wait_ready(5):
            return 0
        try:
            now_str = datetime.now().isoformat()
            # ChromaDB doesn't support date comparisons natively —
            # we fetch all and filter in Python
            all_data = self._collection.get(include=["metadatas"])
            expired_ids = []
            for cid, meta in zip(all_data["ids"], all_data["metadatas"]):
                exp = meta.get("expires_at", "")
                if not exp:
                    continue
                try:
                    if datetime.fromisoformat(exp) < datetime.now():
                        expired_ids.append(cid)
                except Exception:
                    pass

            if expired_ids:
                self._collection.delete(ids=expired_ids)
                console.print(
                    f"[dim]Memory eviction: removed {len(expired_ids)} expired chunks[/]"
                )
            return len(expired_ids)
        except Exception as e:
            console.print(f"[yellow]Eviction error: {e}[/]")
            return 0

    # ── Management ────────────────────────────────────────────────────────────

    def delete_by_source(self, source: str):
        if not self._wait_ready(5):
            return
        with self._lock:
            self._collection.delete(where={"source": source})
        console.print(f"[yellow]Deleted all chunks from source:[/] {source}")

    def clear_all(self):
        """Wipe all memory. Irreversible."""
        if not self._wait_ready(5):
            return
        with self._lock:
            from core.config import CHROMA_COLLECTION
            self._client.delete_collection(CHROMA_COLLECTION)
            from chromadb.config import Settings
            self._collection = self._client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        console.print("[yellow]Memory wiped.[/]")

    def count(self) -> int:
        if not self._ready:
            return 0
        return self._collection.count()

    def stats(self) -> dict:
        count = self.count() if self._ready else 0
        from core.config import CHROMA_PATH, CHROMA_COLLECTION
        return {
            "total_chunks":   count,
            "db_path":        CHROMA_PATH,
            "collection":     CHROMA_COLLECTION,
            "embedding_model": self._embed_model_name,
            "ttl_days":       self._ttl_days,
            "ready":          self._ready,
            "init_error":     self._init_error,
        }

    def is_ready(self) -> bool:
        return self._ready
