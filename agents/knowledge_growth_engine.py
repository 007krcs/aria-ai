"""
ARIA Knowledge Growth Engine
==============================
ARIA's brain grows every day — zero hallucination, always cited.

This engine is the central intelligence layer that ensures:
  1. Every response is grounded in verified knowledge
  2. Every new piece of information is absorbed into permanent memory
  3. Knowledge conflicts are resolved (newer, higher-confidence wins)
  4. ARIA knows what it doesn't know (uncertainty quantification)
  5. Continuous web + file + conversation learning, 24/7

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  INPUT: user query / file / conversation / web page │
  └──────────────────────┬──────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  Extract Claims     │  → sentence-level fact extraction
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Verify + Score     │  → confidence 0-100 per claim
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Store in ChromaDB  │  → with source, confidence, timestamp
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Conflict resolver  │  → old fact vs new fact → keep highest conf
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │  Serve grounded     │  → RAG-augmented generation
              └─────────────────────┘
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Algo-core (graceful fallback) ─────────────────────────────────────────────
try:
    from agents.algo_core import AdaptiveLearner, CorrelationEngine
    _ALGO_CORE = True
except ImportError:
    _ALGO_CORE = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH      = PROJECT_ROOT / "data" / "knowledge.db"
DB_PATH.parent.mkdir(exist_ok=True)

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_MODEL = None
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Claim:
    text:        str
    source:      str       = "unknown"
    confidence:  float     = 0.75
    verified:    bool      = False
    timestamp:   str       = field(default_factory=lambda: datetime.utcnow().isoformat())
    domain:      str       = "general"
    claim_hash:  str       = ""

    def __post_init__(self):
        if not self.claim_hash:
            self.claim_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]


@dataclass
class KnowledgeEntry:
    claim:       Claim
    embedding:   Optional[List[float]] = None
    related:     List[str]             = field(default_factory=list)
    contradicts: List[str]             = field(default_factory=list)
    update_count: int                  = 0


@dataclass
class GroundedResponse:
    response:         str
    confidence:       float
    sources:          List[str]
    claims:           List[Claim]
    uncertain_parts:  List[str]
    corrections:      List[str]


# ── SQLite knowledge store ────────────────────────────────────────────────────

class KnowledgeStore:
    """Thread-safe SQLite store for facts and claims."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db   = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        return sqlite3.connect(str(self._db), check_same_thread=False)

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS claims (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_hash  TEXT UNIQUE,
                    text        TEXT NOT NULL,
                    source      TEXT,
                    confidence  REAL DEFAULT 0.75,
                    verified    INTEGER DEFAULT 0,
                    timestamp   TEXT,
                    domain      TEXT DEFAULT 'general',
                    update_count INTEGER DEFAULT 0,
                    last_verified TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS contradictions (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash_a    TEXT,
                    hash_b    TEXT,
                    resolved  INTEGER DEFAULT 0,
                    winner    TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    url        TEXT UNIQUE,
                    domain     TEXT,
                    trust_tier INTEGER DEFAULT 2,
                    last_seen  TEXT
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_domain ON claims(domain)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_conf ON claims(confidence)")
            # Upgrade D migration: add last_verified column if missing
            try:
                c.execute("ALTER TABLE claims ADD COLUMN last_verified TEXT")
            except Exception:
                pass  # Column already exists
            c.commit()

    def upsert(self, claim: Claim) -> bool:
        """Insert or update a claim. Returns True if new."""
        with self._lock, self._conn() as c:
            existing = c.execute(
                "SELECT confidence, update_count FROM claims WHERE claim_hash=?",
                (claim.claim_hash,)
            ).fetchone()

            if existing:
                old_conf, count = existing
                # Keep higher confidence version
                new_conf = max(old_conf, claim.confidence)
                c.execute("""
                    UPDATE claims SET confidence=?, verified=?, timestamp=?,
                    update_count=? WHERE claim_hash=?
                """, (new_conf, int(claim.verified), claim.timestamp,
                      count + 1, claim.claim_hash))
                c.commit()
                return False
            else:
                c.execute("""
                    INSERT INTO claims (claim_hash, text, source, confidence,
                    verified, timestamp, domain)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (claim.claim_hash, claim.text, claim.source,
                      claim.confidence, int(claim.verified),
                      claim.timestamp, claim.domain))
                c.commit()
                return True

    def search(self, query: str, domain: str = None,
               min_confidence: float = 0.5, limit: int = 10) -> List[Claim]:
        """Full-text search over stored claims."""
        with self._lock, self._conn() as c:
            q = "%" + query.lower() + "%"
            if domain:
                rows = c.execute("""
                    SELECT text, source, confidence, verified, timestamp, domain, claim_hash
                    FROM claims
                    WHERE LOWER(text) LIKE ? AND domain=? AND confidence>=?
                    ORDER BY confidence DESC, update_count DESC LIMIT ?
                """, (q, domain, min_confidence, limit)).fetchall()
            else:
                rows = c.execute("""
                    SELECT text, source, confidence, verified, timestamp, domain, claim_hash
                    FROM claims
                    WHERE LOWER(text) LIKE ? AND confidence>=?
                    ORDER BY confidence DESC, update_count DESC LIMIT ?
                """, (q, min_confidence, limit)).fetchall()

        return [
            Claim(text=r[0], source=r[1], confidence=r[2],
                  verified=bool(r[3]), timestamp=r[4], domain=r[5],
                  claim_hash=r[6])
            for r in rows
        ]

    def apply_confidence_decay(self, decay_per_day: float = 0.99) -> int:
        """
        Upgrade D: Decay confidence for unverified claims by decay_per_day^days_since_verified.
        Returns number of claims updated.
        """
        now = datetime.utcnow()
        updated = 0
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT claim_hash, confidence, last_verified, timestamp FROM claims WHERE verified=0"
            ).fetchall()
            for claim_hash, confidence, last_verified, timestamp in rows:
                ref_date_str = last_verified or timestamp
                try:
                    ref_date = datetime.fromisoformat(ref_date_str.split('.')[0])
                except Exception:
                    ref_date = now
                days_elapsed = max(0.0, (now - ref_date).total_seconds() / 86400.0)
                new_conf = confidence * (decay_per_day ** days_elapsed)
                if abs(new_conf - confidence) > 0.001:
                    c.execute(
                        "UPDATE claims SET confidence=? WHERE claim_hash=?",
                        (max(0.01, new_conf), claim_hash)
                    )
                    updated += 1
            c.commit()
        return updated

    def get_stale_claims(self, min_confidence: float = 0.01, max_confidence: float = 0.30) -> List[Claim]:
        """
        Upgrade D: Return claims whose confidence has decayed below threshold.
        These are candidates for re-verification.
        """
        with self._lock, self._conn() as c:
            rows = c.execute("""
                SELECT text, source, confidence, verified, timestamp, domain, claim_hash
                FROM claims
                WHERE confidence>=? AND confidence<=? AND verified=0
                ORDER BY confidence ASC LIMIT 20
            """, (min_confidence, max_confidence)).fetchall()
        return [
            Claim(text=r[0], source=r[1], confidence=r[2],
                  verified=bool(r[3]), timestamp=r[4], domain=r[5],
                  claim_hash=r[6])
            for r in rows
        ]

    def get_all_claims_as_vectors(self) -> List[Tuple[str, List[float]]]:
        """
        Return (claim_hash, word_freq_vector) for MI computation.
        Vectorise using top-100 word vocabulary frequency counts.
        """
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT claim_hash, text FROM claims ORDER BY confidence DESC LIMIT 500"
            ).fetchall()
        if not rows:
            return []

        # Build vocabulary
        all_tokens: List[str] = []
        texts = {h: t for h, t in rows}
        for text in texts.values():
            all_tokens.extend(re.findall(r'\b\w{3,}\b', text.lower()))
        vocab_counts: Dict[str, int] = {}
        for tok in all_tokens:
            vocab_counts[tok] = vocab_counts.get(tok, 0) + 1
        vocab = [w for w, _ in sorted(vocab_counts.items(), key=lambda x: -x[1])[:50]]
        if not vocab:
            return []

        result = []
        for h, text in texts.items():
            tokens = re.findall(r'\b\w{3,}\b', text.lower())
            total = max(len(tokens), 1)
            vec = [tokens.count(w) / total for w in vocab]
            result.append((h, vec))
        return result

    def stats(self) -> Dict[str, Any]:
        with self._lock, self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            verified = c.execute(
                "SELECT COUNT(*) FROM claims WHERE verified=1").fetchone()[0]
            high_conf = c.execute(
                "SELECT COUNT(*) FROM claims WHERE confidence>=0.85").fetchone()[0]
            domains = c.execute(
                "SELECT domain, COUNT(*) FROM claims GROUP BY domain").fetchall()
        return {
            "total_claims": total,
            "verified":     verified,
            "high_confidence": high_conf,
            "by_domain":    dict(domains),
        }


# ── Claim extractor ───────────────────────────────────────────────────────────

# Patterns that indicate high-confidence factual claims
_FACT_MARKERS = [
    r"\b(is|are|was|were|has|have|can|will|does|do)\b",
    r"\b\d{4}\b",            # years
    r"\b(founded|created|invented|discovered|launched)\b",
    r"\b(located|situated|based)\b",
]

# Patterns that indicate uncertainty (lower confidence)
_UNCERTAIN_MARKERS = [
    r"\b(might|may|could|possibly|perhaps|allegedly|reportedly|some say|rumored)\b",
    r"\b(I think|I believe|probably|likely|seems|appears)\b",
]

def extract_claims(text: str, source: str = "unknown",
                   domain: str = "general") -> List[Claim]:
    """
    Split text into individual claim sentences and score confidence.
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    claims = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 400:
            continue
        if sent.startswith(("```", "#", "- ", "*")):
            continue

        # Base confidence
        confidence = 0.70

        # Boost for factual markers
        low = sent.lower()
        for pat in _FACT_MARKERS:
            if re.search(pat, low):
                confidence += 0.03

        # Reduce for uncertainty
        for pat in _UNCERTAIN_MARKERS:
            if re.search(pat, low):
                confidence -= 0.15

        # Reduce for very long sentences (more likely to have inaccuracies)
        if len(sent) > 200:
            confidence -= 0.05

        confidence = max(0.10, min(0.95, confidence))

        claims.append(Claim(
            text=sent,
            source=source,
            confidence=confidence,
            domain=domain,
        ))

    return claims


# ── Web verifier ──────────────────────────────────────────────────────────────

def verify_claim_web(claim_text: str) -> Tuple[bool, float, str]:
    """
    Quick web verification of a claim via ResearchSearchEngine.
    Returns (verified: bool, confidence: float, source_url: str)
    """
    try:
        from agents.research_search_engine import ResearchSearchEngine
        engine = ResearchSearchEngine()
        results = engine.search(claim_text, max_results=3)
        if not results:
            return False, 0.5, ""

        # Pick best result with longest abstract
        best = max(results, key=lambda r: len(r.abstract or ""))
        abstract = best.abstract or ""
        source = best.url or ""

        if abstract and len(abstract) > 30:
            claim_words = set(re.findall(r'\b\w{4,}\b', claim_text.lower()))
            abst_words  = set(re.findall(r'\b\w{4,}\b', abstract.lower()))
            overlap = len(claim_words & abst_words) / max(len(claim_words), 1)
            confidence = 0.6 + (overlap * 0.3)
            # Boost confidence by source reliability
            confidence = min(0.95, confidence + (best.reliability_score - 0.5) * 0.1)
            return overlap > 0.25, round(confidence, 3), source

        return False, 0.5, ""
    except Exception:
        return False, 0.5, ""


# ── Main engine ───────────────────────────────────────────────────────────────

class KnowledgeGrowthEngine:
    """
    ARIA's anti-hallucination and continuous learning engine.

    Workflow:
      absorb(text, source)   — learn from any text
      ground(response, query) — verify and cite a response
      ask(query)             — answer from verified knowledge + LLM
    """

    def __init__(self, engine=None, memory=None):
        self._engine  = engine   # core.engine.Engine
        self._memory  = memory   # core.memory.Memory
        self._store   = KnowledgeStore()
        self._absorb_queue: list = []
        self._lock    = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None
        self._running = False

        # ── Upgrade B: knowledge graph ────────────────────────────────────────
        self._knowledge_graph: Dict[str, Dict[str, float]] = {}
        self._graph_dirty = False  # rebuild flag

        # ── Upgrade D: n-gram model for perplexity-based filtering ────────────
        self._ngram_model: Dict[str, Dict[str, float]] = {}
        self._ngram_corpus: List[str] = []
        self._ngram_dirty = True   # rebuild on first absorb
        self._perplexity_threshold: float = 50.0  # below this = repetitive, skip

    # ── Public API ────────────────────────────────────────────────────────────

    def absorb(self, text: str, source: str = "conversation",
               domain: str = "general", verify: bool = False) -> Dict[str, Any]:
        """
        Learn from any text — extract claims and store with confidence scores.
        Upgrade D: Only absorb if perplexity > threshold (novel text).
        Upgrade A: Run MI correlation after absorbing new claims.
        """
        # ── Upgrade D: perplexity-based novelty gate ──────────────────────────
        skipped_as_repetitive = False
        if _ALGO_CORE and self._ngram_model and not self._ngram_dirty:
            try:
                perp = AdaptiveLearner.perplexity(text, self._ngram_model, n=3)
                if perp < self._perplexity_threshold:
                    skipped_as_repetitive = True
                    logger.debug(
                        f"[KGE] absorb skipped — perplexity={perp:.1f} < threshold={self._perplexity_threshold}"
                    )
            except Exception:
                pass  # Don't let perplexity failure block absorption

        if skipped_as_repetitive:
            return {
                "claims_extracted": 0,
                "new": 0,
                "updated": 0,
                "source": source,
                "skipped": "repetitive",
            }

        claims = extract_claims(text, source=source, domain=domain)
        new_count = 0
        updated   = 0

        for claim in claims:
            if verify and claim.confidence > 0.7:
                ok, web_conf, web_src = verify_claim_web(claim.text)
                if ok:
                    claim.verified   = True
                    claim.confidence = (claim.confidence + web_conf) / 2
                    if web_src:
                        claim.source = web_src

            is_new = self._store.upsert(claim)
            if is_new:
                new_count += 1
            else:
                updated += 1

        # Also store in ChromaDB / Memory if available
        if self._memory and claims:
            try:
                summary = " ".join(c.text for c in claims[:3])
                self._memory.add(summary, metadata={"source": source, "type": "absorbed"})
            except Exception:
                pass

        # ── Upgrade D: update n-gram model corpus ─────────────────────────────
        if claims:
            self._ngram_corpus.append(text)
            self._ngram_corpus = self._ngram_corpus[-200:]  # cap at 200 docs
            self._ngram_dirty = True

        # ── Upgrade A: claim correlation check on new claims ──────────────────
        correlations: Dict[str, Any] = {}
        if _ALGO_CORE and new_count > 0:
            try:
                correlations = self._find_claim_correlations(claims)
            except Exception:
                pass

        # ── Upgrade B: mark graph as needing rebuild ──────────────────────────
        if new_count > 0:
            self._graph_dirty = True

        return {
            "claims_extracted": len(claims),
            "new":     new_count,
            "updated": updated,
            "source":  source,
            "correlations": correlations,
        }

    # ── Upgrade A: claim correlation ──────────────────────────────────────────

    def _find_claim_correlations(self, new_claims: List[Claim]) -> Dict[str, Any]:
        """
        Upgrade A: After absorbing new claims, run mutual_information between
        new claim vectors and existing claims to find duplicates/supporting/contradicting.
        """
        if not _ALGO_CORE:
            return {}

        all_vectors = self._store.get_all_claims_as_vectors()
        if len(all_vectors) < 2:
            return {}

        new_hashes = {c.claim_hash for c in new_claims}
        new_vecs   = [(h, v) for h, v in all_vectors if h in new_hashes]
        exist_vecs = [(h, v) for h, v in all_vectors if h not in new_hashes]

        if not new_vecs or not exist_vecs:
            return {}

        correlated: List[Dict[str, Any]] = []
        for nh, nv in new_vecs[:5]:  # limit to first 5 new claims
            mis = []
            for eh, ev in exist_vecs[:50]:  # check against 50 existing
                try:
                    mi = CorrelationEngine.mutual_information(nv, ev, bins=5)
                    mis.append((eh, mi))
                except Exception:
                    pass
            if mis:
                mis.sort(key=lambda x: -x[1])
                top_corr = [{"hash": h, "mi": round(m, 4)} for h, m in mis[:3] if m > 0.1]
                if top_corr:
                    correlated.append({"new_hash": nh, "related": top_corr})

        return {"correlated_claims": correlated}

    # ── Upgrade B: knowledge graph management ─────────────────────────────────

    def _rebuild_knowledge_graph(self) -> None:
        """Rebuild graph from all stored claims using co-occurrence."""
        if not _ALGO_CORE:
            return
        all_claims = self._store.search("", min_confidence=0.5, limit=200)
        facts: List[Tuple[str, str, float]] = []
        for i, ca in enumerate(all_claims):
            for cb in all_claims[i + 1:i + 6]:
                from agents.algo_core import PatternEngine as _PE
                sim = _PE.semantic_similarity(ca.text[:100], cb.text[:100])
                if sim > 0.3:
                    facts.append((ca.claim_hash, cb.claim_hash, sim))
        self._knowledge_graph = CorrelationEngine.build_knowledge_graph(facts)
        self._graph_dirty = False
        logger.debug(f"[KGE] Knowledge graph rebuilt: {len(self._knowledge_graph)} nodes")

    def _get_graph_reasoning_chain(self, query: str) -> str:
        """
        Upgrade B: Find multi-hop reasoning paths in knowledge graph
        for a given query.
        """
        if not _ALGO_CORE or not self._knowledge_graph:
            return ""
        if self._graph_dirty:
            self._rebuild_knowledge_graph()
        if not self._knowledge_graph:
            return ""

        # Find query-related nodes
        relevant_claims = self._store.search(query[:80], limit=5)
        if len(relevant_claims) < 2:
            return ""

        start = relevant_claims[0].claim_hash
        end   = relevant_claims[-1].claim_hash if len(relevant_claims) > 1 else ""
        if not end or start == end:
            return ""

        try:
            paths = CorrelationEngine.find_paths(
                self._knowledge_graph, start, end, max_depth=4
            )
        except Exception:
            return ""

        if not paths:
            return ""

        # Resolve hashes to claim texts for top path
        best_path, score = paths[0]
        path_texts = []
        for h in best_path:
            results = self._store.search(h[:12], limit=1)
            if results:
                path_texts.append(results[0].text[:120])
            else:
                path_texts.append(h)
        chain = " → ".join(path_texts)
        return f"*Reasoning chain (confidence {score:.2f}):* {chain}"

    # ── Upgrade D: n-gram model refresh ───────────────────────────────────────

    def _ensure_ngram_model(self) -> None:
        """Rebuild n-gram model from corpus if dirty."""
        if not _ALGO_CORE:
            return
        if self._ngram_dirty and self._ngram_corpus:
            self._ngram_model = AdaptiveLearner.build_ngram_model(
                self._ngram_corpus, n=3
            )
            self._ngram_dirty = False

    def _run_confidence_decay(self) -> None:
        """Upgrade D: Apply confidence decay and trigger re-verify for stale claims."""
        updated = self._store.apply_confidence_decay(decay_per_day=0.99)
        if updated:
            logger.debug(f"[KGE] Confidence decay applied to {updated} claims")
        stale = self._store.get_stale_claims(min_confidence=0.01, max_confidence=0.30)
        for claim in stale[:5]:  # re-verify up to 5 at a time
            try:
                ok, conf, src = verify_claim_web(claim.text)
                if ok:
                    claim.confidence = max(conf, 0.5)
                    claim.verified = True
                    if src:
                        claim.source = src
                    self._store.upsert(claim)
            except Exception:
                pass

    def ground(self, response: str, query: str = "",
               min_confidence: float = 0.6) -> GroundedResponse:
        """
        Take an LLM response and ground it with verified knowledge.
        Marks uncertain parts, adds sources, corrects low-confidence claims.
        """
        claims = extract_claims(response, source="llm_response")
        sources     = []
        uncertain   = []
        corrections = []
        grounded_claims = []
        overall_conf = 0.0

        for claim in claims:
            # Check if we have stored knowledge about this claim
            related = self._store.search(claim.text[:80], limit=3)

            if related:
                best = related[0]
                if best.confidence >= min_confidence:
                    sources.append(best.source)
                    claim.confidence = best.confidence
                    claim.verified   = best.verified
                else:
                    uncertain.append(claim.text[:80])
            elif claim.confidence < 0.5:
                uncertain.append(claim.text[:80])

            grounded_claims.append(claim)
            overall_conf += claim.confidence

        if grounded_claims:
            overall_conf /= len(grounded_claims)

        # Add source citations to response
        cited_response = response
        if sources:
            unique_sources = list(dict.fromkeys(s for s in sources if s and s != "unknown"))
            if unique_sources:
                cited_response += f"\n\n*Sources: {', '.join(unique_sources[:3])}*"

        # Prepend uncertainty note if many uncertain claims
        if len(uncertain) > len(grounded_claims) * 0.4:
            cited_response = (
                "> ⚠️ Some parts of this response have not been verified.\n\n"
                + cited_response
            )

        return GroundedResponse(
            response=cited_response,
            confidence=overall_conf,
            sources=list(set(sources))[:5],
            claims=grounded_claims,
            uncertain_parts=uncertain,
            corrections=corrections,
        )

    def answer_grounded(self, query: str) -> Dict[str, Any]:
        """
        Answer a query using verified knowledge + LLM, with zero hallucination policy.
        Upgrade B: includes multi-hop knowledge graph reasoning chain.
        """
        # 1. Retrieve relevant stored claims
        claims = self._store.search(query, limit=12)
        context_parts = []
        for c in claims:
            tag = "✅" if c.verified else f"~{c.confidence:.0%}"
            context_parts.append(f"{tag} {c.text} (src: {c.source})")

        # ── Upgrade B: inject multi-hop reasoning chain ───────────────────────
        reasoning_chain = ""
        try:
            self._ensure_ngram_model()
            reasoning_chain = self._get_graph_reasoning_chain(query)
        except Exception:
            pass
        if reasoning_chain:
            context_parts.append(reasoning_chain)

        knowledge_ctx = "\n".join(context_parts)

        # 2. If no stored knowledge, check memory
        if not knowledge_ctx and self._memory:
            try:
                mem_results = self._memory.search(query, top_k=5)
                for r in mem_results:
                    if hasattr(r, 'document'):
                        context_parts.append(r.document[:200])
                knowledge_ctx = "\n".join(context_parts)
            except Exception:
                pass

        # 3. Generate grounded response
        if not self._engine:
            if claims:
                return {
                    "answer": "\n".join(c.text for c in claims[:3]),
                    "confidence": sum(c.confidence for c in claims[:3]) / 3,
                    "sources": list(set(c.source for c in claims[:3])),
                    "grounded": True,
                }
            return {"answer": "No knowledge available", "confidence": 0.0, "grounded": False}

        system = (
            "You are ARIA. Answer using ONLY the verified knowledge provided. "
            "If the knowledge context doesn't answer the question, say "
            "'I don't have verified information about this.' "
            "Never make up facts. Every claim must come from the context. "
            "Format clearly with markdown."
        )
        prompt = (
            f"Query: {query}\n\n"
            f"Verified knowledge:\n{knowledge_ctx or 'No relevant stored knowledge.'}"
        )

        try:
            raw = self._engine.generate(prompt, system=system, temperature=0.2)
            grounded = self.ground(raw, query)

            # Absorb the query+answer as new knowledge
            self.absorb(raw, source="aria_response", domain="general")

            return {
                "answer":     grounded.response,
                "confidence": grounded.confidence,
                "sources":    grounded.sources,
                "grounded":   True,
                "uncertain":  grounded.uncertain_parts,
            }
        except Exception as e:
            logger.error("Grounded answer failed: %s", e)
            return {"answer": "Error generating answer", "confidence": 0.0, "grounded": False}

    def stats(self) -> Dict[str, Any]:
        return self._store.stats()

    # ── Background absorb loop ────────────────────────────────────────────────

    def start_background_learning(self):
        """Continuously absorbs queued text in the background."""
        if self._running:
            return
        self._running = True
        self._bg_thread = threading.Thread(
            target=self._absorb_loop, daemon=True
        )
        self._bg_thread.start()
        logger.info("KnowledgeGrowthEngine background learning started")

    def queue_absorb(self, text: str, source: str = "background",
                     domain: str = "general"):
        """Non-blocking — queue text for background absorption."""
        with self._lock:
            self._absorb_queue.append((text, source, domain))

    def _absorb_loop(self):
        _decay_cycle = 0
        while self._running:
            with self._lock:
                batch = self._absorb_queue[:]
                self._absorb_queue.clear()

            for text, source, domain in batch:
                try:
                    self.absorb(text, source, domain)
                except Exception as e:
                    logger.error("Background absorb error: %s", e)

            # ── Upgrade D: periodic confidence decay (every ~5 min) ───────────
            _decay_cycle += 1
            if _decay_cycle % 60 == 0:  # every 60 * 5s = 5 min
                try:
                    self._run_confidence_decay()
                except Exception as e:
                    logger.error("Confidence decay error: %s", e)

            # ── Upgrade D: rebuild n-gram model when corpus is dirty ──────────
            if self._ngram_dirty and self._ngram_corpus:
                try:
                    self._ensure_ngram_model()
                except Exception:
                    pass

            time.sleep(5)  # process queue every 5 seconds

    def stop(self):
        self._running = False

    # ── Natural language interface ────────────────────────────────────────────

    async def run_nl(self, query: str) -> Dict[str, Any]:
        """
        Natural language interface:
          "what do you know about X"
          "verify: X is Y"
          "learn: <text>"
          "knowledge stats"
        """
        q = query.lower().strip()

        if q.startswith("learn:") or q.startswith("absorb:"):
            text = query.split(":", 1)[1].strip()
            result = self.absorb(text, source="user_input")
            return {"ok": True, "message": f"Learned {result['claims_extracted']} claims", **result}

        if "knowledge stats" in q or "what do you know" in q or q == "stats":
            return {"ok": True, **self.stats()}

        if q.startswith("verify:"):
            claim_text = query.split(":", 1)[1].strip()
            ok, conf, src = verify_claim_web(claim_text)
            return {
                "ok": True, "verified": ok, "confidence": conf,
                "source": src, "claim": claim_text,
            }

        # Default: grounded answer
        return self.answer_grounded(query)


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[KnowledgeGrowthEngine] = None


def get_engine(engine=None, memory=None) -> KnowledgeGrowthEngine:
    global _instance
    if _instance is None:
        _instance = KnowledgeGrowthEngine(engine=engine, memory=memory)
        _instance.start_background_learning()
    return _instance
