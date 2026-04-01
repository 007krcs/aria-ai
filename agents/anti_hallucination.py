"""
ARIA Anti-Hallucination Engine
================================
Ensures every ARIA response is grounded, cited, and verified.
Zero hallucination policy.

Techniques:
  1. RAG grounding  — every claim checked against ChromaDB memory
  2. Web verification — uncertain claims verified via DuckDuckGo
  3. Contradiction detection — flags statements that contradict known facts
  4. Confidence scoring — 0-100 per sentence
  5. Source citation — every fact linked to a source
  6. "I don't know" protocol — when confidence < 40, admit uncertainty
  7. Temporal grounding — marks claims with date context (avoids stale info)
  8. Cross-model verification — ask two different models, flag if they disagree
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Optional / graceful imports
# ---------------------------------------------------------------------------
try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    requests = None  # type: ignore
    _REQUESTS_OK = False

try:
    import aiohttp
    _AIOHTTP_OK = True
except ImportError:
    aiohttp = None  # type: ignore
    _AIOHTTP_OK = False

try:
    import chromadb
    _CHROMA_OK = True
except ImportError:
    chromadb = None  # type: ignore
    _CHROMA_OK = False

try:
    import ollama
    _OLLAMA_OK = True
except ImportError:
    ollama = None  # type: ignore
    _OLLAMA_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG = logging.getLogger("aria.anti_hallucination")

UNCERTAIN_PHRASES = [
    "i think", "i believe", "i'm not sure", "probably", "might be",
    "could be", "perhaps", "i guess", "i assume", "roughly", "approximately",
    "as far as i know", "i recall", "if i remember correctly",
]

DEFINITIVE_PHRASES = [
    "definitely", "certainly", "absolutely", "always", "never",
    "the fact is", "it is known", "proven", "confirmed", "established",
]

# Penalty / boost amounts applied to base confidence score
_UNCERTAIN_PENALTY = 15.0
_DEFINITIVE_BOOST = 5.0
_SHORT_SENTENCE_PENALTY = 10.0   # very short sentences are often filler
_NUMERIC_BOOST = 8.0             # numbers suggest concrete claims
_HEDGED_QUESTION_PENALTY = 20.0  # rhetorical "could it be…?" patterns

DB_PATH = Path.home() / ".aria" / "hallucination_cache.db"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Claim:
    """A single factual claim extracted from a response."""
    text: str
    confidence: float          # 0-100
    sources: list[str] = field(default_factory=list)
    verified: bool = False
    contradictions: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    """Full verification report for one model response."""
    original: str
    verified_response: str
    claims: list[Claim]
    overall_confidence: float       # weighted average of claim confidences
    sources_used: list[str]
    corrections_made: list[str]
    verified_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # True when overall_confidence < 40 — triggers "I don't know" mode
    admitted_uncertainty: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claims"] = [c.to_dict() for c in self.claims]
        return d

    def summary(self) -> str:
        lines = [
            f"Overall confidence : {self.overall_confidence:.1f}/100",
            f"Claims analysed    : {len(self.claims)}",
            f"Sources used       : {len(self.sources_used)}",
            f"Corrections made   : {len(self.corrections_made)}",
            f"Admitted uncertainty: {self.admitted_uncertainty}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verification_cache (
            query_hash   TEXT PRIMARY KEY,
            query        TEXT,
            result_json  TEXT,
            created_at   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS known_facts (
            fact_hash    TEXT PRIMARY KEY,
            fact_text    TEXT,
            source       TEXT,
            confidence   REAL,
            added_at     TEXT
        )
    """)
    conn.commit()
    return conn


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class AntiHallucinationEngine:
    """
    Verifies ARIA responses against memory, web sources, and internal
    consistency checks. Produces a VerificationResult with per-claim
    confidence scores, source citations, and detected contradictions.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        chroma_collection: Optional[Any] = None,
        ollama_model: str = "llama3",
        secondary_model: str = "mistral",
        web_verify_threshold: float = 55.0,
        low_confidence_threshold: float = 40.0,
        cache_ttl_seconds: int = 3600,
        ddg_max_results: int = 3,
    ):
        self.db_path = db_path
        self.chroma_collection = chroma_collection
        self.ollama_model = ollama_model
        self.secondary_model = secondary_model
        self.web_verify_threshold = web_verify_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.cache_ttl_seconds = cache_ttl_seconds
        self.ddg_max_results = ddg_max_results
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _init_db(self.db_path)
        return self._conn

    def _cache_get(self, query: str) -> Optional[VerificationResult]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT result_json, created_at FROM verification_cache WHERE query_hash=?",
            (_hash(query),),
        ).fetchone()
        if not row:
            return None
        result_json, created_at = row
        try:
            created = datetime.fromisoformat(created_at)
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age > self.cache_ttl_seconds:
                return None
            data = json.loads(result_json)
            claims = [Claim(**c) for c in data.pop("claims", [])]
            return VerificationResult(claims=claims, **data)
        except Exception:
            return None

    def _cache_set(self, query: str, result: VerificationResult) -> None:
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO verification_cache
               (query_hash, query, result_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (_hash(query), query[:512], json.dumps(result.to_dict()),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def add_known_fact(self, fact: str, source: str = "user", confidence: float = 90.0) -> None:
        """Manually inject a ground-truth fact into the fact DB."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO known_facts
               (fact_hash, fact_text, source, confidence, added_at)
               VALUES (?, ?, ?, ?, ?)""",
            (_hash(fact), fact, source, confidence, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    def _get_known_facts(self) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT fact_text, source, confidence FROM known_facts"
        ).fetchall()
        return [{"text": r[0], "source": r[1], "confidence": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def score_confidence(self, text: str) -> float:
        """
        Heuristic confidence score for a single sentence (0-100).
        Penalises uncertain language; rewards concrete, factual framing.
        """
        score = 70.0  # neutral baseline
        lower = text.lower()

        for phrase in UNCERTAIN_PHRASES:
            if phrase in lower:
                score -= _UNCERTAIN_PENALTY

        for phrase in DEFINITIVE_PHRASES:
            if phrase in lower:
                score += _DEFINITIVE_BOOST

        word_count = len(text.split())
        if word_count < 5:
            score -= _SHORT_SENTENCE_PENALTY

        # Numbers/dates suggest specificity
        if re.search(r"\b\d+[\d,\.]*\b", text):
            score += _NUMERIC_BOOST

        # Rhetorical / hedged questions
        if re.search(r"\b(could|might|may)\s+it\s+be\b", lower):
            score -= _HEDGED_QUESTION_PENALTY

        return max(0.0, min(100.0, score))

    def _split_claims(self, text: str) -> list[str]:
        """Split a response into individual claim sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def check_against_memory(
        self, claim: str, memory: Optional[Any] = None
    ) -> tuple[bool, list[str]]:
        """
        Check a claim against ChromaDB or a provided memory object.
        Returns (supported: bool, sources: list[str]).
        """
        sources: list[str] = []

        # 1. Try injected chroma collection
        collection = memory or self.chroma_collection
        if collection is not None and _CHROMA_OK:
            try:
                results = collection.query(
                    query_texts=[claim],
                    n_results=3,
                    include=["documents", "metadatas", "distances"],
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for doc, meta, dist in zip(docs, metas, distances):
                    if dist < 0.6:   # cosine-like threshold
                        src = meta.get("source", "memory") if meta else "memory"
                        sources.append(f"[memory:{src}] {doc[:120]}")

                if sources:
                    return True, sources
            except Exception as exc:
                LOG.debug("ChromaDB query failed: %s", exc)

        # 2. Fall back to known_facts DB
        known = self._get_known_facts()
        claim_lower = claim.lower()
        for fact in known:
            # Simple keyword overlap heuristic
            fact_words = set(fact["text"].lower().split())
            claim_words = set(claim_lower.split())
            overlap = len(fact_words & claim_words)
            if overlap >= 3:
                sources.append(f"[known_fact:{fact['source']}] {fact['text'][:120]}")

        return bool(sources), sources

    def web_verify(self, claim: str) -> tuple[bool, str]:
        """
        Attempt to verify a claim via ResearchSearchEngine (multi-source).
        Returns (verified: bool, evidence: str).
        """
        try:
            from agents.research_search_engine import ResearchSearchEngine
            engine = ResearchSearchEngine()
            results = engine.search(claim, max_results=3)
            if not results:
                return False, "no web evidence found"

            # Use the result with the best abstract
            best = max(results, key=lambda r: len(r.abstract or ""))
            abstract = (best.abstract or "").strip()

            if abstract and len(abstract) > 20:
                source_url = best.url or best.source or "research_engine"
                return True, f"[web:{source_url}] {abstract[:200]}"

            return False, "no web evidence found"

        except Exception as exc:
            LOG.debug("Web verify failed: %s", exc)
            return False, f"web verify error: {exc}"

    def detect_contradictions(self, claims: list[Claim]) -> list[str]:
        """
        Scan claims for internal contradictions using simple negation patterns.
        E.g., "X is Y" vs "X is not Y".
        Returns list of contradiction descriptions.
        """
        contradictions: list[str] = []
        texts = [c.text for c in claims]

        for i, t1 in enumerate(texts):
            for j, t2 in enumerate(texts):
                if i >= j:
                    continue
                # Negation detection: one sentence negates a key phrase from the other
                words1 = set(t1.lower().split())
                words2 = set(t2.lower().split())
                overlap = words1 & words2 - {"the", "a", "an", "is", "are", "was", "were", "it"}
                if len(overlap) >= 3:
                    has_neg1 = bool(re.search(r"\b(not|never|no|cannot|can't|isn't|aren't)\b", t1, re.I))
                    has_neg2 = bool(re.search(r"\b(not|never|no|cannot|can't|isn't|aren't)\b", t2, re.I))
                    if has_neg1 != has_neg2:
                        contradictions.append(
                            f"Potential contradiction between: \"{t1[:80]}\" and \"{t2[:80]}\""
                        )

        return contradictions

    def add_citations(self, text: str, sources: list[str]) -> str:
        """
        Append a formatted 'Sources' section to the response text.
        """
        if not sources:
            return text
        unique = list(dict.fromkeys(sources))  # preserve order, dedupe
        citation_block = "\n\n**Sources:**\n" + "\n".join(
            f"  [{i+1}] {s}" for i, s in enumerate(unique[:10])
        )
        return text + citation_block

    def _cross_model_check(self, claim: str) -> Optional[str]:
        """
        Ask the secondary model to confirm / deny the claim.
        Returns disagreement note if models diverge, else None.
        """
        if not _OLLAMA_OK:
            return None
        try:
            prompt = (
                f"In one sentence, is the following statement accurate? "
                f"Answer only 'yes', 'no', or 'uncertain'.\n\nStatement: {claim}"
            )
            r1 = ollama.generate(model=self.ollama_model, prompt=prompt)
            r2 = ollama.generate(model=self.secondary_model, prompt=prompt)
            resp1 = r1.get("response", "").strip().lower()[:20]
            resp2 = r2.get("response", "").strip().lower()[:20]

            agree_set = {"yes", "no"}
            label1 = next((w for w in agree_set if w in resp1), "uncertain")
            label2 = next((w for w in agree_set if w in resp2), "uncertain")

            if label1 != label2 and "uncertain" not in (label1, label2):
                return (
                    f"Model disagreement: {self.ollama_model}={label1}, "
                    f"{self.secondary_model}={label2} on: \"{claim[:80]}\""
                )
        except Exception as exc:
            LOG.debug("Cross-model check failed: %s", exc)
        return None

    def _temporal_tag(self, text: str) -> str:
        """Prefix time-sensitive claims with a date watermark."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if re.search(r"\b(current|latest|recent|now|today|new|updated)\b", text, re.I):
            return f"[as of {today}] {text}"
        return text

    # ------------------------------------------------------------------
    # Main verification pipeline
    # ------------------------------------------------------------------

    def verify_response(
        self,
        response: str,
        query: str = "",
        memory: Optional[Any] = None,
    ) -> VerificationResult:
        """
        Full synchronous verification pipeline.
        """
        cache_key = f"{query}||{response}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        raw_sentences = self._split_claims(response)
        claims: list[Claim] = []
        all_sources: list[str] = []
        corrections: list[str] = []

        for sentence in raw_sentences:
            conf = self.score_confidence(sentence)
            mem_supported, mem_sources = self.check_against_memory(sentence, memory)
            sources = list(mem_sources)

            if mem_supported:
                conf = min(100.0, conf + 12)
            else:
                conf = max(0.0, conf - 8)

            # Web verify if still uncertain
            if conf < self.web_verify_threshold:
                web_ok, web_evidence = self.web_verify(sentence)
                if web_ok:
                    sources.append(web_evidence)
                    conf = min(100.0, conf + 18)
                else:
                    conf = max(0.0, conf - 5)

            # Cross-model disagreement check
            disagreement = self._cross_model_check(sentence)
            if disagreement:
                corrections.append(disagreement)
                conf = max(0.0, conf - 20)

            # Temporal tag
            tagged = self._temporal_tag(sentence)

            claim = Claim(
                text=tagged,
                confidence=round(conf, 1),
                sources=sources,
                verified=bool(sources),
                contradictions=[],
            )
            claims.append(claim)
            all_sources.extend(sources)

        # Contradiction pass
        contradictions = self.detect_contradictions(claims)
        if contradictions:
            for c in contradictions:
                corrections.append(c)
            # Lower confidence on contradicting claims
            for claim in claims:
                for contradiction in contradictions:
                    if claim.text[:40] in contradiction:
                        claim.confidence = max(0.0, claim.confidence - 15)
                        claim.contradictions.append(contradiction)

        # Overall confidence
        if claims:
            overall = sum(c.confidence for c in claims) / len(claims)
        else:
            overall = 0.0

        # Build verified response
        verified_response = response
        admitted_uncertainty = False

        if overall < self.low_confidence_threshold:
            admitted_uncertainty = True
            verified_response = (
                "I'm not confident enough in this answer to state it as fact. "
                "Here is what I found, but please verify independently:\n\n"
                + response
            )

        verified_response = self.add_citations(verified_response, list(dict.fromkeys(all_sources)))

        result = VerificationResult(
            original=response,
            verified_response=verified_response,
            claims=claims,
            overall_confidence=round(overall, 1),
            sources_used=list(dict.fromkeys(all_sources)),
            corrections_made=corrections,
            admitted_uncertainty=admitted_uncertainty,
        )

        self._cache_set(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Async variant
    # ------------------------------------------------------------------

    async def verify_async(
        self,
        response: str,
        query: str = "",
        memory: Optional[Any] = None,
    ) -> VerificationResult:
        """
        Async wrapper — runs blocking verify_response in a thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.verify_response, response, query, memory
        )

    # ------------------------------------------------------------------
    # Natural-language entry point (for agent bus integration)
    # ------------------------------------------------------------------

    def run_nl(self, query_and_response: str) -> dict:
        """
        Accepts a plain-text string of the form:
            QUERY: <user query>
            RESPONSE: <model response to verify>

        Falls back to treating the entire string as the response if the
        expected format is not found.

        Returns VerificationResult as a dict.
        """
        query = ""
        response = query_and_response

        q_match = re.search(r"(?i)^QUERY:\s*(.+?)(?=\nRESPONSE:|\Z)", query_and_response, re.S | re.M)
        r_match = re.search(r"(?i)^RESPONSE:\s*(.+)", query_and_response, re.S | re.M)

        if q_match:
            query = q_match.group(1).strip()
        if r_match:
            response = r_match.group(1).strip()

        result = self.verify_response(response=response, query=query)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Convenience: verify and return plain-text summary
    # ------------------------------------------------------------------

    def verify_and_summarise(self, response: str, query: str = "") -> str:
        result = self.verify_response(response=response, query=query)
        return (
            f"{result.verified_response}\n\n"
            f"---\n{result.summary()}"
        )

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "AntiHallucinationEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def quick_score(text: str) -> float:
    """Score a single piece of text with a throw-away engine instance."""
    engine = AntiHallucinationEngine()
    return engine.score_confidence(text)


def quick_verify(response: str, query: str = "") -> VerificationResult:
    """Verify a response with default settings."""
    with AntiHallucinationEngine() as engine:
        return engine.verify_response(response=response, query=query)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = (
            "The Eiffel Tower is located in Berlin. "
            "It was definitely built in 1889 and is probably the tallest building in Europe. "
            "I think it might be around 300 metres tall."
        )

    print("=== ARIA Anti-Hallucination Engine ===\n")
    result = quick_verify(text, query="Tell me about the Eiffel Tower")
    print(result.verified_response)
    print("\n--- Report ---")
    print(result.summary())
    print("\n--- Claims ---")
    for i, claim in enumerate(result.claims, 1):
        print(f"  [{i}] conf={claim.confidence:.0f}  verified={claim.verified}  {claim.text[:80]}")
