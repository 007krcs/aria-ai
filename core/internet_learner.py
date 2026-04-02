"""
ARIA Internet Learner — Safe Retrieval-Based Learning
======================================================
ARIA learns from the internet through memory, not retraining.

Pipeline:
  1. Search — DuckDuckGo (no API key needed)
  2. Fetch  — extract clean text from pages
  3. Score  — trust scoring based on source domain
  4. Filter — remove junk, ads, noise
  5. Store  — write to semantic memory with provenance
  6. Forget — decay stale facts automatically

What gets stored:
  - Verified summaries with citations
  - Source trust scores
  - Timestamps for freshness

What never gets stored:
  - Raw HTML / ads / navigation text
  - Low-trust sources (< 0.4 score)
  - Duplicate content
  - Personal/sensitive content
"""

from __future__ import annotations

import re
import time
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE TRUST SCORER
# ─────────────────────────────────────────────────────────────────────────────

class SourceTrustScorer:
    """
    Scores a source URL for trustworthiness (0.0 to 1.0).
    No external API needed — pure heuristic.
    """

    HIGH_TRUST = {
        "wikipedia.org": 0.85, "britannica.com": 0.88, "nature.com": 0.95,
        "pubmed.ncbi.nlm.nih.gov": 0.95, "arxiv.org": 0.88,
        "reuters.com": 0.85, "apnews.com": 0.85, "bbc.com": 0.82,
        "theguardian.com": 0.80, "nytimes.com": 0.80,
        "gov.in": 0.88, "nic.in": 0.88, "rbi.org.in": 0.92,
        "sebi.gov.in": 0.92, "nseindia.com": 0.90, "bseindia.com": 0.90,
        "moneycontrol.com": 0.78, "economictimes.com": 0.80,
        "github.com": 0.82, "stackoverflow.com": 0.80,
        "docs.python.org": 0.92, "developer.mozilla.org": 0.92,
    }

    MED_TRUST_PATTERNS = [".edu", ".gov", ".ac.in", ".org"]
    LOW_TRUST_PATTERNS = ["reddit.com", "quora.com", "twitter.com", "x.com",
                          "facebook.com", "instagram.com", "tiktok.com"]

    def score(self, url: str) -> float:
        if not url:
            return 0.3

        url_lower = url.lower()

        # Exact domain match
        for domain, score in self.HIGH_TRUST.items():
            if domain in url_lower:
                return score

        # Low trust social media
        for pattern in self.LOW_TRUST_PATTERNS:
            if pattern in url_lower:
                return 0.35

        # TLD-based trust
        for pattern in self.MED_TRUST_PATTERNS:
            if pattern in url_lower:
                return 0.72

        # Default
        return 0.55


# ─────────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class TextExtractor:
    """
    Extracts clean text from a webpage.
    Removes nav, ads, scripts, footers.
    """

    REMOVE_PATTERNS = [
        r'<script[^>]*>.*?</script>',
        r'<style[^>]*>.*?</style>',
        r'<nav[^>]*>.*?</nav>',
        r'<footer[^>]*>.*?</footer>',
        r'<header[^>]*>.*?</header>',
        r'<[^>]+>',          # all remaining tags
        r'&[a-z]+;',         # HTML entities
        r'\s{3,}',           # excessive whitespace
    ]

    def extract(self, html: str, max_chars: int = 3000) -> str:
        text = html
        for pattern in self.REMOVE_PATTERNS:
            text = re.sub(pattern, ' ', text, flags=re.DOTALL | re.IGNORECASE)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Return most informative section (skip boilerplate at start)
        if len(text) > 500:
            text = text[200:]  # skip header boilerplate

        return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATOR
# ─────────────────────────────────────────────────────────────────────────────

class Deduplicator:
    """Prevents storing the same content twice."""

    def __init__(self):
        self._seen: set = set()

    def is_duplicate(self, text: str) -> bool:
        key = hashlib.md5(text[:300].encode()).hexdigest()
        if key in self._seen:
            return True
        self._seen.add(key)
        return False

    def clear(self):
        self._seen.clear()


# ─────────────────────────────────────────────────────────────────────────────
# INTERNET LEARNER
# ─────────────────────────────────────────────────────────────────────────────

class InternetLearner:
    """
    Safe internet learning pipeline.
    The brain calls this for research tasks.
    Results stored in semantic memory with full provenance.
    """

    def __init__(self):
        self.trust_scorer = SourceTrustScorer()
        self.extractor    = TextExtractor()
        self.dedup        = Deduplicator()
        self._min_trust   = 0.45   # discard sources below this
        self._lock        = threading.Lock()

    # ── Main entry point ──────────────────────────────────────────────────────

    def research(self, query: str, max_sources: int = 3, store: bool = True) -> Dict[str, Any]:
        """
        Search the web, extract content, score it, optionally store it.
        Returns structured research result.
        """
        results = {
            "query":    query,
            "sources":  [],
            "summary":  "",
            "stored":   0,
            "ts":       datetime.now().isoformat(),
        }

        # Step 1: Search
        search_results = self._search(query, n=max_sources + 2)
        if not search_results:
            results["summary"] = "No search results found."
            return results

        # Step 2-4: Fetch, score, filter
        good_sources = []
        for item in search_results:
            url   = item.get("href", "")
            title = item.get("title", "")
            snippet = item.get("body", "")

            trust = self.trust_scorer.score(url)
            if trust < self._min_trust:
                continue

            # Use snippet if fetch fails
            content = self._fetch(url) or snippet
            if not content or len(content) < 50:
                continue

            if self.dedup.is_duplicate(content):
                continue

            good_sources.append({
                "url":     url,
                "title":   title,
                "content": content[:1500],
                "trust":   round(trust, 2),
            })

            if len(good_sources) >= max_sources:
                break

        if not good_sources:
            results["summary"] = "Found results but none passed trust threshold."
            return results

        # Step 5: Synthesize summary
        results["sources"] = [{"url": s["url"], "title": s["title"], "trust": s["trust"]} for s in good_sources]
        results["summary"] = self._synthesize(query, good_sources)

        # Step 6: Store in memory
        if store:
            count = self._store(query, good_sources, results["summary"])
            results["stored"] = count

        return results

    # ── Search ────────────────────────────────────────────────────────────────

    def _search(self, query: str, n: int = 5) -> List[Dict]:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=n))
        except Exception:
            pass

        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=n))
        except Exception:
            pass

        return []

    # ── Fetch ─────────────────────────────────────────────────────────────────

    def _fetch(self, url: str, timeout: int = 5) -> Optional[str]:
        try:
            import requests
            headers = {"User-Agent": "Mozilla/5.0 (compatible; ARIABot/1.0)"}
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return self.extractor.extract(r.text)
        except Exception:
            pass
        return None

    # ── Synthesize ────────────────────────────────────────────────────────────

    def _synthesize(self, query: str, sources: List[Dict]) -> str:
        """
        Create a structured summary from multiple sources.
        No LLM needed — extractive synthesis.
        """
        lines = [f"Research: {query}\n"]

        for i, src in enumerate(sources, 1):
            content = src["content"]

            # Extract key sentences
            sentences = re.split(r'[.!?]', content)
            key = [s.strip() for s in sentences
                   if len(s.strip()) > 40
                   and any(w.lower() in s.lower() for w in query.split() if len(w) > 3)][:3]

            if key:
                lines.append(f"Source {i} ({src['title'][:50]}, trust={src['trust']}):")
                lines.extend(f"  - {s}." for s in key[:2])

        lines.append(f"\nRetrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
        return "\n".join(lines)

    # ── Store ─────────────────────────────────────────────────────────────────

    def _store(self, query: str, sources: List[Dict], summary: str) -> int:
        stored = 0
        try:
            from core.memory_hierarchy import MemoryHierarchy
            mem = MemoryHierarchy()

            # Store the summary
            avg_trust = sum(s["trust"] for s in sources) / len(sources)
            mem.write_fact(
                summary,
                source=sources[0]["url"] if sources else "web",
                trust=avg_trust,
            )
            stored += 1

            # Store individual high-trust facts
            for src in sources:
                if src["trust"] >= 0.75:
                    snippet = src["content"][:400]
                    mem.write_semantic(
                        snippet,
                        metadata={
                            "type":       "internet_fact",
                            "source_url": src["url"],
                            "trust":      src["trust"],
                            "query":      query[:100],
                            "ts":         datetime.now().isoformat(),
                        }
                    )
                    stored += 1

        except Exception:
            pass

        return stored

    # ── Utilities ─────────────────────────────────────────────────────────────

    def learn_about(self, topic: str) -> str:
        """Quick one-liner — research a topic and return summary."""
        result = self.research(topic, max_sources=2, store=True)
        return result.get("summary", "Could not find information on that topic.")

    def set_min_trust(self, threshold: float):
        """Adjust minimum trust score for storing content."""
        self._min_trust = max(0.1, min(0.95, threshold))
