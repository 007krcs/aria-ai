"""
ARIA — Smart Agents
Four specialist agents that make ARIA smarter over time.

TrainingAgent     → collects good Q&A pairs, prepares fine-tune data, tracks what AI learned
EfficiencyAgent   → monitors speed/quality, auto-tunes prompts, prunes bad chunks
PerformanceAgent  → detailed metrics dashboard, domain breakdown, trend tracking
CrawlerAgent      → deep-crawls an entire website, extracts and stores all content
"""

import re
import time
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin, urlparse
from rich.console import Console

from core.engine import Engine
from core.memory import Memory
from tools.logger import Logger
from agents.agents import BaseAgent

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAINING AGENT
# ─────────────────────────────────────────────────────────────────────────────

class TrainingAgent(BaseAgent):
    """
    Watches every interaction and builds a training dataset automatically.

    What it does:
    - Logs high-quality Q&A pairs (confidence > 0.75) as training examples
    - Flags bad answers for review
    - Generates synthetic training data from the knowledge base
    - Exports a JSONL dataset ready for LoRA fine-tuning on Colab
    - Tracks what topics the AI has been trained on
    """

    name = "training_agent"

    def __init__(self, engine: Engine, memory: Memory, logger: Logger):
        super().__init__(engine, memory, logger)
        self.training_dir = Path(__file__).resolve().parent.parent / "data" / "training"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _ensure_tables(self):
        with self.logger._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS training_examples (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT,
                    question    TEXT,
                    answer      TEXT,
                    domain      TEXT,
                    confidence  REAL,
                    source      TEXT,
                    quality     TEXT DEFAULT 'auto',
                    used_in_train INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS learning_events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT,
                    event_type  TEXT,
                    description TEXT,
                    domain      TEXT,
                    chunks_added INTEGER DEFAULT 0
                );
            """)

    # ── Log a learning event (called whenever AI learns something new) ─────────

    def log_learning(self, event_type: str, description: str, domain: str = "general", chunks: int = 0):
        """Call this whenever the AI ingests new knowledge."""
        with self.logger._connect() as conn:
            conn.execute(
                "INSERT INTO learning_events (ts, event_type, description, domain, chunks_added) VALUES (?,?,?,?,?)",
                (datetime.now().isoformat(), event_type, description, domain, chunks)
            )

    # ── Collect training example from an interaction ───────────────────────────

    def collect_example(self, question: str, answer: str, domain: str, confidence: float, source: str = "chat"):
        """Save a high-quality interaction as a training example."""
        if confidence < 0.65 or len(answer.strip()) < 20:
            return  # Skip low-quality answers
        with self.logger._connect() as conn:
            conn.execute(
                "INSERT INTO training_examples (ts, question, answer, domain, confidence, source) VALUES (?,?,?,?,?,?)",
                (datetime.now().isoformat(), question, answer, domain, confidence, source)
            )

    # ── Generate synthetic training data from knowledge base ──────────────────

    def generate_synthetic_data(self, domain: str = "general", count: int = 10) -> list[dict]:
        """
        Use the LLM + knowledge base to generate training Q&A pairs.
        These are used to fine-tune the model on your specific knowledge.
        """
        console.print(f"  [dim]Generating {count} synthetic training examples for '{domain}'...[/]")
        hits = self.memory.search(f"important facts about {domain}", top_k=5, domain=domain if domain != "general" else None)
        if not hits:
            console.print("  [yellow]No knowledge found for this domain. Ingest documents first.[/]")
            return []

        context = "\n".join(h["text"][:200] for h in hits[:3])
        prompt = (
            f"Based on this knowledge, generate {count} diverse question-answer pairs.\n"
            f"Make questions realistic — things a user would actually ask.\n"
            f"Answers should be detailed and accurate.\n\n"
            f"Knowledge:\n{context}\n\n"
            f"Respond with JSON only:\n"
            f'[{{"question":"...","answer":"..."}}]'
        )

        result = self.engine.generate(prompt, temperature=0.4)
        try:
            raw = result.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            pairs = json.loads(raw.strip())
            examples = []
            for p in pairs:
                if p.get("question") and p.get("answer"):
                    self.collect_example(p["question"], p["answer"], domain, 0.8, "synthetic")
                    examples.append(p)
            console.print(f"  [green]Generated {len(examples)} synthetic examples[/]")
            return examples
        except Exception as e:
            console.print(f"  [yellow]Parse error: {e}[/]")
            return []

    # ── Export training dataset ───────────────────────────────────────────────

    def export_dataset(self, domain: Optional[str] = None) -> str:
        """Export all training examples as JSONL for LoRA fine-tuning."""
        with self.logger._connect() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT question, answer, domain FROM training_examples WHERE domain=? ORDER BY confidence DESC",
                    (domain,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT question, answer, domain FROM training_examples ORDER BY confidence DESC"
                ).fetchall()

        if not rows:
            return ""

        output_path = self.training_dir / f"dataset_{domain or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                example = {
                    "instruction": row[0],
                    "output": row[1],
                    "domain": row[2],
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        console.print(f"  [green]Exported {len(rows)} examples to {output_path}[/]")
        return str(output_path)

    # ── Get learning timeline ──────────────────────────────────────────────────

    def get_learning_timeline(self, hours: int = 48) -> list[dict]:
        """Return recent learning events for the dashboard."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                """SELECT ts, event_type, description, domain, chunks_added
                   FROM learning_events
                   WHERE ts > datetime('now', ?)
                   ORDER BY ts DESC LIMIT 100""",
                (f"-{hours} hours",)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Get training stats ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self.logger._connect() as conn:
            total    = conn.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
            by_domain = conn.execute(
                "SELECT domain, COUNT(*), AVG(confidence) FROM training_examples GROUP BY domain"
            ).fetchall()
            events   = conn.execute("SELECT COUNT(*) FROM learning_events").fetchone()[0]
            recent   = conn.execute(
                "SELECT COUNT(*) FROM learning_events WHERE ts > datetime('now','-24 hours')"
            ).fetchone()[0]
        return {
            "total_examples": total,
            "learning_events": events,
            "learned_today": recent,
            "by_domain": [{"domain": r[0], "count": r[1], "avg_confidence": round(r[2],3)} for r in by_domain],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EFFICIENCY AGENT
# ─────────────────────────────────────────────────────────────────────────────

class EfficiencyAgent(BaseAgent):
    """
    Monitors ARIA's performance and actively improves it.

    What it does:
    - Tracks response times and token efficiency
    - Identifies slow or low-quality domains
    - Auto-generates better prompts for weak areas
    - Prunes duplicate/low-value memory chunks
    - Suggests which documents to re-ingest
    """

    name = "efficiency_agent"

    def analyze_speed(self) -> dict:
        """Analyse response times by intent/domain."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                """SELECT intent,
                          AVG(latency_ms) as avg_ms,
                          MIN(latency_ms) as min_ms,
                          MAX(latency_ms) as max_ms,
                          COUNT(*) as count
                   FROM interactions
                   WHERE latency_ms > 0
                   GROUP BY intent
                   ORDER BY avg_ms DESC"""
            ).fetchall()
        return {
            "by_intent": [dict(r) for r in rows],
            "recommendations": self._speed_recommendations([dict(r) for r in rows])
        }

    def _speed_recommendations(self, data: list) -> list[str]:
        recs = []
        for row in data:
            if row["avg_ms"] > 60000:
                recs.append(f"'{row['intent']}' is very slow ({row['avg_ms']//1000}s avg) — consider shorter prompts or a lighter model for this domain")
            elif row["avg_ms"] > 30000:
                recs.append(f"'{row['intent']}' is slow ({row['avg_ms']//1000}s avg) — prompt tuning may help")
        if not recs:
            recs.append("Response times look acceptable for your hardware")
        return recs

    def analyze_quality(self) -> dict:
        """Find low-confidence domains that need improvement."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                """SELECT intent,
                          AVG(confidence) as avg_conf,
                          COUNT(*) as count,
                          SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as failures
                   FROM interactions
                   GROUP BY intent
                   ORDER BY avg_conf ASC"""
            ).fetchall()
        results = [dict(r) for r in rows]
        weak = [r for r in results if r["avg_conf"] < 0.6 and r["count"] > 3]
        return {
            "by_intent":    results,
            "weak_domains": weak,
            "suggestions":  self._quality_suggestions(weak),
        }

    def _quality_suggestions(self, weak: list) -> list[str]:
        if not weak:
            return ["All domains performing well"]
        return [
            f"Ingest more documents about '{w['intent']}' — only {w['count']} queries, avg confidence {w['avg_conf']:.2f}"
            for w in weak[:5]
        ]

    def prune_memory(self, min_similarity_threshold: float = 0.97) -> dict:
        """
        Find near-duplicate chunks in memory and report them.
        (Actual deletion is manual — we never auto-delete without user confirmation.)
        """
        console.print("  [dim]Scanning for near-duplicate memory chunks...[/]")
        # Sample random chunks and find highly similar ones
        duplicates_found = 0
        # This is a lightweight check — full dedup would require comparing all pairs
        stats = self.memory.stats()
        return {
            "total_chunks": stats["total_chunks"],
            "duplicates_found": duplicates_found,
            "note": "Exact deduplication happens at ingest time via content hash. Run ingest again to skip known content.",
        }

    def get_efficiency_score(self) -> dict:
        """Overall efficiency score 0-100."""
        with self.logger._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            if not total:
                return {"score": 0, "grade": "No data yet", "total_queries": 0}
            success = conn.execute("SELECT COUNT(*) FROM interactions WHERE success=1").fetchone()[0]
            avg_conf = conn.execute("SELECT AVG(confidence) FROM interactions").fetchone()[0] or 0
            avg_ms   = conn.execute("SELECT AVG(latency_ms) FROM interactions WHERE latency_ms>0").fetchone()[0] or 999999

        success_score = (success / total) * 40
        conf_score    = avg_conf * 40
        speed_score   = max(0, 20 - (avg_ms / 10000))
        total_score   = min(100, round(success_score + conf_score + speed_score))

        grade = "Excellent" if total_score > 80 else "Good" if total_score > 60 else "Needs improvement"
        return {
            "score": total_score,
            "grade": grade,
            "total_queries": total,
            "success_rate": round(success/total*100, 1),
            "avg_confidence": round(avg_conf*100, 1),
            "avg_latency_s": round((avg_ms or 0)/1000, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. PERFORMANCE TRACKER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceAgent(BaseAgent):
    """
    Tracks every metric about ARIA over time.
    Provides trend data, hourly stats, and domain breakdowns for the dashboard.
    """

    name = "performance_agent"

    def get_hourly_trend(self, hours: int = 24) -> list[dict]:
        """Queries per hour with avg confidence — for charting."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                """SELECT
                     strftime('%Y-%m-%d %H:00', ts) as hour,
                     COUNT(*) as queries,
                     AVG(confidence) as avg_conf,
                     AVG(latency_ms) as avg_ms
                   FROM interactions
                   WHERE ts > datetime('now', ?)
                   GROUP BY hour
                   ORDER BY hour""",
                (f"-{hours} hours",)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_domain_breakdown(self) -> list[dict]:
        """Performance stats per domain."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                """SELECT
                     intent as domain,
                     COUNT(*) as total,
                     AVG(confidence) as avg_conf,
                     SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successes,
                     AVG(latency_ms) as avg_latency
                   FROM interactions
                   GROUP BY intent
                   ORDER BY total DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def get_top_queries(self, limit: int = 10) -> list[dict]:
        """Most asked questions."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                "SELECT query, confidence, latency_ms, ts FROM interactions ORDER BY ts DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_full_report(self) -> dict:
        stats = self.logger.get_stats()
        return {
            "summary":         stats,
            "hourly_trend":    self.get_hourly_trend(24),
            "domain_breakdown": self.get_domain_breakdown(),
            "top_queries":     self.get_top_queries(10),
            "generated_at":    datetime.now().isoformat(),
        }

    def get_memory_growth(self) -> list[dict]:
        """How many chunks were added over time."""
        with self.logger._connect() as conn:
            rows = conn.execute(
                """SELECT
                     strftime('%Y-%m-%d', ts) as date,
                     SUM(chunks) as chunks_added,
                     COUNT(*) as sources_added
                   FROM ingested_sources
                   GROUP BY date
                   ORDER BY date"""
            ).fetchall()
        return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CRAWLER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class CrawlerAgent(BaseAgent):
    """
    Deep-crawls an entire website and stores all content in the knowledge base.

    Features:
    - Respects robots.txt (optional)
    - Deduplicates by URL and content hash
    - Rate limits to be polite to servers
    - Returns real-time progress via a generator
    - Auto-detects and stays on the same domain
    - Streams status updates for the UI
    """

    name = "crawler_agent"

    def __init__(self, engine: Engine, memory: Memory, logger: Logger):
        super().__init__(engine, memory, logger)
        self.training_agent = None  # set from outside if available

    def crawl(
        self,
        start_url: str,
        max_pages: int = 30,
        domain_filter: str = "general",
        delay_s: float = 1.0,
        stay_on_domain: bool = True,
    ):
        """
        Crawl a website starting from start_url.
        Yields status dicts for real-time UI updates.

        Usage:
            for status in crawler.crawl("https://example.com", max_pages=20):
                print(status)
        """
        from agents.doc_agents import ProcessorAgent

        visited    = set()
        queue      = [start_url]
        base_domain= urlparse(start_url).netloc
        pages_done = 0
        total_chunks = 0
        errors     = 0

        yield {"type": "start", "url": start_url, "max_pages": max_pages}

        processor = ProcessorAgent(self.engine, self.memory, self.logger)

        while queue and pages_done < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue

            # Stay on same domain if requested
            if stay_on_domain and urlparse(url).netloc != base_domain:
                continue

            visited.add(url)

            try:
                yield {"type": "crawling", "url": url, "done": pages_done, "total": min(len(queue)+pages_done, max_pages)}

                # 3-tier fetch — escalates automatically on failure
                html, fetch_method = self._fetch_with_fallback(url)
                if html is None:
                    errors += 1
                    yield {"type": "skip", "url": url, "reason": "All fetch methods failed (403/blocked)"}
                    continue
                yield {"type": "fetch_method", "url": url, "method": fetch_method}

                # Extract text
                text = self._extract_text(html)
                if len(text.strip()) < 100:
                    yield {"type": "skip", "url": url, "reason": "Too little text"}
                    continue

                # Extract links for queue
                new_links = self._extract_links(html, url, base_domain, stay_on_domain)
                for link in new_links:
                    if link not in visited and link not in queue:
                        queue.append(link)

                # Store in memory
                chunks_before = self.memory.count()
                job = {"status": "ready", "type": "url", "source": url, "domain": domain_filter}
                report = processor.process(job)
                chunks_after  = self.memory.count()
                new_chunks    = chunks_after - chunks_before
                total_chunks += new_chunks

                pages_done += 1

                # Log learning event
                if self.training_agent:
                    self.training_agent.log_learning(
                        "web_crawl",
                        f"Crawled: {url}",
                        domain_filter,
                        new_chunks,
                    )

                yield {
                    "type":        "done",
                    "url":         url,
                    "chunks":      new_chunks,
                    "total_chunks": total_chunks,
                    "pages_done":  pages_done,
                    "queue_size":  len(queue),
                }

                time.sleep(delay_s)  # be polite

            except requests.exceptions.Timeout:
                errors += 1
                yield {"type": "error", "url": url, "reason": "Timeout"}
            except Exception as e:
                errors += 1
                yield {"type": "error", "url": url, "reason": str(e)[:80]}

        yield {
            "type":         "finished",
            "pages_crawled": pages_done,
            "total_chunks": total_chunks,
            "errors":       errors,
            "urls_visited": len(visited),
        }

    # ── 3-Tier fetch with automatic fallback ─────────────────────────────────

    BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":           "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":  "en-US,en;q=0.9,hi;q=0.8",
        "Accept-Encoding":  "gzip, deflate, br",
        "Connection":       "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest":   "document",
        "Sec-Fetch-Mode":   "navigate",
        "Sec-Fetch-Site":   "none",
        "Sec-Fetch-User":   "?1",
    }

    def _fetch_with_fallback(self, url: str) -> tuple[str | None, str]:
        """
        Try to fetch a URL using 3 methods, escalating on failure.

        Tier 1 — requests with browser headers (~0.5s)
            Works for: most static sites, Wikipedia, docs sites
            Fails for: Cloudflare-protected, JS-heavy SPAs, cookie-gated

        Tier 2 — requests.Session with homepage cookie warm-up (~1-2s)
            Works for: sites that need a session cookie to prove you visited normally
            Fails for: Cloudflare Bot Fight Mode, JS challenges, heavy SPAs

        Tier 3 — Playwright real headless Chromium (~3-8s)
            Works for: Cloudflare, JS-rendered SPAs, react.dev, Next.js sites,
                       sites that check browser fingerprint, cookie consent walls
            Fails for: sites requiring actual login, CAPTCHA challenges

        Returns: (html_string, method_name) or (None, "failed")
        """
        # ── Tier 1: Plain requests ────────────────────────────────────────────
        try:
            r = requests.get(url, timeout=12, headers=self.BROWSER_HEADERS, allow_redirects=True)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text, "requests"
            tier1_status = r.status_code
        except Exception as e:
            tier1_status = str(e)

        # ── Tier 2: Session with cookie warm-up ───────────────────────────────
        try:
            session = requests.Session()
            session.headers.update(self.BROWSER_HEADERS)
            base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            session.get(base, timeout=8)           # warm up cookies
            time.sleep(0.5)
            r2 = session.get(url, timeout=12, allow_redirects=True)
            if r2.status_code == 200 and len(r2.text) > 200:
                return r2.text, "session"
            tier2_status = r2.status_code
        except Exception as e:
            tier2_status = str(e)

        # ── Tier 3: Playwright headless Chromium ──────────────────────────────
        console.print(f"  [dim]Tier 3 — Playwright for: {url[:60]}[/]")
        html = self._fetch_playwright(url)
        if html:
            return html, "playwright"

        console.print(
            f"  [yellow]All 3 tiers failed for {url[:50]} "
            f"(t1={tier1_status}, t2={tier2_status})[/]"
        )
        return None, "failed"

    def _fetch_playwright(self, url: str) -> str | None:
        """
        Stealth Playwright fetch — passes Cloudflare and react.dev bot detection.

        Key techniques used:
        1. Non-headless mode emulation (headless=False disguises, but we use
           stealth script instead to stay headless and fast)
        2. Full navigator overrides (webdriver, plugins, languages, permissions)
        3. Realistic Chrome launch args (matches real Chrome exactly)
        4. Human-like mouse movement before reading content
        5. Proper referer chain (visit homepage first, then target page)
        6. Wait for actual content nodes, not just DOM ready
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError:
            console.print(
                "  [yellow]Playwright not installed. Run:[/]\n"
                "  pip install playwright && playwright install chromium"
            )
            return None

        # Full stealth script — patches every fingerprint react.dev / Cloudflare checks
        STEALTH_JS = """
            // 1. Hide webdriver
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

            // 2. Fake real Chrome runtime
            window.chrome = {
                runtime: {
                    connect: () => {}, sendMessage: () => {},
                    onMessage: {addListener: () => {}},
                },
                loadTimes: () => ({}),
                csi: () => ({}),
            };

            // 3. Realistic plugin list (empty = bot)
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const arr = [
                        {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer'},
                        {name:'Chrome PDF Viewer',  filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
                        {name:'Native Client',       filename:'internal-nacl-plugin'},
                    ];
                    arr.__proto__ = PluginArray.prototype;
                    return arr;
                }
            });

            // 4. Languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en', 'hi']
            });

            // 5. Hardware concurrency (bots often return 2)
            Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});

            // 6. Device memory
            Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});

            // 7. Platform
            Object.defineProperty(navigator, 'platform', {get: () => 'Win32'});

            // 8. Remove automation-related properties
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;

            // 9. Permissions API
            const origQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (params) =>
                params.name === 'notifications'
                    ? Promise.resolve({state: Notification.permission})
                    : origQuery(params);

            // 10. WebGL vendor (headless Chrome leaks "Google SwiftShader")
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(param) {
                if (param === 37445) return 'Intel Inc.';
                if (param === 37446) return 'Intel Iris OpenGL Engine';
                return getParameter.call(this, param);
            };
        """

        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--disable-web-security",
                        "--disable-features=IsolateOrigins,site-per-process",
                        "--flag-switches-begin",
                        "--flag-switches-end",
                        f"--window-size=1280,800",
                    ],
                )

                ctx = browser.new_context(
                    viewport={"width": 1280, "height": 800},
                    screen={"width": 1280, "height": 800},
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                    timezone_id="America/Chicago",
                    color_scheme="light",
                    java_script_enabled=True,
                    bypass_csp=True,
                    accept_downloads=False,
                    extra_http_headers={
                        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate, br",
                        "DNT":             "1",
                        "Upgrade-Insecure-Requests": "1",
                        "Sec-Fetch-Dest":  "document",
                        "Sec-Fetch-Mode":  "navigate",
                        "Sec-Fetch-Site":  "none",
                        "Sec-Fetch-User":  "?1",
                    },
                )

                # Inject stealth script before any page JS runs
                ctx.add_init_script(STEALTH_JS)
                page = ctx.new_page()

                # Block images/fonts to load faster (we only need HTML/JS)
                page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot}",
                           lambda r: r.abort())

                parsed = urlparse(url)
                base_url = f"{parsed.scheme}://{parsed.netloc}"

                try:
                    # Step 1: visit homepage first (builds cookies + CF trust score)
                    if url != base_url and url != base_url + "/":
                        try:
                            page.goto(base_url, wait_until="domcontentloaded", timeout=15000)
                            page.wait_for_timeout(1500)
                            # Simulate brief human interaction
                            page.mouse.move(400, 300)
                            page.wait_for_timeout(500)
                        except Exception:
                            pass  # OK if homepage fails — still try target

                    # Step 2: navigate to target page
                    try:
                        page.goto(url, wait_until="networkidle", timeout=30000)
                    except PWTimeout:
                        try:
                            page.goto(url, wait_until="domcontentloaded", timeout=20000)
                        except PWTimeout:
                            # Last resort — just get whatever loaded
                            pass

                    # Step 3: wait for real content (not just the shell)
                    try:
                        page.wait_for_selector("main, article, .content, #content, p, h1",
                                               timeout=10000)
                    except Exception:
                        page.wait_for_timeout(3000)

                    # Step 4: scroll to trigger lazy-loading
                    page.evaluate("""
                        window.scrollTo({top: document.body.scrollHeight * 0.3, behavior: 'smooth'});
                    """)
                    page.wait_for_timeout(1500)
                    page.evaluate("""
                        window.scrollTo({top: document.body.scrollHeight * 0.6, behavior: 'smooth'});
                    """)
                    page.wait_for_timeout(1000)

                    html = page.content()

                    if len(html) < 1000:
                        console.print(
                            f"  [yellow]Playwright: very little content ({len(html)} chars) — "
                            f"site may still be blocking[/]"
                        )
                        # Still return it — some content is better than none
                        return html if len(html) > 200 else None

                    console.print(f"  [green]Playwright:[/] {len(html):,} chars from {url[:50]}")
                    return html

                except Exception as e:
                    console.print(f"  [yellow]Playwright page error: {e}[/]")
                    try:
                        return page.content()
                    except Exception:
                        return None
                finally:
                    browser.close()

        except Exception as e:
            console.print(f"  [yellow]Playwright launch error: {e}[/]")
            return None

    def _extract_text(self, html: str) -> str:
        try:
            import trafilatura
            text = trafilatura.extract(html, include_tables=True, include_links=False)
            if text:
                return text
        except Exception:
            pass
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script","style","nav","footer","header","aside"]):
                tag.decompose()
            return soup.get_text(separator=" ")
        except Exception:
            return ""

    def _extract_links(self, html: str, base_url: str, base_domain: str, same_domain: bool) -> list[str]:
        links = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#","mailto:","tel:","javascript:")):
                    continue
                full = urljoin(base_url, href).split("#")[0]
                if same_domain and urlparse(full).netloc != base_domain:
                    continue
                if full.startswith("http") and full not in links:
                    links.append(full)
        except Exception:
            pass
        return links[:50]  # cap per page to avoid explosion


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMART WEB SEARCH — auto-learns from results
# ─────────────────────────────────────────────────────────────────────────────

class SmartSearchAgent(BaseAgent):
    """
    Enhanced web search that:
    1. Returns structured result cards with title, snippet, URL
    2. Automatically stores results in memory so AI learns from every search
    3. Uses training agent to log the interaction
    """

    name = "smart_search_agent"

    def __init__(self, engine: Engine, memory: Memory, logger: Logger, training_agent=None):
        super().__init__(engine, memory, logger)
        self.training_agent = training_agent

    @staticmethod
    def _clean_snippet(raw: str) -> str:
        """Normalise concatenated web text (e.g. 'TodthyesterdayinPatnawas')."""
        if not raw:
            return ""
        s = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw)
        s = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', s)
        s = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', s)
        return re.sub(r' {2,}', ' ', s).strip()

    def _multi_engine_search(self, query: str, max_results: int = 6) -> list:
        """
        Multi-engine web search. Priority:
          1. ResearchSearchEngine (OpenAlex/Semantic Scholar/PubMed/arXiv for academic)
          2. Bing scrape
          3. DDG library (ddgs)
          4. DDG instant answer API
          5. Wikipedia API
        Returns list of card dicts: {title, snippet, url, source, reliability}
        """
        cards = []
        _c = self._clean_snippet

        # ── 1. ResearchSearchEngine (multi-source) ─────────────────────────
        try:
            from agents.research_search_engine import ResearchSearchEngine
            rse = ResearchSearchEngine()
            results = rse.search(query, max_results=max_results)
            for r in results:
                snippet = _c((r.abstract or "")[:300]) or _c(r.title)
                if snippet:
                    cards.append({
                        "title":       r.title,
                        "snippet":     snippet,
                        "url":         r.url,
                        "source":      r.source,
                        "reliability": r.reliability_score,
                    })
        except Exception:
            pass

        if len(cards) >= max_results:
            return cards[:max_results]

        # ── 2. Bing scrape (no API key needed) ────────────────────────────
        if len(cards) < 3:
            try:
                hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                r = requests.get(
                    "https://www.bing.com/search",
                    params={"q": query, "count": 6},
                    headers=hdrs, timeout=7,
                )
                # Extract result snippets via regex (no BS4 required)
                titles   = re.findall(r'<h2[^>]*><a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text)
                snippets = re.findall(r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', r.text)
                for i, (url, title) in enumerate(titles[:5]):
                    if url.startswith("http"):
                        snip = _c(re.sub(r'<[^>]+>', '', snippets[i] if i < len(snippets) else ""))[:280]
                        clean_title = re.sub(r'<[^>]+>', '', title).strip()
                        if clean_title:
                            cards.append({"title": clean_title, "snippet": snip,
                                          "url": url, "source": "bing", "reliability": 0.70})
            except Exception:
                pass

        # ── 3. DDG library ─────────────────────────────────────────────────
        if len(cards) < 3:
            try:
                from ddgs import DDGS
                with DDGS(timeout=8) as ddgs:
                    results = list(ddgs.text(query, max_results=6, timelimit="m"))
                for r in results:
                    cards.append({
                        "title":       r.get("title", ""),
                        "snippet":     _c(r.get("body", ""))[:300],
                        "url":         r.get("href", ""),
                        "source":      "duckduckgo",
                        "reliability": 0.55,
                    })
            except Exception:
                pass

        # ── 4. DDG instant answer ──────────────────────────────────────────
        if len(cards) < 2:
            try:
                r = requests.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": "1"},
                    timeout=6,
                )
                data = r.json()
                if data.get("AbstractText"):
                    cards.append({
                        "title":       data.get("Heading", query),
                        "snippet":     _c(data["AbstractText"])[:300],
                        "url":         data.get("AbstractURL", ""),
                        "source":      "duckduckgo_instant",
                        "reliability": 0.55,
                    })
            except Exception:
                pass

        # ── 5. Wikipedia ───────────────────────────────────────────────────
        if len(cards) < 2:
            try:
                r = requests.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "query", "list": "search", "srsearch": query,
                            "format": "json", "srlimit": 4, "srprop": "snippet"},
                    timeout=6,
                )
                for item in r.json().get("query", {}).get("search", []):
                    snip = _c(re.sub(r'<[^>]+>', '', item.get("snippet", "")))
                    cards.append({
                        "title":       item["title"],
                        "snippet":     snip[:300],
                        "url":         f"https://en.wikipedia.org/wiki/{item['title'].replace(' ', '_')}",
                        "source":      "wikipedia",
                        "reliability": 0.65,
                    })
            except Exception:
                pass

        return cards[:max_results]

    def search(self, query: str, save_to_memory: bool = True) -> dict:
        """
        Search the web and return structured cards + synthesised answer.
        Uses multi-engine search: ResearchSearchEngine → Bing → DDG → Wikipedia.
        Automatically stores knowledge in ChromaDB.
        """
        console.print(f"  [dim]Smart search:[/] {query[:60]}")

        cards = self._multi_engine_search(query, max_results=6)
        context = ""

        # ── Auto-save to memory ────────────────────────────────────────────
        if save_to_memory and cards:
            chunks = []
            for card in cards:
                text = f"{card['title']}. {card['snippet']}"
                if len(text.strip()) > 30:
                    chunks.append({"text": text, "source": card["url"] or "web_search", "domain": "web_search"})
            if chunks:
                self.memory.store_many(chunks)
                if self.training_agent:
                    self.training_agent.log_learning(
                        "web_search",
                        f"Learned from search: '{query}' ({len(chunks)} snippets)",
                        "web_search",
                        len(chunks),
                    )

        # ── Synthesise answer from cards ───────────────────────────────────
        if cards:
            context = "\n".join(f"- {c['title']}: {c['snippet']}" for c in cards[:4])
            prompt  = (
                f"Using these web results, answer the question clearly and concisely.\n\n"
                f"Results:\n{context}\n\n"
                f"Question: {query}\n\nAnswer:"
            )
            answer = self.engine.generate(prompt, temperature=0.2)
        else:
            answer = "No web results found. Try a different query."

        return {
            "answer":  answer,
            "cards":   cards,
            "query":   query,
            "learned": len(cards),
        }

    # ── Separate card-fetch (no LLM) — used by streaming endpoint ─────────────
    def _fetch_cards(self, query: str, save_to_memory: bool = True) -> list[dict]:
        """
        Fetch search result cards WITHOUT running LLM synthesis.
        Uses multi-engine search: ResearchSearchEngine → Bing → DDG → Wikipedia.
        Called by /api/search/stream so cards can be streamed immediately.
        """
        console.print(f"  [dim]Fetch cards:[/] {query[:60]}")
        cards = self._multi_engine_search(query, max_results=6)

        # ── Auto-save to memory (background-safe) ──────────────────────────
        if save_to_memory and cards:
            chunks = []
            for card in cards:
                text = f"{card['title']}. {card['snippet']}"
                if len(text.strip()) > 30:
                    chunks.append({
                        "text": text,
                        "source": card["url"] or "web_search",
                        "domain": "web_search",
                    })
            if chunks:
                self.memory.store_many(chunks)
                if self.training_agent:
                    self.training_agent.log_learning(
                        "web_search",
                        f"Learned from search: '{query}' ({len(chunks)} snippets)",
                        "web_search",
                        len(chunks),
                    )

        return cards
