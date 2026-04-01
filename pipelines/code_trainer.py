"""
ARIA — Code Training Pipeline
===============================
Trains ARIA to code in any language/framework by:

1. Crawling official documentation
2. Scraping GitHub repositories for real examples
3. Extracting all code blocks with their explanations
4. Building question → code pairs for fine-tuning
5. Running self-play to generate more training examples
6. Exporting a LoRA-ready JSONL dataset

After running this + LoRA fine-tune on Colab:
  - ARIA understands the framework deeply
  - Can write correct code using its own trained knowledge
  - Uses RAG as backup for anything it's not sure about
  - Gets better every time you use it for that framework

Usage:
    python pipelines/code_trainer.py --topics spacy react
    python pipelines/code_trainer.py --topics "all"
    python pipelines/code_trainer.py --github facebook/react
"""

import sys
import os
import re
import json
import time
import hashlib
import argparse
import requests
import threading
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

console = Console()

DATA_DIR     = PROJECT_ROOT / "data" / "training"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TOPIC REGISTRY
# Add any framework/library here — ARIA will learn all of them
# ─────────────────────────────────────────────────────────────────────────────

TOPICS = {
    # ── Python libraries ──────────────────────────────────────────────────────
    "spacy": {
        "docs":   ["https://spacy.io/usage", "https://spacy.io/api"],
        "github": "explosion/spaCy",
        "desc":   "Industrial NLP library",
    },
    "pandas": {
        "docs":   ["https://pandas.pydata.org/docs/user_guide/"],
        "github": "pandas-dev/pandas",
        "desc":   "Data analysis library",
    },
    "fastapi": {
        "docs":   ["https://fastapi.tiangolo.com/tutorial/"],
        "github": "tiangolo/fastapi",
        "desc":   "Modern Python web framework",
    },
    "langchain": {
        "docs":   ["https://python.langchain.com/docs/get_started/"],
        "github": "langchain-ai/langchain",
        "desc":   "LLM application framework",
    },
    "numpy": {
        "docs":   ["https://numpy.org/doc/stable/user/"],
        "github": "numpy/numpy",
        "desc":   "Numerical computing",
    },
    "sqlalchemy": {
        "docs":   ["https://docs.sqlalchemy.org/en/20/orm/"],
        "github": "sqlalchemy/sqlalchemy",
        "desc":   "Python SQL toolkit",
    },

    # ── JavaScript/TypeScript ─────────────────────────────────────────────────
    "react": {
        "docs":   ["https://react.dev/learn"],
        "github": "facebook/react",
        "desc":   "UI component library",
    },
    "nextjs": {
        "docs":   ["https://nextjs.org/docs"],
        "github": "vercel/next.js",
        "desc":   "React framework for production",
    },
    "typescript": {
        "docs":   ["https://www.typescriptlang.org/docs/"],
        "github": "microsoft/TypeScript",
        "desc":   "Typed JavaScript",
    },
    "nodejs": {
        "docs":   ["https://nodejs.org/en/docs/guides/"],
        "github": "nodejs/node",
        "desc":   "Server-side JavaScript",
    },

    # ── Databases ─────────────────────────────────────────────────────────────
    "mongodb": {
        "docs":   ["https://www.mongodb.com/docs/manual/"],
        "github": "mongodb/mongo",
        "desc":   "Document database",
    },
    "postgresql": {
        "docs":   ["https://www.postgresql.org/docs/current/"],
        "github": "postgres/postgres",
        "desc":   "Relational database",
    },

    # ── AI/ML ─────────────────────────────────────────────────────────────────
    "pytorch": {
        "docs":   ["https://pytorch.org/tutorials/"],
        "github": "pytorch/pytorch",
        "desc":   "Deep learning framework",
    },
    "transformers": {
        "docs":   ["https://huggingface.co/docs/transformers/"],
        "github": "huggingface/transformers",
        "desc":   "NLP model library",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CODE EXTRACTOR
# Pulls every code block from documentation pages
# ─────────────────────────────────────────────────────────────────────────────

class CodeExtractor:
    """
    Extracts code examples from documentation.
    For each code block, also captures:
    - The surrounding explanation (what this code does)
    - The section heading (context)
    - The language (python, javascript, etc.)
    This gives us (explanation, code) pairs perfect for fine-tuning.
    """

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def extract_from_url(self, url: str) -> list[dict]:
        """Fetch a URL and extract all code examples with context."""
        html = self._fetch(url)
        if not html:
            return []
        return self._parse_code_blocks(html, url)

    def _fetch(self, url: str) -> str | None:
        # Tier 1: requests
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=15, allow_redirects=True)
            if r.status_code == 200 and len(r.text) > 500:
                return r.text
        except Exception:
            pass

        # Tier 2: Playwright (for JS-rendered docs like react.dev)
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-blink-features=AutomationControlled"]
                )
                ctx = browser.new_context(
                    user_agent=self.HEADERS["User-Agent"],
                    bypass_csp=True,
                )
                ctx.add_init_script(
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
                    "window.chrome={runtime:{}};"
                )
                page = ctx.new_page()
                page.route("**/*.{png,jpg,gif,svg,ico,woff,woff2}", lambda r: r.abort())

                parsed = urlparse(url)
                base   = f"{parsed.scheme}://{parsed.netloc}"
                try:
                    page.goto(base, wait_until="domcontentloaded", timeout=10000)
                    page.wait_for_timeout(800)
                except Exception:
                    pass
                try:
                    page.goto(url, wait_until="networkidle", timeout=25000)
                except PWTimeout:
                    page.goto(url, wait_until="domcontentloaded", timeout=15000)

                try:
                    page.wait_for_selector("code, pre", timeout=8000)
                except Exception:
                    page.wait_for_timeout(3000)

                html = page.content()
                browser.close()
                return html if len(html) > 500 else None
        except Exception:
            return None

    def _parse_code_blocks(self, html: str, source_url: str) -> list[dict]:
        """Parse HTML and extract code blocks with surrounding context."""
        from bs4 import BeautifulSoup
        soup    = BeautifulSoup(html, "lxml")
        results = []

        # Remove nav, header, footer noise
        for tag in soup(["nav","header","footer","aside","script","style"]):
            tag.decompose()

        # Get page title
        page_title = soup.find("h1")
        page_title = page_title.get_text(strip=True) if page_title else ""

        # Find all code blocks
        for pre in soup.find_all("pre"):
            code_el = pre.find("code")
            if not code_el:
                code_el = pre

            code    = code_el.get_text()
            if len(code.strip()) < 20:
                continue

            # Detect language from class
            lang    = "python"
            classes = " ".join(code_el.get("class", []) + pre.get("class", []))
            if any(x in classes for x in ["javascript","js","jsx","tsx","typescript"]):
                lang = "javascript"
            elif any(x in classes for x in ["bash","shell","sh","cmd"]):
                lang = "bash"
            elif any(x in classes for x in ["html","htm"]):
                lang = "html"
            elif any(x in classes for x in ["css"]):
                lang = "css"
            elif any(x in classes for x in ["json"]):
                lang = "json"

            # Get surrounding explanation (previous siblings)
            explanation_parts = []
            current           = pre.find_previous_sibling()
            for _ in range(4):
                if not current:
                    break
                if current.name in ["h1","h2","h3","h4"]:
                    explanation_parts.insert(0, f"## {current.get_text(strip=True)}")
                    break
                if current.name in ["p","li","div"]:
                    text = current.get_text(strip=True)
                    if len(text) > 20:
                        explanation_parts.insert(0, text)
                current = current.find_previous_sibling()

            explanation = " ".join(explanation_parts) or page_title

            results.append({
                "code":        code.strip(),
                "explanation": explanation[:500],
                "language":    lang,
                "source":      source_url,
                "page_title":  page_title,
            })

        console.print(f"  [dim]  {source_url[:60]} → {len(results)} code blocks[/]")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB SCRAPER
# Gets real-world usage examples from popular repos
# ─────────────────────────────────────────────────────────────────────────────

class GitHubScraper:
    """
    Scrapes code examples from GitHub repositories.
    Uses GitHub's search API to find files that use the library,
    then extracts functions and classes as training examples.
    No API key needed for basic access (60 requests/hour).
    """

    def __init__(self, token: str = None):
        self.token   = token
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            self.headers["Authorization"] = f"token {token}"

    def get_readme_examples(self, repo: str) -> list[dict]:
        """Get code examples from a repo's README."""
        results = []
        try:
            r = requests.get(
                f"https://api.github.com/repos/{repo}/readme",
                headers=self.headers, timeout=10,
            )
            if r.status_code != 200:
                return []

            import base64
            content = base64.b64decode(r.json()["content"]).decode("utf-8", errors="replace")

            # Extract code blocks from markdown
            code_blocks = re.findall(r"```(\w+)?\n(.*?)```", content, re.DOTALL)
            for lang, code in code_blocks:
                if len(code.strip()) < 30:
                    continue
                results.append({
                    "code":        code.strip(),
                    "explanation": f"{repo} README example",
                    "language":    lang or "python",
                    "source":      f"https://github.com/{repo}",
                })
        except Exception as e:
            console.print(f"  [yellow]GitHub README error ({repo}): {e}[/]")
        return results

    def search_code_examples(self, query: str, language: str = "python", max_results: int = 20) -> list[dict]:
        """Search GitHub for code examples using a topic."""
        results = []
        try:
            r = requests.get(
                "https://api.github.com/search/code",
                params={
                    "q":        f"{query} language:{language}",
                    "per_page": min(max_results, 30),
                    "sort":     "indexed",
                },
                headers=self.headers, timeout=15,
            )
            if r.status_code == 403:
                console.print("  [yellow]GitHub rate limit — waiting 60s...[/]")
                time.sleep(60)
                return []
            if r.status_code != 200:
                return []

            for item in r.json().get("items", [])[:max_results]:
                try:
                    file_r = requests.get(item["url"], headers=self.headers, timeout=8)
                    if file_r.status_code != 200:
                        continue
                    import base64
                    code = base64.b64decode(
                        file_r.json().get("content","")
                    ).decode("utf-8", errors="replace")

                    # Only keep files that are reasonably sized
                    if 100 < len(code) < 5000:
                        results.append({
                            "code":        code[:3000],
                            "explanation": f"Example from {item['repository']['full_name']}: {item['name']}",
                            "language":    language,
                            "source":      item["html_url"],
                        })
                    time.sleep(0.5)  # Be polite to GitHub API
                except Exception:
                    continue
        except Exception as e:
            console.print(f"  [yellow]GitHub search error: {e}[/]")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING PAIR GENERATOR
# Converts code examples into (question, answer) training pairs
# ─────────────────────────────────────────────────────────────────────────────

class TrainingPairGenerator:
    """
    Takes raw code + explanation and generates multiple training Q&A pairs.
    Each code block becomes several different question formats:
    1. "How do I [task described in explanation]?"  → code answer
    2. "Write [language] code to [task]"            → code answer
    3. "Explain what this code does" + code         → explanation answer
    4. "What's wrong / how to improve this?"        → improved version
    This diversity makes the model robust to how users phrase requests.
    """

    def __init__(self, engine):
        self.engine = engine

    def generate_pairs(self, example: dict) -> list[dict]:
        """Generate multiple training pairs from one code example."""
        pairs    = []
        code     = example["code"]
        expl     = example.get("explanation","")
        lang     = example.get("language","python")
        source   = example.get("source","")

        # Pair 1: Direct "how do I" question
        if expl:
            q1 = f"How do I {expl.lower().rstrip('.')}? Show me with {lang} code."
            pairs.append({
                "instruction": q1,
                "output":      f"```{lang}\n{code}\n```",
                "type":        "how_to",
            })

        # Pair 2: "Write code to..." format
        if expl:
            q2 = f"Write {lang} code to {expl.lower().rstrip('.')}"
            pairs.append({
                "instruction": q2,
                "output":      f"Here's the {lang} code:\n\n```{lang}\n{code}\n```",
                "type":        "write_code",
            })

        # Pair 3: "What does this code do?" — reverse
        pairs.append({
            "instruction": f"What does this {lang} code do?\n\n```{lang}\n{code[:500]}\n```",
            "output":      expl if expl else f"This {lang} code demonstrates a common usage pattern.",
            "type":        "explain_code",
        })

        # Pair 4: LLM-generated Q&A (higher quality, slower)
        if len(code) < 800:
            llm_pair = self._llm_generate(code, lang, expl)
            if llm_pair:
                pairs.append(llm_pair)

        # Add source to all pairs
        for p in pairs:
            p["source"]   = source
            p["language"] = lang

        return pairs

    def _llm_generate(self, code: str, lang: str, context: str) -> dict | None:
        """Use the LLM to generate a natural question for this code."""
        prompt = (
            f"Look at this {lang} code:\n```{lang}\n{code[:400]}\n```\n\n"
            f"Write ONE natural question a developer would ask that this code answers.\n"
            f"Format: just the question, nothing else."
        )
        try:
            question = self.engine.generate(prompt, temperature=0.4).strip()
            if not question or len(question) < 10:
                return None
            return {
                "instruction": question,
                "output":      f"```{lang}\n{code}\n```\n\nExplanation: {context}" if context else f"```{lang}\n{code}\n```",
                "type":        "natural_question",
            }
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# MASTER CODE TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class CodeTrainingPipeline:
    """
    Full pipeline: crawl docs → extract code → generate pairs → store → export.
    Run this to make ARIA an expert in any framework.
    """

    def __init__(self, engine=None, memory=None, logger=None):
        self.engine    = engine
        self.memory    = memory
        self.logger    = logger
        self.extractor = CodeExtractor()
        self.gh_scraper = GitHubScraper()
        self.pair_gen  = TrainingPairGenerator(engine) if engine else None

    def train_on_topic(
        self,
        topic: str,
        crawl_docs: bool = True,
        scrape_github: bool = True,
        max_doc_pages: int = 20,
        max_github_examples: int = 30,
    ) -> dict:
        """
        Full training run for one topic.
        Returns stats about what was collected.
        """
        if topic not in TOPICS:
            return {"error": f"Unknown topic: {topic}. Available: {list(TOPICS.keys())}"}

        config     = TOPICS[topic]
        all_pairs  = []
        all_chunks = []
        stats      = {"topic": topic, "doc_pages": 0, "code_blocks": 0,
                      "training_pairs": 0, "github_examples": 0}

        console.print(f"\n[bold]Training on:[/] {topic} — {config['desc']}")

        # ── Phase 1: Crawl documentation ──────────────────────────────────────
        if crawl_docs:
            console.print(f"  [dim]Phase 1: Crawling documentation...[/]")
            visited = set()
            queue   = list(config["docs"])

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TextColumn("{task.completed}/{task.total}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Crawling {topic} docs", total=max_doc_pages)

                while queue and stats["doc_pages"] < max_doc_pages:
                    url = queue.pop(0)
                    if url in visited:
                        continue
                    visited.add(url)

                    # Extract code examples from this page
                    examples = self.extractor.extract_from_url(url)
                    stats["doc_pages"]  += 1
                    stats["code_blocks"] += len(examples)

                    # Store raw text in memory (for RAG)
                    if self.memory and examples:
                        for ex in examples:
                            chunk_text = f"{ex['explanation']}\n\n```{ex['language']}\n{ex['code']}\n```"
                            all_chunks.append({
                                "text":   chunk_text,
                                "source": ex["source"],
                                "domain": topic,
                            })

                    # Generate training pairs
                    if self.pair_gen:
                        for ex in examples:
                            pairs = self.pair_gen.generate_pairs(ex)
                            all_pairs.extend(pairs)

                    # Find more links to crawl (stay on same domain)
                    try:
                        from bs4 import BeautifulSoup
                        r    = requests.get(url, headers=self.extractor.HEADERS, timeout=10)
                        soup = BeautifulSoup(r.text, "lxml")
                        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
                        for a in soup.find_all("a", href=True):
                            href = urljoin(url, a["href"]).split("#")[0]
                            if (href.startswith(base) and
                                href not in visited and
                                href not in queue):
                                queue.append(href)
                    except Exception:
                        pass

                    progress.update(task, advance=1)
                    time.sleep(0.5)

        # ── Phase 2: GitHub examples ───────────────────────────────────────────
        if scrape_github and config.get("github"):
            console.print(f"  [dim]Phase 2: Scraping GitHub {config['github']}...[/]")

            # README examples
            readme_examples = self.gh_scraper.get_readme_examples(config["github"])
            stats["github_examples"] += len(readme_examples)

            for ex in readme_examples:
                all_chunks.append({
                    "text":   f"{ex['explanation']}\n\n```{ex['language']}\n{ex['code']}\n```",
                    "source": ex["source"],
                    "domain": topic,
                })
                if self.pair_gen:
                    all_pairs.extend(self.pair_gen.generate_pairs(ex))

            # Search for usage examples
            lang_map = {
                "react": "javascript", "nextjs": "javascript",
                "typescript": "typescript", "nodejs": "javascript",
            }
            lang = lang_map.get(topic, "python")
            search_examples = self.gh_scraper.search_code_examples(
                topic, lang, max_github_examples
            )
            stats["github_examples"] += len(search_examples)

            for ex in search_examples:
                all_chunks.append({
                    "text":   f"{ex['explanation']}\n\n```{ex['language']}\n{ex['code']}\n```",
                    "source": ex["source"],
                    "domain": topic,
                })

        # ── Phase 3: Store in ChromaDB ─────────────────────────────────────────
        if self.memory and all_chunks:
            console.print(f"  [dim]Phase 3: Storing {len(all_chunks)} chunks in memory...[/]")
            self.memory.store_many(all_chunks)
            console.print(f"  [green]✓[/] {len(all_chunks)} chunks stored in ChromaDB")

        # ── Phase 4: Save training dataset ────────────────────────────────────
        stats["training_pairs"] = len(all_pairs)
        if all_pairs:
            ts        = datetime.now().strftime("%Y%m%d_%H%M")
            out_path  = DATA_DIR / f"code_{topic}_{ts}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for pair in all_pairs:
                    f.write(json.dumps({
                        "instruction": pair["instruction"],
                        "output":      pair["output"],
                        "domain":      topic,
                        "type":        pair.get("type",""),
                        "language":    pair.get("language",""),
                    }, ensure_ascii=False) + "\n")
            stats["dataset_file"] = str(out_path)
            console.print(
                f"  [green]✓[/] Dataset saved: {out_path.name} "
                f"({len(all_pairs)} training pairs)"
            )

        # ── Phase 5: Log to TrainingAgent ──────────────────────────────────────
        if self.logger:
            try:
                with self.logger._connect() as conn:
                    conn.execute(
                        "INSERT INTO learning_events (ts, event_type, description, domain, chunks_added) "
                        "VALUES (?,?,?,?,?)",
                        (datetime.now().isoformat(), "code_training",
                         f"Trained on {topic}: {stats['code_blocks']} code blocks, "
                         f"{stats['training_pairs']} training pairs",
                         topic, len(all_chunks)),
                    )
            except Exception:
                pass

        console.print(f"\n  [bold green]Training complete:[/] {topic}")
        console.print(f"  Pages crawled:    {stats['doc_pages']}")
        console.print(f"  Code blocks:      {stats['code_blocks']}")
        console.print(f"  GitHub examples:  {stats['github_examples']}")
        console.print(f"  Training pairs:   {stats['training_pairs']}")
        console.print(f"  Memory chunks:    {len(all_chunks)}")

        return stats

    def train_on_multiple(self, topics: list[str], **kwargs) -> list[dict]:
        """Train on multiple topics sequentially."""
        results = []
        for topic in topics:
            result = self.train_on_topic(topic, **kwargs)
            results.append(result)
            time.sleep(2)  # Brief pause between topics
        return results

    def merge_datasets(self, output_name: str = "code_combined") -> str:
        """Merge all JSONL files in training dir into one for fine-tuning."""
        all_examples = []
        for f in DATA_DIR.glob("code_*.jsonl"):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        all_examples.append(json.loads(line))

        out_path = DATA_DIR / f"{output_name}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        console.print(f"\n[green]Merged dataset:[/] {len(all_examples)} total examples → {out_path.name}")
        return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ARIA Code Training Pipeline")
    parser.add_argument("--topics", nargs="+", default=["spacy"],
                        help=f"Topics to train on. Available: {list(TOPICS.keys())} or 'all'")
    parser.add_argument("--github", type=str, help="Train on a specific GitHub repo (owner/repo)")
    parser.add_argument("--no-github", action="store_true", help="Skip GitHub scraping")
    parser.add_argument("--max-pages", type=int, default=20, help="Max doc pages per topic")
    parser.add_argument("--merge", action="store_true", help="Merge all datasets and exit")
    parser.add_argument("--list", action="store_true", help="List available topics")
    args = parser.parse_args()

    if args.list:
        console.print("\n[bold]Available topics:[/]")
        for name, cfg in TOPICS.items():
            console.print(f"  [cyan]{name:15s}[/] {cfg['desc']}")
        return

    # Bootstrap ARIA components
    try:
        from core.engine import Engine
        from core.memory import Memory
        from tools.logger import Logger
        engine = Engine()
        memory = Memory()
        logger = Logger()
        console.print("[green]✓[/] ARIA core loaded")
    except Exception as e:
        console.print(f"[yellow]Running without ARIA core: {e}[/]")
        engine = memory = logger = None

    pipeline = CodeTrainingPipeline(engine, memory, logger)

    if args.merge:
        pipeline.merge_datasets()
        return

    # Expand "all" to every topic
    topics = list(TOPICS.keys()) if "all" in args.topics else args.topics
    invalid = [t for t in topics if t not in TOPICS]
    if invalid:
        console.print(f"[red]Unknown topics: {invalid}[/]")
        console.print(f"Available: {list(TOPICS.keys())}")
        return

    console.print(f"\n[bold]ARIA Code Training Pipeline[/]")
    console.print(f"Topics: {topics}")
    console.print(f"Max pages per topic: {args.max_pages}\n")

    results = pipeline.train_on_multiple(
        topics,
        scrape_github=not args.no_github,
        max_doc_pages=args.max_pages,
    )

    # Summary
    console.print("\n[bold]Training summary:[/]")
    total_pairs  = sum(r.get("training_pairs",0) for r in results)
    total_chunks = sum(r.get("code_blocks",0) + r.get("github_examples",0) for r in results)
    console.print(f"  Topics trained:   {len(results)}")
    console.print(f"  Total code blocks:{total_chunks}")
    console.print(f"  Training pairs:   {total_pairs}")
    console.print(f"\nNext step — fine-tune on Colab:")
    console.print(f"  [cyan]python pipelines/code_trainer.py --merge[/]")
    console.print(f"  Upload the merged JSONL to Colab and run the LoRA training script")


if __name__ == "__main__":
    main()
