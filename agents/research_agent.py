"""
ARIA — Scientific Research Agent
===================================
Searches scientific databases, finds connections between papers,
and correlates research findings with your actual behavioural data.

Data sources (all free, no API key):
- arXiv          — preprints (AI, CS, physics, math, biology, psychology)
- PubMed          — biomedical and life science research
- Semantic Scholar — citation graph + AI-powered recommendations
- CORE            — open access research papers
- Europe PMC      — life sciences literature

Capabilities:
1. Search across all sources simultaneously
2. Find papers that cite each other (citation graph)
3. Identify emerging research trends
4. Connect dots between your behaviour data and research findings
5. Summarise technical papers in plain language
6. Find contradictions between papers
7. Build a knowledge graph of related concepts
"""

import re
import json
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
CACHE_DIR    = PROJECT_ROOT / "data" / "research_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ResearchAgent:
    """
    Searches multiple scientific databases simultaneously and synthesises findings.
    """

    HEADERS = {
        "User-Agent": "ARIA-Research/1.0 (educational; non-commercial)",
        "Accept": "application/json",
    }

    def __init__(self, engine=None, memory=None):
        self.engine = engine
        self.memory = memory

    # ── Multi-source search ───────────────────────────────────────────────────

    def search_all(
        self,
        query:       str,
        max_per_source: int = 5,
        year_from:   int = None,
    ) -> dict:
        """
        Search ALL sources in parallel and return unified results.
        """
        import concurrent.futures
        console.print(f"  [dim]Research: '{query}' across all sources...[/]")

        sources = {
            "arxiv":           lambda: self.search_arxiv(query, max_per_source),
            "semantic_scholar": lambda: self.search_semantic_scholar(query, max_per_source),
            "pubmed":          lambda: self.search_pubmed(query, max_per_source),
            "core":            lambda: self.search_core(query, max_per_source),
        }

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {name: ex.submit(fn) for name, fn in sources.items()}
            for name, future in futures.items():
                try:
                    results[name] = future.result(timeout=15)
                except Exception as e:
                    console.print(f"  [yellow]{name} failed: {e}[/]")
                    results[name] = []

        # Deduplicate by title similarity
        all_papers = []
        seen_titles = set()
        for source, papers in results.items():
            for p in papers:
                title_key = re.sub(r'\W+','',p.get('title','').lower())[:40]
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    p["source"] = source
                    all_papers.append(p)

        # Auto-store in ChromaDB
        if self.memory and all_papers:
            chunks = []
            for p in all_papers:
                text = f"{p.get('title','')}. {p.get('abstract','')}"
                if len(text.strip()) > 50:
                    chunks.append({
                        "text":   text[:1000],
                        "source": p.get("url",""),
                        "domain": "research",
                    })
            self.memory.store_many(chunks)

        return {
            "query":     query,
            "total":     len(all_papers),
            "papers":    all_papers,
            "by_source": {s: len(r) for s, r in results.items()},
        }

    # ── arXiv ────────────────────────────────────────────────────────────────

    def search_arxiv(self, query: str, max_results: int = 8) -> list[dict]:
        cache_key = f"arxiv_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        cached    = self._load_cache(cache_key, ttl_min=60)
        if cached:
            return cached

        try:
            import urllib.request, urllib.parse, xml.etree.ElementTree as ET
            url = (
                f"https://export.arxiv.org/api/query?"
                f"search_query=all:{urllib.parse.quote(query)}"
                f"&start=0&max_results={max_results}"
                f"&sortBy=lastUpdatedDate&sortOrder=descending"
            )
            import ssl as _ssl
            _ctx = _ssl.create_default_context()
            _ctx.check_hostname = False
            _ctx.verify_mode    = _ssl.CERT_NONE
            with urllib.request.urlopen(url, timeout=12, context=_ctx) as r:
                xml_data = r.read().decode("utf-8")

            ns      = {"atom": "http://www.w3.org/2005/Atom"}
            root    = ET.fromstring(xml_data)
            papers  = []

            for entry in root.findall("atom:entry", ns):
                title   = entry.findtext("atom:title","",ns).strip().replace("\n"," ")
                abstract= entry.findtext("atom:summary","",ns).strip().replace("\n"," ")
                url_str = entry.findtext("atom:id","",ns).strip()
                updated = entry.findtext("atom:updated","",ns)[:10]
                authors = [a.findtext("atom:name","",ns)
                           for a in entry.findall("atom:author",ns)][:3]
                cats    = [c.get("term","") for c in entry.findall("atom:category",ns)]

                papers.append({
                    "title":    title,
                    "abstract": abstract[:500],
                    "url":      url_str,
                    "authors":  authors,
                    "date":     updated,
                    "categories": cats[:3],
                    "source":   "arxiv",
                })

            self._save_cache(cache_key, papers)
            return papers

        except Exception as e:
            console.print(f"  [yellow]arXiv error: {e}[/]")
            return []

    # ── Semantic Scholar ──────────────────────────────────────────────────────

    def search_semantic_scholar(self, query: str, max_results: int = 8) -> list[dict]:
        cache_key = f"ss_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        cached    = self._load_cache(cache_key, ttl_min=60)
        if cached:
            return cached

        try:
            r = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query":  query,
                    "limit":  max_results,
                    "fields": "title,abstract,year,citationCount,openAccessPdf,authors,tldr",
                },
                headers=self.HEADERS,
                timeout=12,
            )
            papers = []
            for p in r.json().get("data", []):
                papers.append({
                    "title":      p.get("title",""),
                    "abstract":   (p.get("abstract") or "")[:500],
                    "tldr":       (p.get("tldr") or {}).get("text",""),
                    "year":       p.get("year"),
                    "citations":  p.get("citationCount",0),
                    "url":        (p.get("openAccessPdf") or {}).get("url",""),
                    "authors":    [a["name"] for a in p.get("authors",[])[:3]],
                    "source":     "semantic_scholar",
                })
            self._save_cache(cache_key, papers)
            return papers
        except Exception as e:
            console.print(f"  [yellow]Semantic Scholar error: {e}[/]")
            return []

    # ── PubMed ────────────────────────────────────────────────────────────────

    def search_pubmed(self, query: str, max_results: int = 6) -> list[dict]:
        cache_key = f"pm_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        cached    = self._load_cache(cache_key, ttl_min=60)
        if cached:
            return cached

        try:
            # Step 1: Search IDs
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed", "term": query,
                    "retmax": max_results, "retmode": "json",
                    "sort": "date",
                },
                timeout=10,
            )
            ids = r.json().get("esearchresult",{}).get("idlist",[])
            if not ids:
                return []

            # Step 2: Fetch details
            r2 = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(ids),
                    "retmode": "json",
                },
                timeout=10,
            )
            papers = []
            result = r2.json().get("result", {})
            for pmid in ids:
                item = result.get(pmid, {})
                if not item:
                    continue
                papers.append({
                    "title":   item.get("title",""),
                    "abstract": "",  # summary endpoint doesn't include abstract
                    "date":    item.get("pubdate",""),
                    "url":     f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "authors": [a["name"] for a in item.get("authors",[])[:3]],
                    "journal": item.get("source",""),
                    "source":  "pubmed",
                })
            self._save_cache(cache_key, papers)
            return papers
        except Exception as e:
            console.print(f"  [yellow]PubMed error: {e}[/]")
            return []

    # ── CORE (open access) ────────────────────────────────────────────────────

    def search_core(self, query: str, max_results: int = 6) -> list[dict]:
        """CORE — free open access research. No API key needed for basic search."""
        cache_key = f"core_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        cached    = self._load_cache(cache_key, ttl_min=120)
        if cached:
            return cached

        try:
            r = requests.get(
                "https://api.core.ac.uk/v3/search/works",
                params={
                    "q":     query,
                    "limit": max_results,
                },
                headers={"Accept": "application/json"},
                timeout=12,
            )
            papers = []
            for item in r.json().get("results", []):
                papers.append({
                    "title":    item.get("title",""),
                    "abstract": (item.get("abstract") or "")[:400],
                    "year":     item.get("yearPublished"),
                    "url":      item.get("downloadUrl") or item.get("sourceFulltextUrls",[""])[0],
                    "authors":  [a.get("name","") for a in item.get("authors",[])[:3]],
                    "source":   "core",
                })
            self._save_cache(cache_key, papers)
            return papers
        except Exception as e:
            console.print(f"  [yellow]CORE error: {e}[/]")
            return []

    # ── Connect dots ──────────────────────────────────────────────────────────

    def connect_dots(
        self,
        topic_a:    str,
        topic_b:    str,
        depth:      int = 2,
    ) -> dict:
        """
        Find hidden connections between two seemingly unrelated topics.
        e.g. "sleep deprivation" + "software bugs"
             → finds research connecting sleep, cognitive performance, error rates in coding

        Method:
        1. Search both topics
        2. Find papers that mention both
        3. Find intermediate concepts that bridge them
        4. Use LLM to synthesise the connection
        """
        console.print(f"  [dim]Connecting: '{topic_a}' ↔ '{topic_b}'[/]")

        results_a = self.search_all(f"{topic_a}", max_per_source=5)
        results_b = self.search_all(f"{topic_b}", max_per_source=5)
        results_ab= self.search_all(f"{topic_a} {topic_b}", max_per_source=5)

        # Find bridging papers (appear relevant to both)
        bridge_papers = results_ab.get("papers",[])

        # Extract key concepts from abstracts
        all_abstracts = " ".join(
            p.get("abstract","") or p.get("tldr","")
            for p in results_a["papers"] + results_b["papers"]
        )[:3000]

        # LLM synthesis
        connection = ""
        if self.engine and (results_a["papers"] or results_b["papers"]):
            context = "\n".join(
                f"- {p['title']}: {(p.get('tldr') or p.get('abstract',''))[:200]}"
                for p in bridge_papers[:5]
            )
            prompt = (
                f"Based on these research papers, explain the scientific connection "
                f"between '{topic_a}' and '{topic_b}'.\n\n"
                f"Papers:\n{context}\n\n"
                f"What does the research say about how these are connected? "
                f"Be specific and cite patterns across papers:"
            )
            try:
                connection = self.engine.generate(prompt, temperature=0.3)
            except Exception:
                pass

        return {
            "topic_a":        topic_a,
            "topic_b":        topic_b,
            "bridge_papers":  bridge_papers[:5],
            "papers_a":       results_a["papers"][:3],
            "papers_b":       results_b["papers"][:3],
            "connection":     connection,
            "confidence":     "high" if len(bridge_papers) >= 3 else
                              "medium" if len(bridge_papers) >= 1 else "low",
        }

    def correlate_with_behaviour(
        self,
        behaviour_data: dict,
        engine=None,
    ) -> dict:
        """
        Takes the behavioural profile and finds relevant research.
        e.g. "You have peak focus at 9am → search: circadian rhythm cognitive performance"
             "You average 12min sessions → search: context switching cognitive cost"
             "High stress score → search: stress productivity research"
        """
        eng     = engine or self.engine
        queries = []

        # Map behaviour patterns to research queries
        style   = behaviour_data.get("cognitive_style","")
        hours   = behaviour_data.get("peak_hours",[])
        stress  = behaviour_data.get("stress_level","")
        focus   = behaviour_data.get("avg_focus_min",0)

        if style == "deep_worker":
            queries.append(("deep work cognitive performance", "Deep work research"))
        elif style == "high_switcher":
            queries.append(("context switching attention cost", "Context switching cost"))

        if hours:
            avg_h = sum(hours)/len(hours)
            if avg_h < 10:
                queries.append(("morning cognitive performance circadian", "Circadian morning peak"))
            elif avg_h >= 20:
                queries.append(("night owl chronotype creativity", "Night owl research"))

        if stress == "high":
            queries.append(("work stress burnout prevention", "Stress & burnout research"))

        if focus < 15:
            queries.append(("attention span digital devices", "Attention span research"))
        elif focus > 45:
            queries.append(("flow state deep focus neuroscience", "Flow state research"))

        # Add default
        queries.append(("productivity pattern behaviour research", "General productivity"))

        results = []
        for query, label in queries[:3]:
            papers = self.search_semantic_scholar(query, max_results=3)
            if papers:
                results.append({
                    "label":   label,
                    "query":   query,
                    "papers":  papers[:2],
                })

        # LLM synthesis of your data + research
        synthesis = ""
        if eng and results:
            research_text = "\n".join(
                f"{r['label']}: {r['papers'][0]['title']} — "
                f"{(r['papers'][0].get('tldr') or r['papers'][0].get('abstract',''))[:200]}"
                for r in results if r["papers"]
            )
            prompt = (
                f"A person's computer usage shows:\n"
                f"- Cognitive style: {style}\n"
                f"- Peak hours: {hours}\n"
                f"- Average focus session: {focus:.0f} minutes\n"
                f"- Stress level: {stress}\n\n"
                f"Relevant research:\n{research_text}\n\n"
                f"Write 3-4 specific, evidence-based insights connecting their "
                f"behaviour patterns to the research findings. Be actionable:"
            )
            try:
                synthesis = eng.generate(prompt, temperature=0.3)
            except Exception:
                pass

        return {
            "behaviour_summary": behaviour_data,
            "research_links":   results,
            "synthesis":        synthesis,
            "generated_at":     datetime.now().isoformat(),
        }


    # ── Citation graph ────────────────────────────────────────────────────────

    def get_citations(
        self,
        paper_id:  str,
        direction: str = "both",   # "references" | "citations" | "both"
        depth:     int = 1,
        max_each:  int = 10,
    ) -> dict:
        """
        Traverse the citation graph from a seed paper.
        direction="references" → what this paper cites
        direction="citations"  → what cites this paper
        direction="both"       → full neighbourhood

        paper_id: Semantic Scholar paper ID (from search results)
        depth:    how many hops to traverse (keep ≤2 or it explodes)
        """
        console.print(f"  [dim]Citation graph: {paper_id} depth={depth}[/]")
        graph: dict = {"nodes": {}, "edges": []}
        self._traverse(paper_id, direction, depth, 0, graph, max_each)

        # Store all discovered papers in memory
        if self.memory and graph["nodes"]:
            chunks = []
            for pid, p in graph["nodes"].items():
                text = f"{p.get('title','')}.\n{p.get('abstract','')}"
                if len(text.strip()) > 50:
                    chunks.append({
                        "text":   text[:800],
                        "source": p.get("url",""),
                        "domain": "research_citation",
                    })
            if chunks:
                self.memory.store_many(chunks)

        return {
            "seed":     paper_id,
            "nodes":    len(graph["nodes"]),
            "edges":    len(graph["edges"]),
            "papers":   list(graph["nodes"].values()),
            "graph":    graph["edges"],
        }

    def _traverse(
        self, paper_id: str, direction: str, max_depth: int,
        current_depth: int, graph: dict, max_each: int
    ):
        """Recursively traverse the citation graph."""
        if current_depth >= max_depth or paper_id in graph["nodes"]:
            return

        # Fetch paper details
        paper = self._fetch_paper(paper_id)
        if not paper:
            return
        graph["nodes"][paper_id] = paper

        if current_depth >= max_depth:
            return

        # Fetch references (what this paper cites)
        if direction in ("references", "both"):
            refs = self._fetch_references(paper_id, max_each)
            for ref in refs:
                ref_id = ref.get("paperId","")
                if ref_id:
                    graph["edges"].append({"from": paper_id, "to": ref_id, "type": "cites"})
                    self._traverse(ref_id, direction, max_depth,
                                   current_depth + 1, graph, max_each)

        # Fetch citations (what cites this paper)
        if direction in ("citations", "both"):
            cites = self._fetch_citing_papers(paper_id, max_each)
            for cite in cites:
                cite_id = cite.get("paperId","")
                if cite_id:
                    graph["edges"].append({"from": cite_id, "to": paper_id, "type": "cites"})
                    if cite_id not in graph["nodes"]:
                        graph["nodes"][cite_id] = cite

    def _fetch_paper(self, paper_id: str) -> dict | None:
        cache_key = f"paper_{paper_id[:20]}"
        cached    = self._load_cache(cache_key, ttl_min=1440)
        if cached:
            return cached[0] if cached else None
        try:
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}",
                params={"fields": "title,abstract,year,citationCount,authors,openAccessPdf"},
                headers=self.HEADERS, timeout=8,
            )
            if r.ok:
                p = r.json()
                paper = {
                    "paperId":   paper_id,
                    "title":     p.get("title",""),
                    "abstract":  (p.get("abstract") or "")[:400],
                    "year":      p.get("year"),
                    "citations": p.get("citationCount",0),
                    "url":       (p.get("openAccessPdf") or {}).get("url",""),
                    "authors":   [a["name"] for a in p.get("authors",[])[:3]],
                }
                self._save_cache(cache_key, [paper])
                return paper
        except Exception:
            pass
        return None

    def _fetch_references(self, paper_id: str, limit: int = 10) -> list[dict]:
        cache_key = f"refs_{paper_id[:20]}"
        cached    = self._load_cache(cache_key, ttl_min=1440)
        if cached is not None:
            return cached
        try:
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references",
                params={"fields": "title,year,citationCount,paperId", "limit": limit},
                headers=self.HEADERS, timeout=8,
            )
            if r.ok:
                refs = [
                    {**d.get("citedPaper",{})}
                    for d in r.json().get("data",[])
                    if d.get("citedPaper",{}).get("paperId")
                ]
                self._save_cache(cache_key, refs)
                return refs
        except Exception:
            pass
        return []

    def _fetch_citing_papers(self, paper_id: str, limit: int = 10) -> list[dict]:
        cache_key = f"cites_{paper_id[:20]}"
        cached    = self._load_cache(cache_key, ttl_min=1440)
        if cached is not None:
            return cached
        try:
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
                params={"fields": "title,year,citationCount,paperId", "limit": limit},
                headers=self.HEADERS, timeout=8,
            )
            if r.ok:
                cites = [
                    {**d.get("citingPaper",{})}
                    for d in r.json().get("data",[])
                    if d.get("citingPaper",{}).get("paperId")
                ]
                self._save_cache(cache_key, cites)
                return cites
        except Exception:
            pass
        return []

    def fetch_fulltext(self, paper: dict) -> str | None:
        """
        Download full paper text from open-access URL.
        Works for arXiv, CORE, Europe PMC — any open access PDF.
        Returns extracted text or None.
        """
        url = paper.get("url","")
        if not url:
            # Try arXiv PDF from paper URL
            paper_url = paper.get("url","")
            if "arxiv.org/abs/" in paper_url:
                url = paper_url.replace("/abs/","/pdf/") + ".pdf"

        if not url:
            return None

        try:
            r = requests.get(url, timeout=15, headers={"User-Agent":"ARIA/1.0"})
            if not r.ok:
                return None

            content_type = r.headers.get("content-type","")

            if "pdf" in content_type.lower():
                # Extract text from PDF
                try:
                    import io
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                        text = "\n".join(
                            page.extract_text() or ""
                            for page in pdf.pages[:10]  # first 10 pages
                        )
                        return text[:8000] if text.strip() else None
                except ImportError:
                    console.print("  [dim]pip install pdfplumber for PDF extraction[/]")
                    return None

            elif "html" in content_type.lower():
                # Extract text from HTML
                from html.parser import HTMLParser
                class _TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text = []
                        self._skip = False
                    def handle_starttag(self, tag, attrs):
                        if tag in ("script","style","nav","header","footer"):
                            self._skip = True
                    def handle_endtag(self, tag):
                        if tag in ("script","style","nav","header","footer"):
                            self._skip = False
                    def handle_data(self, data):
                        if not self._skip:
                            self.text.append(data)
                p = _TextExtractor()
                p.feed(r.text)
                text = " ".join(p.text)
                import re
                text = re.sub(r"\s+", " ", text).strip()
                return text[:8000] if text else None

        except Exception as e:
            console.print(f"  [dim]Fulltext fetch error: {e}[/]")
            return None

    def get_paper_id_from_url(self, url: str) -> str | None:
        """Extract Semantic Scholar paper ID from a URL or DOI."""
        import re
        # arXiv ID
        m = re.search(r"arxiv\.org/abs/([\d.]+)", url)
        if m:
            arxiv_id = m.group(1)
            try:
                r = requests.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}",
                    params={"fields":"paperId"}, headers=self.HEADERS, timeout=5,
                )
                if r.ok:
                    return r.json().get("paperId")
            except Exception:
                pass
        # Direct Semantic Scholar URL
        m2 = re.search(r"semanticscholar\.org/paper/[^/]+/([a-f0-9]{40})", url)
        if m2:
            return m2.group(1)
        return None

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        return CACHE_DIR / f"{key}.json"

    def _load_cache(self, key: str, ttl_min: int = 60) -> Optional[list]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        if (time.time() - p.stat().st_mtime) > ttl_min * 60:
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, key: str, data: list):
        try:
            self._cache_path(key).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass
