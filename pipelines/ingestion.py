"""
ARIA — Document ingestion pipeline (the Fuel Injector)
Reads any document → chunks it → embeds it → stores in memory.

Supported formats:
  - PDF          (pymupdf)
  - Web URL      (trafilatura + beautifulsoup4)
  - Word (.docx) (python-docx)
  - Plain text   (.txt, .md)

All free. All run offline (except web URLs which need internet).
"""

import re
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import track

from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.memory import Memory
from tools.logger import Logger

console = Console()


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-chunks.
    Overlap ensures context at boundaries isn't lost.
    """
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def clean_text(text: str) -> str:
    """Normalize whitespace, remove garbage characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\u0900-\u097F\u0600-\u06FF]", " ", text)  # keep Latin + Devanagari + Arabic
    return text.strip()


# ── Format readers ────────────────────────────────────────────────────────────

def read_pdf(path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz). Free, fast, no OCR needed for digital PDFs."""
    try:
        import fitz  # pymupdf
        doc  = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return clean_text(text)
    except ImportError:
        console.print("[red]Install pymupdf:[/] pip install pymupdf")
        return ""


def read_docx(path: str) -> str:
    """Extract text from Word document."""
    try:
        from docx import Document
        doc  = Document(path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return clean_text(text)
    except ImportError:
        console.print("[red]Install python-docx:[/] pip install python-docx")
        return ""


def read_txt(path: str) -> str:
    """Read plain text or markdown file."""
    for enc in ["utf-8", "utf-16", "latin-1"]:
        try:
            return clean_text(Path(path).read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    return ""


def read_url(url: str) -> str:
    """Extract clean article text from a URL. Uses trafilatura (best free web reader)."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_tables=True)
            if text:
                return clean_text(text)

        # Fallback: raw BeautifulSoup
        import requests
        from bs4 import BeautifulSoup
        r    = requests.get(url, timeout=15, headers={"User-Agent": "ARIA/1.0"})
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return clean_text(soup.get_text(separator=" "))

    except Exception as e:
        console.print(f"[red]URL read failed:[/] {e}")
        return ""


def read_file(path: str) -> str:
    """Auto-detect format and read."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    elif suffix == ".docx":
        return read_docx(path)
    elif suffix in (".txt", ".md", ".rst", ".csv"):
        return read_txt(path)
    else:
        console.print(f"[yellow]Unknown format {suffix}, trying as plain text[/]")
        return read_txt(path)


# ── Main ingestor ─────────────────────────────────────────────────────────────

class Ingestor:
    """
    High-level ingestion interface.
    Feed it files or URLs, it handles everything and stores in memory.

    Usage:
        ingestor = Ingestor(memory, logger)
        ingestor.ingest_file("research_paper.pdf", domain="science")
        ingestor.ingest_url("https://example.com/article", domain="tech")
        ingestor.ingest_folder("./my_documents/", domain="work")
    """

    def __init__(self, memory: Memory, logger: Logger):
        self.memory = memory
        self.logger = logger

    def ingest_file(
        self,
        path: str,
        domain: str = "general",
        chunk_size: int = CHUNK_SIZE,
    ) -> int:
        """Ingest a single file. Returns number of chunks stored."""
        console.print(f"[cyan]Reading:[/] {path}")
        text = read_file(path)
        if not text:
            console.print(f"[red]Could not extract text from {path}[/]")
            return 0
        return self._process(text, source=path, domain=domain, chunk_size=chunk_size)

    def ingest_url(
        self,
        url: str,
        domain: str = "general",
        chunk_size: int = CHUNK_SIZE,
    ) -> int:
        """Ingest a web page. Returns number of chunks stored."""
        console.print(f"[cyan]Fetching:[/] {url}")
        text = read_url(url)
        if not text:
            console.print(f"[red]Could not extract content from {url}[/]")
            return 0
        return self._process(text, source=url, domain=domain, chunk_size=chunk_size)

    def ingest_folder(
        self,
        folder: str,
        domain: str = "general",
        extensions: list[str] = None,
    ) -> int:
        """Ingest all files in a folder recursively."""
        extensions = extensions or [".pdf", ".docx", ".txt", ".md"]
        folder_path = Path(folder)
        files = [f for f in folder_path.rglob("*") if f.suffix.lower() in extensions]

        if not files:
            console.print(f"[yellow]No supported files found in {folder}[/]")
            return 0

        total = 0
        for f in track(files, description=f"Ingesting {len(files)} files..."):
            total += self.ingest_file(str(f), domain=domain)

        console.print(f"[green]Folder ingested:[/] {total} chunks from {len(files)} files")
        return total

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
        domain: str = "general",
    ) -> int:
        """Directly ingest raw text (e.g. from API, paste, or dynamic source)."""
        return self._process(text, source=source, domain=domain)

    # ── Internal ────────────────────────────────────────────────────────────

    def _process(self, text: str, source: str, domain: str, chunk_size: int = CHUNK_SIZE) -> int:
        chunks      = chunk_text(text, chunk_size=chunk_size)
        chunk_dicts = [{"text": c, "source": source, "domain": domain} for c in chunks]

        self.memory.store_many(chunk_dicts)
        self.logger.log_ingestion(
            source=source,
            source_type="file" if not source.startswith("http") else "url",
            chunks=len(chunks),
            domain=domain,
        )
        console.print(f"[green]Ingested:[/] {len(chunks)} chunks → domain='{domain}' source='{source[:60]}'")
        return len(chunks)
