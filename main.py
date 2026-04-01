"""
ARIA — Main entry point
Wires every module together into one working system.

Usage:
    python main.py chat                  # interactive chat
    python main.py ingest file.pdf       # ingest a document
    python main.py ingest https://...    # ingest a web page
    python main.py ingest ./folder/      # ingest entire folder
    python main.py adapt                 # run one adaptation cycle
    python main.py stats                 # show performance stats
    python main.py memory                # inspect what is stored in memory
    python main.py upload  <file>        # upload + process a document into knowledge base
    python main.py read    <file> <q>    # ask a question about a specific document
    python main.py kb                    # knowledge base dashboard (sources, stats)
    python main.py kb-ask  <question>    # ask a question across entire knowledge base
    python main.py server                # start FastAPI server (plugin gateway)
    python main.py setup                 # check everything is installed
"""

import sys
import os
from pathlib import Path

# ── Fix Python path so all modules resolve correctly on Windows + Mac + Linux ──
# This ensures `from core.engine import Engine` works no matter where you
# run main.py from (python main.py / python aria/main.py / cd aria && python main.py)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)   # also set working dir so relative paths (data/, logs/) work

import time
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def build_aria():
    """
    Instantiate every module and wire them together.
    Returns the fully assembled MetaAgent — the one entry point for everything.
    """
    from core.engine import Engine
    from core.memory import Memory
    from tools.logger import Logger
    from agents.agents import ResearcherAgent, ReasonerAgent, CriticAgent, MetaAgent
    from agents.doc_agents import UploadAgent, ProcessorAgent, ReaderAgent, KnowledgeAgent
    from pipelines.adaptation import AdaptationEngine
    from pipelines.ingestion import Ingestor

    console.print(Panel("[bold]ARIA — Adaptive Reasoning Intelligence Architecture[/]", style="blue"))
    console.print("Initializing modules...\n")

    engine     = Engine()
    memory     = Memory()
    logger     = Logger()

    researcher = ResearcherAgent(engine, memory, logger)
    reasoner   = ReasonerAgent(engine, memory, logger)
    critic     = CriticAgent(engine, memory, logger)
    meta       = MetaAgent(engine, memory, logger, researcher, reasoner, critic)

    adaptation = AdaptationEngine(logger, engine, memory)
    adaptation.load_tuned_prompts()

    ingestor   = Ingestor(memory, logger)

    # Document agents
    uploader   = UploadAgent(engine, memory, logger)
    processor  = ProcessorAgent(engine, memory, logger)
    reader     = ReaderAgent(engine, memory, logger)
    kb_agent   = KnowledgeAgent(engine, memory, logger)

    return meta, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent


def cmd_chat(meta):
    """Interactive chat loop."""
    console.print("\n[bold green]ARIA is ready.[/] Type your question. 'quit' to exit.\n")

    while True:
        try:
            query = Prompt.ask("[bold blue]You[/]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "bye"):
            console.print("[dim]Goodbye.[/]")
            break

        result = meta.run(query)

        console.print(f"\n[bold green]ARIA[/] [dim](conf={result['confidence']:.2f}, "
                      f"{result['latency_ms']}ms, intent={result['intent']})[/]")
        console.print(result["answer"])

        if result.get("sources"):
            console.print(f"[dim]Sources: {', '.join(str(s) for s in result['sources'][:3])}[/]")
        console.print()


def cmd_ingest(ingestor, target: str):
    """Ingest a file, URL, or folder."""
    domain = Prompt.ask("Domain/category for this knowledge", default="general")

    if target.startswith("http://") or target.startswith("https://"):
        n = ingestor.ingest_url(target, domain=domain)
    elif __import__("pathlib").Path(target).is_dir():
        n = ingestor.ingest_folder(target, domain=domain)
    else:
        n = ingestor.ingest_file(target, domain=domain)

    console.print(f"\n[green]Done.[/] {n} chunks stored in memory.")


def cmd_stats(memory, logger):
    """Print system statistics."""
    from rich.table import Table

    console.rule("[bold]ARIA System Stats[/]")

    log_stats = logger.get_stats()
    mem_stats = memory.stats()

    table = Table()
    table.add_column("Module")
    table.add_column("Metric")
    table.add_column("Value")

    table.add_row("Memory",  "Total chunks",       str(mem_stats["total_chunks"]))
    table.add_row("Memory",  "DB path",            mem_stats["db_path"])
    table.add_row("Logs",    "Total interactions", str(log_stats["total_interactions"]))
    table.add_row("Logs",    "Success rate",       f"{log_stats['success_rate']}%")
    table.add_row("Logs",    "Avg confidence",     str(log_stats["avg_confidence"]))
    table.add_row("Logs",    "Sources ingested",   str(log_stats["ingested_sources"]))
    table.add_row("Agents",  "Active agents",      str(log_stats["active_agents"]))

    console.print(table)


def cmd_adapt(adaptation):
    """Run one adaptation cycle manually."""
    console.print("[bold]Running adaptation cycle...[/]")
    summary = adaptation.run_cycle()
    console.print(f"\nDone: {summary}")


def cmd_server(meta, ingestor):
    """Start the FastAPI plugin gateway (NeuralOrchestrator-powered)."""
    try:
        import asyncio, json as _json
        import uvicorn
        # Windows SSL fix — Python often lacks the system CA bundle; patch to skip verify
        import ssl as _ssl_mod
        try:
            _ssl_mod._create_default_https_context = _ssl_mod._create_unverified_context
        except Exception:
            pass
        try:
            import urllib3
            urllib3.disable_warnings()
        except Exception:
            pass
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from core.config import API_HOST, API_PORT
        from core.memory import Memory

        # ── Wire up NeuralOrchestrator ─────────────────────────────────────────
        _neural = None
        try:
            from core.neural_bus      import NeuralBus
            from core.synaptic_state  import SynapticState
            from agents.neural_orchestrator import NeuralOrchestrator
            _synaptic = SynapticState()
            _bus      = NeuralBus(synaptic_state=_synaptic)
            _neural   = NeuralOrchestrator(
                aria_components={"engine": meta.engine, "memory": meta.memory,
                                 "logger": meta.logger},
                neural_bus=_bus,
                synaptic_state=_synaptic,
            )
            console.print("  [green]NeuralOrchestrator active[/] — physics/math/quantum solvers enabled")
        except Exception as _ne:
            console.print(f"  [yellow]NeuralOrchestrator unavailable ({type(_ne).__name__}), using MetaAgent fallback[/]")

        app = FastAPI(title="ARIA API", description="Adaptive Reasoning Intelligence Architecture")
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        class QueryRequest(BaseModel):
            query: str
            stream: bool = False

        class IngestRequest(BaseModel):
            source: str   # URL or text
            domain: str = "general"

        async def _neural_run(query: str) -> dict:
            """Collect NeuralOrchestrator SSE stream into a single result dict."""
            answer = ""
            intent = ""
            confidence = 0.0
            sources = []
            reasoning = ""
            steps = []
            mode_used = ""
            latency_start = asyncio.get_event_loop().time()

            async for chunk in _neural.stream(query):
                chunk = chunk.strip()
                if not chunk or not chunk.startswith("data:"):
                    continue
                try:
                    data = _json.loads(chunk[5:].strip())
                except Exception:
                    continue
                t = data.get("type", "")
                if t == "answer":
                    answer = data.get("text", answer)
                    intent = data.get("intent", intent)
                    confidence = data.get("confidence", confidence)
                    sources = data.get("sources", sources)
                    reasoning = data.get("reasoning", reasoning)
                    steps = data.get("steps", steps)
                    mode_used = data.get("mode_used", mode_used)
                elif t == "token" and not answer:
                    answer += data.get("text", "")

            latency_ms = int((asyncio.get_event_loop().time() - latency_start) * 1000)
            return {
                "answer":     answer,
                "intent":     intent,
                "confidence": confidence,
                "sources":    sources,
                "reasoning":  reasoning,
                "steps":      steps,
                "mode_used":  mode_used,
                "latency_ms": latency_ms,
            }

        @app.post("/query")
        async def query(req: QueryRequest):
            if _neural is not None:
                return await _neural_run(req.query)
            # Fallback to MetaAgent
            result = meta.run(req.query)
            return result

        @app.post("/ingest")
        def ingest(req: IngestRequest):
            if req.source.startswith("http"):
                n = ingestor.ingest_url(req.source, domain=req.domain)
            else:
                n = ingestor.ingest_text(req.source, domain=req.domain)
            return {"chunks_stored": n}

        @app.get("/health")
        def health():
            return {"status": "ok", "memory_chunks": meta.memory.count(),
                    "orchestrator": "neural" if _neural else "meta"}

        console.print(f"\n[green]ARIA API running at http://{API_HOST}:{API_PORT}[/]")
        console.print("POST /query   — ask a question")
        console.print("POST /ingest  — ingest a URL or text")
        console.print("GET  /health  — system health\n")

        uvicorn.run(app, host=API_HOST, port=API_PORT)

    except ImportError:
        console.print("[red]Install fastapi and uvicorn:[/] pip install fastapi uvicorn")


def cmd_memory(memory):
    """Interactive memory inspector — see what ARIA has stored."""
    from rich.table import Table
    from rich.prompt import Prompt

    console.rule("[bold]Memory Inspector[/]")
    stats = memory.stats()
    console.print(f"  Total chunks stored: [bold]{stats['total_chunks']}[/]")
    console.print(f"  Database path:       [dim]{stats['db_path']}[/]\n")

    if stats['total_chunks'] == 0:
        console.print("[yellow]Memory is empty.[/] Feed documents with: python main.py ingest <file or url>")
        return

    while True:
        query = Prompt.ask("[bold blue]Search memory (or 'quit')[/]").strip()
        if not query or query.lower() == "quit":
            break

        hits = memory.search(query, top_k=5, min_similarity=0.0)
        if not hits:
            console.print("[yellow]No results found.[/]")
            continue

        table = Table(title=f"Top results for: {query}", show_lines=True)
        table.add_column("#",       width=3)
        table.add_column("Score",   width=6)
        table.add_column("Source",  width=30)
        table.add_column("Preview", width=60)

        for i, h in enumerate(hits, 1):
            preview = h["text"][:120].replace("\n", " ") + "..."
            table.add_row(
                str(i),
                str(h["similarity"]),
                h["source"][-30:],
                preview,
            )
        console.print(table)
        console.print()



def cmd_upload(uploader, processor, target: str, domain: str = "general"):
    """Upload + process a file or URL into the knowledge base."""
    from rich.prompt import Prompt

    if not domain or domain == "general":
        domain = Prompt.ask("Domain/topic for this document", default="general")

    # Upload
    if target.startswith("http"):
        job = uploader.upload_url(target, domain=domain)
    else:
        job = uploader.upload_file(target, domain=domain)

    if job.get("status") == "error":
        console.print(f"[red]Upload failed:[/] {job['error']}")
        return

    # Process
    console.print()
    report = processor.process(job)

    if report.get("status") == "error":
        console.print(f"[red]Processing failed:[/] {report['error']}")
        return

    # Show report
    console.print()
    console.print(f"[bold green]Document processed successfully[/]")
    console.print(f"  Domain:    {report['domain']}")
    console.print(f"  Language:  {report['language']}")
    console.print(f"  Words:     {report['word_count']}")
    console.print(f"  Chunks:    {report['chunks']} stored in memory")
    console.print(f"\n[bold]Summary:[/]\n{report['summary']}")


def cmd_read(reader, source: str, question: str = ""):
    """Ask a question about a specific document."""
    from rich.prompt import Prompt

    if not question:
        question = Prompt.ask(f"What do you want to know from '{Path(source).name}'")

    result = reader.answer(question, source)

    console.print(f"\n[bold green]Reader[/] [dim](conf={result['confidence']:.2f}, {result['relevant_chunks']} chunks from this document)[/]")
    console.print(result["answer"])

    # Interactive loop — keep asking about this document
    console.print()
    while True:
        try:
            q = Prompt.ask("[dim]Ask another question about this doc (or 'quit')[/]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() == "quit":
            break
        result = reader.answer(q, source)
        console.print(f"\n[bold green]Reader[/] [dim](conf={result['confidence']:.2f})[/]")
        console.print(result["answer"])
        console.print()


def cmd_kb(kb_agent):
    """Knowledge base dashboard."""
    from rich.prompt import Prompt

    console.rule("[bold]Knowledge Base[/]")
    kb_agent.print_stats()
    console.print()
    kb_agent.print_sources()

    console.print()
    summary = kb_agent.summarise_kb()
    console.print(f"\n[bold]What this knowledge base covers:[/]\n{summary}")




def cmd_kb_ask(kb_agent, question: str = ""):
    """Ask a question across the entire knowledge base."""
    from rich.prompt import Prompt

    if not question:
        question = Prompt.ask("[bold blue]Ask your knowledge base[/]").strip()

    if not question:
        return

    result = kb_agent.ask(question)
    console.print(f"[bold green]Knowledge Base[/] [dim]({result['chunks_used']} chunks used)[/]")
    console.print(result["answer"])
    if result.get("sources"):
        console.print(f"[dim]Sources: {', '.join(str(s)[-40:] for s in result['sources'][:3])}[/]")

    # Loop
    console.print()
    while True:
        try:
            q = Prompt.ask("[dim]Another question (or 'quit')[/]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q or q.lower() == "quit":
            break
        result = kb_agent.ask(q)
        console.print(f"[bold green]Knowledge Base[/]")
        console.print(result["answer"])
        console.print()


def cmd_setup():
    """Check everything is installed and Ollama is running."""
    import platform
    is_windows = platform.system() == "Windows"

    console.rule("[bold]ARIA Setup Check[/]")
    console.print(f"  OS: {platform.system()} | Python: {platform.python_version()} | Root: {PROJECT_ROOT}\n")

    checks = {
        "requests":              "import requests",
        "rich":                  "from rich.console import Console",
        "chromadb":              "import chromadb",
        "sentence_transformers": "from sentence_transformers import SentenceTransformer",
        "langgraph":             "import langgraph",
        "langchain":             "import langchain",
        "pymupdf":               "import fitz",
        "bs4":                   "from bs4 import BeautifulSoup",
        "trafilatura":           "import trafilatura",
        "langdetect":            "from langdetect import detect",
        "fastapi":               "import fastapi",
        "sqlalchemy":            "import sqlalchemy",
        "pydantic":              "import pydantic",
        "dotenv":                "from dotenv import load_dotenv",
        "ollama_running":        "import requests; requests.get('http://localhost:11434/api/tags', timeout=3)",
    }

    missing = []
    for name, check in checks.items():
        try:
            exec(check)
            console.print(f"  [green]✓[/] {name}")
        except Exception as e:
            label = "Ollama not running — start it first" if name == "ollama_running" else str(e).split("(")[0]
            console.print(f"  [red]✗[/] {name} — {label}")
            if name != "ollama_running":
                missing.append(name)

    console.print()
    if missing:
        console.print(f"[yellow]{len(missing)} packages missing. Run:[/]")
        console.print("  [bold]pip install -r requirements.txt[/]\n")
    else:
        console.print("[green]All Python packages installed.[/]\n")

    console.print("[bold]Ollama setup:[/]")
    if is_windows:
        console.print("  Download: [cyan]https://ollama.com/download[/]")
        console.print("  Then open a new terminal and run: [bold]ollama serve[/]")
    else:
        console.print("  Install: [bold]curl -fsSL https://ollama.com/install.sh | sh[/]")
        console.print("  Start:   [bold]ollama serve[/]")

    console.print("\n[bold]Pull a model (pick by RAM):[/]")
    console.print("  4GB RAM:  [bold]ollama pull phi3:mini[/]")
    console.print("  8GB RAM:  [bold]ollama pull llama3.1:8b[/]   (recommended)")
    console.print("  16GB RAM: [bold]ollama pull llama3.1:13b[/]  (smarter)")
    console.print("\nThen run: [bold]python main.py chat[/]")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "setup":
        cmd_setup()
        sys.exit(0)

    cmd = args[0]

    if cmd == "chat":
        meta, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_chat(meta)

    elif cmd == "ingest":
        if len(args) < 2:
            console.print("[red]Usage:[/] python main.py ingest <file|url|folder>")
            sys.exit(1)
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_ingest(ingestor, args[1])

    elif cmd == "adapt":
        meta, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_adapt(adaptation)

    elif cmd == "memory":
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_memory(memory)

    elif cmd == "stats":
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_stats(memory, logger)

    elif cmd == "upload":
        if len(args) < 2:
            console.print("[red]Usage:[/] python main.py upload <file_or_url> [domain]")
            sys.exit(1)
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        domain = args[2] if len(args) > 2 else "general"
        cmd_upload(uploader, processor, args[1], domain)

    elif cmd == "read":
        if len(args) < 2:
            console.print("[red]Usage:[/] python main.py read <source_file> [question]")
            sys.exit(1)
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        question = " ".join(args[2:]) if len(args) > 2 else ""
        cmd_read(reader, args[1], question)

    elif cmd == "kb":
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_kb(kb_agent)

    elif cmd == "kb-ask":
        _, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        question = " ".join(args[1:]) if len(args) > 1 else ""
        cmd_kb_ask(kb_agent, question)

    elif cmd == "server":
        meta, adaptation, ingestor, memory, logger, uploader, processor, reader, kb_agent = build_aria()
        cmd_server(meta, ingestor)

    else:
        console.print(f"[red]Unknown command:[/] {cmd}")
        console.print("Commands: chat | ingest | adapt | stats | server | setup")
