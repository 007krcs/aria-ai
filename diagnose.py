"""
ARIA Diagnostics
=================
Run this to find out exactly why ARIA is not starting.

    python diagnose.py

It checks every dependency and prints exactly what to install.
"""

import sys
import os
import json
import subprocess
import socket
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
W = "\033[1m";  D = "\033[2m";  X = "\033[0m"

def ok(msg):   print(f"  {G}✓{X}  {msg}")
def fail(msg): print(f"  {R}✗{X}  {msg}")
def warn(msg): print(f"  {Y}!{X}  {msg}")
def info(msg): print(f"  {D}{msg}{X}")


print(f"\n{W}ARIA Diagnostics{X}\n{'─'*50}\n")

# ── 1. Python version ──────────────────────────────────────────────────────
v = sys.version_info
if v >= (3, 10):
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(f"Python {v.major}.{v.minor} — need 3.10+")
    print(f"  {C}Download: https://python.org/downloads{X}")

# ── 2. Core packages ───────────────────────────────────────────────────────
print("\nCore packages:")
core = [
    ("fastapi",              "pip install fastapi"),
    ("uvicorn",              "pip install uvicorn"),
    ("requests",             "pip install requests"),
    ("pydantic",             "pip install pydantic"),
    ("rich",                 "pip install rich"),
    ("python_dotenv",        "pip install python-dotenv"),
    ("chromadb",             "pip install chromadb"),
    ("sentence_transformers","pip install sentence-transformers"),
]
core_ok = True
for pkg, fix in core:
    try:
        __import__(pkg)
        ok(pkg)
    except ImportError:
        fail(f"{pkg}   →   {C}{fix}{X}")
        core_ok = False

# ── 3. Document packages ───────────────────────────────────────────────────
print("\nDocument reading (optional):")
docs = [
    ("fitz",     "pip install pymupdf"),
    ("docx",     "pip install python-docx"),
    ("openpyxl", "pip install openpyxl"),
    ("pptx",     "pip install python-pptx"),
    ("bs4",      "pip install beautifulsoup4"),
    ("trafilatura","pip install trafilatura"),
]
for pkg, fix in docs:
    try:
        __import__(pkg)
        ok(pkg)
    except ImportError:
        warn(f"{pkg}  (optional)  →  {C}{fix}{X}")

# ── 4. Ollama ──────────────────────────────────────────────────────────────
print("\nOllama AI engine:")
try:
    import urllib.request
    r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
    import json
    data = json.loads(r.read())
    models = [m["name"] for m in data.get("models", [])]
    if models:
        ok(f"Ollama running — models: {', '.join(models[:4])}")
    else:
        warn("Ollama running but NO MODELS installed")
        print(f"  {C}Run: ollama pull phi3:mini{X}")
except Exception as e:
    fail(f"Ollama not reachable: {e}")
    print(f"  {C}Download Ollama: https://ollama.com{X}")
    print(f"  {C}Then run: ollama serve{X}")

# ── 5. ARIA server modules ─────────────────────────────────────────────────
print("\nARIA server modules:")
modules = [
    ("core.engine",  "core/engine.py"),
    ("core.memory",  "core/memory.py"),
    ("tools.logger", "tools/logger.py"),
    ("agents.agents","agents/agents.py"),
    ("agents.doc_agents","agents/doc_agents.py"),
    ("agents.smart_agents","agents/smart_agents.py"),
    ("pipelines.adaptation","pipelines/adaptation.py"),
]
for mod, path in modules:
    try:
        import importlib
        importlib.import_module(mod)
        ok(mod)
    except Exception as e:
        fail(f"{mod}: {Y}{e}{X}")

# ── 6. Try actually starting the server ───────────────────────────────────
print("\nServer start test:")
try:
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'.');  "
         "from core.engine import Engine; "
         "from core.memory import Memory; "
         "from tools.logger import Logger; "
         "e=Engine(); m=Memory(); l=Logger(); print('CORE_OK')"],
        capture_output=True, text=True, timeout=30,
        cwd=str(ROOT),
    )
    if "CORE_OK" in result.stdout:
        ok("Core modules load successfully")
    else:
        fail("Core modules failed to load")
        if result.stderr:
            lines = result.stderr.strip().split("\n")
            for line in lines[-5:]:
                info(f"  {line}")
        print(f"\n  {Y}Most common fix:{X}")
        print(f"  {C}pip install chromadb sentence-transformers torch{X}")
except subprocess.TimeoutExpired:
    warn("Core load timed out (may be downloading model weights — normal on first run)")
except Exception as e:
    fail(f"Could not test: {e}")

# ── 7. Port 8000 ───────────────────────────────────────────────────────────
print("\nNetwork:")
try:
    s = socket.create_connection(("localhost", 8000), timeout=1)
    s.close()
    ok("Port 8000 is already open — ARIA may already be running!")
    print(f"  Try opening: {C}http://localhost:8000{X}")
except ConnectionRefusedError:
    warn("Port 8000 not open — server is not running (expected if you just installed)")
except Exception as e:
    warn(f"Port check: {e}")

# ── 8. Summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
if core_ok:
    print(f"\n{G}{W}Core packages look good.{X}")
    print(f"\nStart ARIA with:")
    print(f"  {C}python server.py{X}   ← run this to see the exact error if it fails")
    print(f"  {C}python run.py{X}       ← run this once server.py works\n")
else:
    print(f"\n{R}{W}Some core packages are missing.{X}")
    print(f"\nFix all missing packages above, then run:")
    print(f"  {C}python diagnose.py{X}  ← run again to verify")
    print(f"  {C}python server.py{X}    ← try starting the server\n")
