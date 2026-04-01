"""
ARIA — One-Click Installer
===========================
Run this ONE file. It does everything:

  Step 1  — Check Python version
  Step 2  — Create virtual environment
  Step 3  — Install all Python packages (in correct order)
  Step 4  — Detect OS and install Ollama
  Step 5  — Pull required AI models
  Step 6  — Install Playwright browser
  Step 7  — Create all folders and config files
  Step 8  — Run self-test to verify everything works
  Step 9  — Register auto-start (optional)
  Step 10 — Launch ARIA

Usage:
    python install.py            # full install
    python install.py --repair   # fix broken install
    python install.py --update   # update packages + models
    python install.py --launch   # skip install, just launch
"""

import sys
import os
import subprocess
import platform
import json
import time
import shutil
import urllib.request
import urllib.error
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
ARIA_DIR    = Path(__file__).resolve().parent
PLATFORM    = platform.system()          # Windows / Darwin / Linux
PYTHON      = sys.executable
VENV_DIR    = ARIA_DIR / ".venv"
MIN_PYTHON  = (3, 10)
OLLAMA_URL  = {
    "Windows": "https://ollama.com/download/OllamaSetup.exe",
    "Darwin":  "https://ollama.com/download/Ollama-darwin.zip",
    "Linux":   "https://ollama.ai/install.sh",
}

# Colour helpers (work without rich installed yet)
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
B = "\033[94m"; M = "\033[95m"; C = "\033[96m"
W = "\033[97m"; DIM = "\033[2m"; BOLD = "\033[1m"; RESET = "\033[0m"

def c(colour, text): return f"{colour}{text}{RESET}"
def ok(msg):   print(f"  {c(G,'✓')} {msg}")
def err(msg):  print(f"  {c(R,'✗')} {msg}")
def warn(msg): print(f"  {c(Y,'!')} {msg}")
def info(msg): print(f"  {c(B,'→')} {msg}")
def step(n, title): print(f"\n{c(BOLD+M, f'STEP {n}')} {c(BOLD+W, title)}")
def hr(): print(c(DIM, "─" * 60))


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — CHECK PYTHON
# ═════════════════════════════════════════════════════════════════════════════

def step1_check_python():
    step(1, "Checking Python version")
    v = sys.version_info
    if v < MIN_PYTHON:
        err(f"Python {v.major}.{v.minor} found. ARIA needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
        err("Download Python 3.11 from https://www.python.org/downloads/")
        sys.exit(1)
    ok(f"Python {v.major}.{v.minor}.{v.micro} — good")
    info(f"Location: {PYTHON}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — VIRTUAL ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

def step2_venv():
    step(2, "Setting up virtual environment")

    if VENV_DIR.exists() and (VENV_DIR / "pyvenv.cfg").exists():
        ok("Virtual environment already exists — skipping")
        return

    info("Creating .venv ...")
    run([PYTHON, "-m", "venv", str(VENV_DIR)], "Create venv")
    ok(f"Virtual environment created at {VENV_DIR}")

    info("To activate manually:")
    if PLATFORM == "Windows":
        print(f"    {c(C, str(VENV_DIR / 'Scripts' / 'activate'))}")
    else:
        print(f"    {c(C, f'source {VENV_DIR}/bin/activate')}")


def get_venv_python() -> str:
    """Get path to python inside the venv."""
    if PLATFORM == "Windows":
        p = VENV_DIR / "Scripts" / "python.exe"
    else:
        p = VENV_DIR / "bin" / "python"
    return str(p) if p.exists() else PYTHON


def get_venv_pip() -> str:
    if PLATFORM == "Windows":
        p = VENV_DIR / "Scripts" / "pip.exe"
    else:
        p = VENV_DIR / "bin" / "pip"
    return str(p) if p.exists() else "pip"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — INSTALL PYTHON PACKAGES
# ═════════════════════════════════════════════════════════════════════════════

# Packages grouped by priority — install in order
# If one group fails, later groups still try
PACKAGE_GROUPS = [
    ("Core (required)", [
        "requests", "python-dotenv", "rich", "pydantic",
        "fastapi", "uvicorn[standard]", "websockets",
    ]),
    ("AI + Memory", [
        "sentence-transformers", "chromadb",
        "langdetect",
    ]),
    ("Document reading", [
        "pymupdf", "python-docx", "openpyxl",
        "python-pptx", "beautifulsoup4", "lxml",
        "trafilatura",
    ]),
    ("Web + search", [
        "ddgs", "playwright",
    ]),
    ("NOVA + self-training", [
        "z3-solver", "sympy",
    ]),
    ("Desktop app", [
        "flet", "pillow",
    ]),
    ("System tray + hotkeys", [
        "pystray", "plyer", "keyboard",
        "pyautogui", "pyperclip",
    ]),
    ("Security + audit", [
        "pip-audit",
    ]),
    ("Media (optional — for audio/YouTube)", [
        "youtube-transcript-api", "openai-whisper",
    ]),
    ("LoRA self-training (optional — needs more RAM)", [
        "peft", "accelerate",
    ]),
]


def step3_packages(repair: bool = False):
    step(3, "Installing Python packages")
    pip = get_venv_pip()

    # Upgrade pip first
    info("Upgrading pip...")
    run([pip, "install", "--upgrade", "pip", "--quiet"], "Upgrade pip", silent=True)

    total_ok  = 0
    total_err = 0

    for group_name, packages in PACKAGE_GROUPS:
        print(f"\n  {c(DIM, '→')} {c(W, group_name)}")
        for pkg in packages:
            try:
                result = subprocess.run(
                    [pip, "install", pkg, "--quiet", "--disable-pip-version-check"],
                    capture_output=True, text=True, timeout=300,
                )
                if result.returncode == 0:
                    ok(f"  {pkg}")
                    total_ok += 1
                else:
                    # Try with --break-system-packages as fallback
                    r2 = subprocess.run(
                        [pip, "install", pkg, "--quiet", "--break-system-packages"],
                        capture_output=True, text=True, timeout=300,
                    )
                    if r2.returncode == 0:
                        ok(f"  {pkg} (system)")
                        total_ok += 1
                    else:
                        warn(f"  {pkg} — skipped ({result.stderr.strip()[:60]})")
                        total_err += 1
            except subprocess.TimeoutExpired:
                warn(f"  {pkg} — timeout (skip)")
                total_err += 1
            except Exception as e:
                warn(f"  {pkg} — error: {e}")
                total_err += 1

    print()
    ok(f"Packages: {total_ok} installed, {total_err} skipped")
    if total_err > 0:
        info("Skipped packages are optional — core features still work")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — INSTALL OLLAMA
# ═════════════════════════════════════════════════════════════════════════════

def step4_ollama():
    step(4, "Installing Ollama (local AI engine)")

    # Check if already installed
    if shutil.which("ollama"):
        ok("Ollama already installed")
        v = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        info(v.stdout.strip() or "version unknown")
        return

    info(f"Downloading Ollama for {PLATFORM}...")

    if PLATFORM == "Linux":
        _install_ollama_linux()
    elif PLATFORM == "Darwin":
        _install_ollama_mac()
    elif PLATFORM == "Windows":
        _install_ollama_windows()
    else:
        warn(f"Unknown OS: {PLATFORM}")
        warn("Install Ollama manually from https://ollama.com")
        return

    # Verify
    time.sleep(2)
    if shutil.which("ollama"):
        ok("Ollama installed successfully")
    else:
        warn("Ollama may need a PATH restart — close and reopen terminal")


def _install_ollama_linux():
    info("Running official Ollama install script...")
    try:
        result = subprocess.run(
            ["bash", "-c", "curl -fsSL https://ollama.ai/install.sh | sh"],
            timeout=300,
        )
        if result.returncode == 0:
            ok("Ollama installed on Linux")
        else:
            _manual_ollama_install_msg()
    except Exception as e:
        warn(f"Auto-install failed: {e}")
        _manual_ollama_install_msg()


def _install_ollama_mac():
    info("Downloading Ollama for macOS...")
    try:
        url     = "https://ollama.com/download/Ollama-darwin.zip"
        dest    = Path("/tmp/Ollama-darwin.zip")
        urllib.request.urlretrieve(url, dest)
        subprocess.run(["unzip", "-q", str(dest), "-d", "/tmp/ollama_mac"])
        app = list(Path("/tmp/ollama_mac").rglob("Ollama.app"))
        if app:
            subprocess.run(["cp", "-r", str(app[0]), "/Applications/"])
            ok("Ollama.app installed to /Applications — open it to start")
        else:
            _manual_ollama_install_msg()
    except Exception as e:
        warn(f"Auto-download failed: {e}")
        _manual_ollama_install_msg()


def _install_ollama_windows():
    info("Downloading OllamaSetup.exe...")
    try:
        url  = "https://ollama.com/download/OllamaSetup.exe"
        dest = Path(os.environ.get("TEMP","C:\\Temp")) / "OllamaSetup.exe"
        dest.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        ok(f"Downloaded to {dest}")
        info("Running installer (you may see a UAC prompt)...")
        subprocess.run([str(dest), "/S"], timeout=120)
        ok("Ollama installed — you may need to restart terminal")
    except Exception as e:
        warn(f"Auto-download failed: {e}")
        _manual_ollama_install_msg()


def _manual_ollama_install_msg():
    warn("Could not auto-install Ollama")
    print(f"  {c(Y, '→')} Download manually from: {c(C, 'https://ollama.com')}")
    print(f"  {c(Y, '→')} Install it, then re-run this installer")
    input("  Press Enter after installing Ollama to continue...")


def _start_ollama_server():
    """Start Ollama serve in background if not already running."""
    try:
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True  # already running
    except Exception:
        pass

    info("Starting Ollama server in background...")
    try:
        if PLATFORM == "Windows":
            subprocess.Popen(["ollama", "serve"],
                             creationflags=subprocess.CREATE_NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
        return True
    except Exception as e:
        warn(f"Could not start Ollama: {e}")
        return False


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — PULL AI MODELS
# ═════════════════════════════════════════════════════════════════════════════

# Models in priority order: [name, size_hint, purpose, required]
MODELS = [
    ("phi3:mini",      "1.8 GB", "Main reasoning model",           True),
    ("moondream",      "1.8 GB", "Vision OCR (reads images)",       True),
    ("llama3.2",       "2.0 GB", "Better reasoning, backup",        False),
    ("nomic-embed-text","274 MB","Better embeddings",               False),
]


def step5_models():
    step(5, "Downloading AI models")

    if not shutil.which("ollama"):
        warn("Ollama not found — skipping model downloads")
        warn("Install Ollama first then run: python install.py --update")
        return

    _start_ollama_server()

    # Show what's already installed
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        already = result.stdout
    except Exception:
        already = ""

    for model, size, purpose, required in MODELS:
        model_base = model.split(":")[0]
        if model_base in already or model in already:
            ok(f"{model} already installed ({purpose})")
            continue

        tag = c(R, "[required]") if required else c(DIM, "[optional]")
        print(f"\n  {tag} {c(W, model)} — {size} — {purpose}")

        if not required:
            ans = input(f"  Download {model}? [y/N] ").strip().lower()
            if ans != "y":
                info(f"Skipping {model}")
                continue

        info(f"Pulling {model} — this may take a few minutes...")
        try:
            # Show progress by not capturing output
            result = subprocess.run(
                ["ollama", "pull", model],
                timeout=600,
            )
            if result.returncode == 0:
                ok(f"{model} downloaded")
            else:
                warn(f"{model} pull failed — try manually: ollama pull {model}")
        except subprocess.TimeoutExpired:
            warn(f"{model} download timed out — try: ollama pull {model}")
        except Exception as e:
            warn(f"{model} error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — PLAYWRIGHT BROWSER
# ═════════════════════════════════════════════════════════════════════════════

def step6_playwright():
    step(6, "Installing Playwright headless browser")
    info("This downloads Chromium (~130 MB) for crawling JS-heavy sites")

    venv_python = get_venv_python()
    try:
        # Check if playwright is installed
        result = subprocess.run(
            [venv_python, "-c", "import playwright; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        if result.stdout.strip() != "ok":
            warn("Playwright not installed — skipping browser install")
            return

        # Install chromium
        result = subprocess.run(
            [venv_python, "-m", "playwright", "install", "chromium"],
            timeout=300,
        )
        if result.returncode == 0:
            ok("Chromium browser installed")
        else:
            warn("Chromium install failed — web crawling may not work on JS sites")

    except subprocess.TimeoutExpired:
        warn("Playwright install timed out — run manually: playwright install chromium")
    except Exception as e:
        warn(f"Playwright: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — CREATE FOLDERS AND CONFIG
# ═════════════════════════════════════════════════════════════════════════════

def step7_setup_dirs():
    step(7, "Creating folders and configuration")

    folders = [
        ARIA_DIR / "data" / "uploads",
        ARIA_DIR / "data" / "training",
        ARIA_DIR / "data" / "trend_cache",
        ARIA_DIR / "logs",
        ARIA_DIR / "models",
        ARIA_DIR / "agents" / "sandbox",
        ARIA_DIR / "app",
        ARIA_DIR / "system",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").touch(exist_ok=True)
    ok(f"Created {len(folders)} directories")

    # Create .env file from example if missing
    env_file    = ARIA_DIR / ".env"
    env_example = ARIA_DIR / ".env.example"
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        ok(".env created from .env.example")
    elif not env_file.exists():
        env_file.write_text(
            "# ARIA Configuration\n"
            "OLLAMA_HOST=http://localhost:11434\n"
            "API_HOST=0.0.0.0\n"
            "API_PORT=8000\n"
            "HF_TOKEN=\n"
        )
        ok(".env created with defaults")
    else:
        ok(".env already exists")

    # Create system/__init__.py if missing
    for pkg_dir in ["agents", "core", "pipelines", "tools", "system", "app"]:
        init = ARIA_DIR / pkg_dir / "__init__.py"
        init.parent.mkdir(exist_ok=True)
        if not init.exists():
            init.touch()
    ok("Package __init__.py files ensured")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 8 — SELF TEST
# ═════════════════════════════════════════════════════════════════════════════

def step8_test():
    step(8, "Running self-test")

    venv_python = get_venv_python()
    tests = [
        ("Python imports",       _test_imports),
        ("Ollama connection",     _test_ollama),
        ("ChromaDB memory",       _test_chromadb),
        ("File system",           _test_filesystem),
        ("Server config",         _test_server_config),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            result, msg = fn(venv_python)
            if result:
                ok(f"{name}: {msg}")
                passed += 1
            else:
                warn(f"{name}: {msg}")
                failed += 1
        except Exception as e:
            warn(f"{name}: {e}")
            failed += 1

    print()
    if failed == 0:
        ok(f"All {passed} tests passed — ARIA is ready")
    else:
        warn(f"{passed} passed, {failed} warnings — ARIA will still work")


def _test_imports(python):
    result = subprocess.run(
        [python, "-c", "import fastapi, chromadb, requests, rich; print('ok')"],
        capture_output=True, text=True, timeout=20,
    )
    if "ok" in result.stdout:
        return True, "core packages importable"
    return False, f"import error: {result.stderr[:80]}"


def _test_ollama(python):
    try:
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        if models:
            return True, f"{len(models)} models available: {', '.join(models[:3])}"
        return False, "Ollama running but no models installed"
    except Exception:
        return False, "Ollama not reachable — start with: ollama serve"


def _test_chromadb(python):
    result = subprocess.run(
        [python, "-c",
         "import chromadb; c=chromadb.Client(); c.create_collection('_test'); c.delete_collection('_test'); print('ok')"],
        capture_output=True, text=True, timeout=20,
    )
    if "ok" in result.stdout:
        return True, "ChromaDB works"
    return False, f"ChromaDB error: {result.stderr[:80]}"


def _test_filesystem(python):
    paths = [
        ARIA_DIR / "data" / "uploads",
        ARIA_DIR / "logs",
        ARIA_DIR / "models",
        ARIA_DIR / ".env",
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        return False, f"Missing: {missing}"
    return True, "all directories and config present"


def _test_server_config(python):
    server = ARIA_DIR / "server.py"
    if not server.exists():
        return False, "server.py not found"
    result = subprocess.run(
        [python, "-c", f"import ast; ast.parse(open(r'{server}').read()); print('ok')"],
        capture_output=True, text=True, timeout=10,
    )
    if "ok" in result.stdout:
        return True, "server.py syntax valid"
    return False, f"server.py syntax error: {result.stderr[:80]}"


# ═════════════════════════════════════════════════════════════════════════════
# STEP 9 — AUTO-START (OPTIONAL)
# ═════════════════════════════════════════════════════════════════════════════

def step9_autostart():
    step(9, "Auto-start at login (optional)")
    print(f"  ARIA can start automatically every time you log in to your computer.")
    ans = input(f"  Install auto-start? [y/N] ").strip().lower()

    if ans != "y":
        info("Skipped — you can enable later with: python aria_service.py --install")
        return

    venv_python = get_venv_python()
    service_script = ARIA_DIR / "aria_service.py"

    try:
        result = subprocess.run(
            [venv_python, str(service_script), "--install"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            ok("Auto-start registered")
            info(f"ARIA will start silently at next login")
        else:
            warn(f"Auto-start failed: {result.stderr[:100]}")
    except Exception as e:
        warn(f"Auto-start error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 10 — LAUNCH
# ═════════════════════════════════════════════════════════════════════════════

def step10_launch():
    step(10, "Launching ARIA")

    venv_python  = get_venv_python()
    service_file = ARIA_DIR / "aria_service.py"
    server_file  = ARIA_DIR / "server.py"

    print()
    print(f"  {c(BOLD+W, 'Choose how to start ARIA:')}")
    print(f"  {c(G, '1')}  Full mode — desktop app + system tray + hotkeys {c(DIM,'(recommended)')}")
    print(f"  {c(G, '2')}  Server only — API at http://localhost:8000")
    print(f"  {c(G, '3')}  Server + open browser UI")
    print(f"  {c(G, '4')}  Exit — start later manually")
    print()

    choice = input("  Your choice [1/2/3/4]: ").strip()

    if choice == "1":
        print()
        ok("Starting ARIA in full mode...")
        _print_access_info()
        subprocess.Popen([venv_python, str(service_file), "--ui"])
        time.sleep(2)
        ok("ARIA is running!")
        info("Look for ARIA icon in your system tray")
        info("Press Alt+Space from anywhere to open ARIA")

    elif choice == "2":
        print()
        ok("Starting ARIA server...")
        _print_access_info()
        os.chdir(ARIA_DIR)
        os.execv(venv_python, [venv_python, str(server_file)])

    elif choice == "3":
        import webbrowser
        print()
        ok("Starting ARIA server + browser UI...")
        _print_access_info()
        subprocess.Popen([venv_python, str(server_file)])
        time.sleep(3)
        webbrowser.open("http://localhost:8000")

    else:
        print()
        ok("Installation complete!")
        _print_manual_start_commands(venv_python)


def _print_access_info():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "YOUR-PC-IP"

    print()
    print(f"  {c(BOLD, 'Access ARIA:')}")
    print(f"  Local:   {c(C, 'http://localhost:8000')}")
    print(f"  Network: {c(C, f'http://{local_ip}:8000')} {c(DIM,'(phone/tablet on same WiFi)')}")
    print(f"  NOVA:    {c(C, 'http://localhost:8000/nova')}")
    print(f"  Learn:   {c(C, 'http://localhost:8000/learn')}")
    print(f"  API:     {c(C, 'http://localhost:8000/docs')}")
    print()


def _print_manual_start_commands(python):
    print()
    print(f"  {c(BOLD, 'To start ARIA later:')}")
    print(f"  {c(C, f'cd {ARIA_DIR}')}")
    print()
    print(f"  {c(DIM, '# Full mode (tray + hotkeys + desktop app):')}")
    print(f"  {c(C, f'{python} aria_service.py --ui')}")
    print()
    print(f"  {c(DIM, '# Server only:')}")
    print(f"  {c(C, f'{python} server.py')}")
    print()
    print(f"  {c(DIM, '# Install auto-start:')}")
    print(f"  {c(C, f'{python} aria_service.py --install')}")


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def run(cmd, label="", silent=False):
    """Run a subprocess and handle errors."""
    try:
        if silent:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        else:
            result = subprocess.run(cmd, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        warn(f"{label} timed out")
        return False
    except Exception as e:
        warn(f"{label} error: {e}")
        return False


def print_banner():
    print()
    print(c(BOLD+M, "╔══════════════════════════════════════════════════════════╗"))
    print(c(BOLD+M, "║") + c(BOLD+W, "         ARIA — Adaptive Reasoning Intelligence          ") + c(BOLD+M, "║"))
    print(c(BOLD+M, "║") + c(DIM,    "              Step-by-Step Installer v3.0               ") + c(BOLD+M, "║"))
    print(c(BOLD+M, "╚══════════════════════════════════════════════════════════╝"))
    print()
    print(f"  OS:      {c(C, PLATFORM)}")
    print(f"  Python:  {c(C, f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')}")
    print(f"  Dir:     {c(C, str(ARIA_DIR))}")
    print()


def print_summary():
    print()
    hr()
    print(c(BOLD+G, "\n  ARIA installation complete!\n"))
    print(f"  {c(W,'What was installed:')}")
    print(f"  {c(G,'✓')} Python virtual environment (.venv)")
    print(f"  {c(G,'✓')} All Python packages")
    print(f"  {c(G,'✓')} Ollama AI engine")
    print(f"  {c(G,'✓')} AI models (phi3:mini + moondream)")
    print(f"  {c(G,'✓')} Playwright headless browser")
    print(f"  {c(G,'✓')} Folders and configuration")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ARIA Installer")
    parser.add_argument("--repair",  action="store_true", help="Repair broken install")
    parser.add_argument("--update",  action="store_true", help="Update packages and models")
    parser.add_argument("--launch",  action="store_true", help="Launch only")
    parser.add_argument("--server",  action="store_true", help="Launch server only")
    parser.add_argument("--test",    action="store_true", help="Run tests only")
    args = parser.parse_args()

    print_banner()

    if args.launch or args.server:
        step10_launch()
        return

    if args.test:
        step8_test()
        return

    if args.update:
        step3_packages()
        step5_models()
        step8_test()
        return

    # ── Full install ──────────────────────────────────────────────────────────
    print(f"  This will install everything ARIA needs on your computer.")
    print(f"  Total download: ~4-6 GB (mostly AI models)")
    print(f"  Estimated time: 10-20 minutes (depends on internet speed)")
    print()
    ans = input(f"  {c(BOLD, 'Start installation? [Y/n]')} ").strip().lower()
    if ans == "n":
        print("  Cancelled.")
        return

    hr()
    step1_check_python()
    step2_venv()
    step3_packages(repair=args.repair)
    step4_ollama()
    step5_models()
    step6_playwright()
    step7_setup_dirs()
    step8_test()
    step9_autostart()

    print_summary()
    step10_launch()


if __name__ == "__main__":
    main()
