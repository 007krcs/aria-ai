"""
ARIA Windows Launcher
======================
This file is compiled by PyInstaller into ARIA.exe.

What it does:
  1. Finds the Python interpreter (venv preferred, system fallback)
  2. On first run, installs dependencies into a local .venv
  3. Starts the ARIA backend server (server.py) in a background window
  4. Opens the browser to http://localhost:8000
  5. Shows a system-tray-like console window the user can minimize

Keep this file LEAN — import only stdlib modules so PyInstaller
produces a small exe (~5 MB) rather than bundling all of torch/chromadb.
"""

import sys
import os
import subprocess
import socket
import time
import webbrowser
import ctypes
import threading
import signal
from pathlib import Path

# ── Locate project root (works both frozen and unfrozen) ───────────────────────
if getattr(sys, "frozen", False):
    # Running as PyInstaller .exe — walk up from exe until we find server.py
    _exe_dir = Path(sys.executable).resolve().parent
    ROOT = _exe_dir
    for _candidate in [_exe_dir, _exe_dir.parent, _exe_dir.parent.parent]:
        if (_candidate / "server.py").exists():
            ROOT = _candidate
            break
else:
    ROOT = Path(__file__).resolve().parent

os.chdir(ROOT)

# ── Paths ──────────────────────────────────────────────────────────────────────
VENV_DIR  = ROOT / ".venv"
VENV_PY   = VENV_DIR / "Scripts" / "python.exe"
REQ_FILE  = ROOT / "requirements.txt"
SERVER_PY = ROOT / "server.py"
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LAUNCH_LOG = LOG_DIR / "launcher.log"

# ── Console helpers ────────────────────────────────────────────────────────────
def _set_title(t: str):
    try:
        ctypes.windll.kernel32.SetConsoleTitleW(t)
    except Exception:
        pass

def _green(s):  return f"\033[92m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _red(s):    return f"\033[91m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"
def _dim(s):    return f"\033[2m{s}\033[0m"

def info(msg):  print(f"  {_green('▶')}  {msg}", flush=True)
def warn(msg):  print(f"  {_yellow('⚠')}  {msg}", flush=True)
def error(msg): print(f"  {_red('✗')}  {msg}", flush=True)
def step(msg):  print(f"  {_dim('·')}  {msg}", flush=True)

# ── Find Python ────────────────────────────────────────────────────────────────
def _find_python() -> str:
    """Return the best Python executable to use."""
    if VENV_PY.exists():
        return str(VENV_PY)
    # Fall back to system Python (find python3.11+ if possible)
    for candidate in ["python", "python3", "python3.11", "python3.12"]:
        try:
            r = subprocess.run([candidate, "--version"], capture_output=True, text=True)
            if r.returncode == 0 and "3." in r.stdout + r.stderr:
                return candidate
        except FileNotFoundError:
            continue
    return sys.executable  # last resort: the Python that runs THIS launcher

# ── Check dependencies ────────────────────────────────────────────────────────
def _deps_ok(py: str) -> bool:
    """Quick check that key packages are importable."""
    try:
        r = subprocess.run(
            [py, "-c", "import fastapi, uvicorn, chromadb, sentence_transformers"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
        return r.returncode == 0
    except Exception:
        return False

# ── Create venv + install deps ────────────────────────────────────────────────
def _setup_venv():
    """Create .venv and install requirements. Shows progress."""
    step(f"Creating virtual environment in {_dim('.venv')}...")
    r = subprocess.run(
        [sys.executable, "-m", "venv", str(VENV_DIR)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        error(f"Failed to create venv: {r.stderr.strip()}")
        return False

    pip = str(VENV_DIR / "Scripts" / "pip.exe")
    step("Installing ARIA dependencies (this runs once, ~2-5 minutes)...")
    step("You can watch progress in the window below.\n")
    r = subprocess.run(
        [pip, "install", "-r", str(REQ_FILE), "--quiet", "--no-warn-script-location"],
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        warn("Some packages may have failed. ARIA will try to run anyway.")
    return True

# ── Port check ────────────────────────────────────────────────────────────────
def _port_in_use(port: int) -> bool:
    try:
        s = socket.create_connection(("localhost", port), timeout=1)
        s.close()
        return True
    except (ConnectionRefusedError, OSError):
        return False

def _wait_for_server(port=8000, timeout=60) -> bool:
    print(f"  {_dim('Waiting for server')} ", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_in_use(port):
            print(f" {_green('ready!')}", flush=True)
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(f" {_red('timed out.')}")
    return False

# ── Kill process on port ──────────────────────────────────────────────────────
def _kill_port(port: int):
    """Kill whatever process is listening on the given port."""
    try:
        r = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True,
        )
        for line in r.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit():
                    subprocess.run(["taskkill", "/PID", pid, "/F"],
                                   capture_output=True)
                    step(f"Stopped old ARIA process (PID {pid}).")
                    time.sleep(1)
                    return
    except Exception:
        pass

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    _set_title("ARIA - Starting...")

    # Force UTF-8 output on Windows so Unicode chars don't crash
    import io as _io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    # Enable ANSI escape codes on Windows 10+
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

    print()
    print(_bold("  +------------------------------------------+"))
    print(_bold("  |   ARIA  - Adaptive Reasoning AI         |"))
    print(_bold("  +------------------------------------------+"))
    print()

    # ── Step 1: Find Python ───────────────────────────────────────────────────
    py = _find_python()
    info(f"Python: {_dim(py)}")

    # ── Step 2: Check / install deps ─────────────────────────────────────────
    if not _deps_ok(py):
        warn("Dependencies not found. Setting up environment...")
        if not VENV_PY.exists():
            if not _setup_venv():
                error("Setup failed. Run install.py manually.")
                input("\nPress Enter to close...")
                sys.exit(1)
        py = str(VENV_PY) if VENV_PY.exists() else py
        if not _deps_ok(py):
            warn("Some packages still missing — ARIA will attempt to start anyway.")
    else:
        info(f"Dependencies: {_green('OK')}")

    # ── Step 3: Kill stale server on port 8000 ───────────────────────────────
    if _port_in_use(8000):
        info("Stopping previous ARIA instance on port 8000...")
        _kill_port(8000)
        time.sleep(1)
        if _port_in_use(8000):
            error("Could not stop existing process on port 8000.")
            error("Close the other process manually and try again.")
            input("\nPress Enter to close...")
            sys.exit(1)

    # ── Step 5: Start ARIA server ─────────────────────────────────────────────
    info(f"Starting ARIA server…")
    _set_title("ARIA — Running on http://localhost:8000")

    log_file = open(LAUNCH_LOG, "w", encoding="utf-8", errors="replace")

    # Open server in a NEW visible console window so users see errors
    server_proc = subprocess.Popen(
        [py, str(SERVER_PY)],
        cwd=str(ROOT),
        stdout=log_file,
        stderr=log_file,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    # ── Step 6: Wait + open browser ──────────────────────────────────────────
    if _wait_for_server(port=8000, timeout=60):
        ip = _get_local_ip()
        print()
        info("ARIA is live at  http://localhost:8000")
        info(f"Local network:   http://{ip}:8000")
        print()
        webbrowser.open("http://localhost:8000")
    else:
        error("Server did not start in 60 seconds.")
        _show_log_help()

    # ── Step 7: Keep window open ──────────────────────────────────────────────
    print(f"\n  {_dim('Press Ctrl+C or close this window to stop ARIA.')}\n")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        info("Stopping ARIA...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        info("ARIA stopped.")

    log_file.close()


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def _show_log_help():
    print()
    warn(f"Check the log for details: {_dim(str(LAUNCH_LOG))}")
    try:
        lines = LAUNCH_LOG.read_text(encoding="utf-8", errors="replace").strip().split("\n")
        last = [l for l in lines if l.strip()][-10:]
        print(_dim("  Last log lines:"))
        for l in last:
            print(f"    {_dim(l)}")
    except Exception:
        pass
    print()
    print("  Common fixes:")
    print(f"  {_dim('  python install.py        ← fix missing packages')}")
    print(f"  {_dim('  Check logs/launcher.log   ← full error details')}")
    print()
    input("Press Enter to close...")
    sys.exit(1)


if __name__ == "__main__":
    main()
