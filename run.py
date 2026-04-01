"""
ARIA Quick Launcher
====================
Usage:
    python run.py           # server + opens browser (default)
    python run.py --tray    # server + system tray + hotkeys
    python run.py --app     # server + Flet desktop app
    python run.py --check   # check what's installed without starting
"""

import sys
import os
import subprocess
import socket
import time
import webbrowser
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Use venv python if it exists
if sys.platform == "win32":
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    VENV_PY = ROOT / ".venv" / "bin" / "python"

PY = str(VENV_PY) if VENV_PY.exists() else sys.executable


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def wait_for_server(host="localhost", port=8000, timeout=40) -> bool:
    """
    Poll until the server accepts connections or timeout.
    Shows a progress indicator while waiting.
    """
    print("  Waiting for server", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close()
            print(" ready!")
            return True
        except (ConnectionRefusedError, OSError):
            print(".", end="", flush=True)
            time.sleep(1)
    print(" timed out!")
    return False


def check_install():
    """Check which components are installed."""
    print("\n  Checking ARIA installation...\n")

    checks = [
        ("Python packages",
         [PY, "-c", "import fastapi, uvicorn, chromadb; print('ok')"]),
        ("Sentence transformers",
         [PY, "-c", "import sentence_transformers; print('ok')"]),
        ("Ollama reachable",
         [PY, "-c",
          "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags',timeout=3); print('ok')"]),
        ("Flet (desktop UI)",
         [PY, "-c", "import flet; print('ok')"]),
        ("Playwright (web crawler)",
         [PY, "-c", "import playwright; print('ok')"]),
        ("Vision OCR",
         [PY, "-c", "from agents.vision_ocr import VisionOCR; print('ok')"]),
        ("Z3 (formal logic)",
         [PY, "-c", "import z3; print('ok')"]),
        ("SymPy (exact math)",
         [PY, "-c", "import sympy; print('ok')"]),
    ]

    for name, cmd in checks:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            status = "  \033[92m✓\033[0m" if "ok" in r.stdout else "  \033[93m⚠\033[0m"
            print(f"{status}  {name}")
            if "ok" not in r.stdout and r.stderr:
                err = r.stderr.strip().split("\n")[-1][:80]
                print(f"     \033[2m{err}\033[0m")
        except Exception as e:
            print(f"  \033[91m✗\033[0m  {name}: {e}")

    print("\n  Run `python install.py` to fix any missing items.\n")


def start_server_visible():
    """
    Start server in the SAME terminal so errors are visible.
    Used when --check is not passed.
    Returns the process.
    """
    if sys.platform == "win32":
        # On Windows, open a new console window so errors show
        proc = subprocess.Popen(
            [PY, str(ROOT / "server.py")],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=str(ROOT),
        )
    else:
        # On Mac/Linux, run in background but pipe to a log file
        log = ROOT / "logs" / "server_run.log"
        log.parent.mkdir(exist_ok=True)
        with open(log, "w") as f:
            proc = subprocess.Popen(
                [PY, str(ROOT / "server.py")],
                stdout=f, stderr=f,
                cwd=str(ROOT),
            )
        print(f"  Server log: {log}")

    return proc


def print_access_info():
    ip = local_ip()
    print()
    print("\033[1m  ARIA is running\033[0m")
    print(f"  \033[96mhttp://localhost:8000\033[0m              ← main UI")
    print(f"  \033[96mhttp://{ip}:8000\033[0m  ← from phone/tablet (same WiFi)")
    print(f"  \033[96mhttp://localhost:8000/nova\033[0m         ← NOVA engine")
    print(f"  \033[96mhttp://localhost:8000/learn\033[0m        ← learning dashboard")
    print(f"  \033[96mhttp://localhost:8000/docs\033[0m         ← API reference")
    print()
    print("  Press \033[1mCtrl+C\033[0m to stop")
    print()


def main():
    parser = argparse.ArgumentParser(description="ARIA Launcher")
    parser.add_argument("--tray",  action="store_true", help="System tray mode")
    parser.add_argument("--app",   action="store_true", help="Desktop app mode")
    parser.add_argument("--check", action="store_true", help="Check install only")
    args = parser.parse_args()

    if args.check:
        check_install()
        return

    print("\n\033[1;35m  ARIA\033[0m \033[2mAdaptive Reasoning Intelligence\033[0m\n")

    # ── Tray / App modes ──────────────────────────────────────────────────────
    if args.tray or args.app:
        flag = "--ui" if args.app else ""
        cmd  = [PY, str(ROOT / "aria_service.py")]
        if flag:
            cmd.append(flag)
        proc = subprocess.Popen(cmd, cwd=str(ROOT))

        if wait_for_server(timeout=45):
            print_access_info()
            if args.app:
                print("  Desktop app should have opened.")
            else:
                print("  Running in system tray — right-click the tray icon for options.")
                print("  Hotkeys: Alt+Space=open, Alt+Shift+S=screen OCR, Alt+Shift+A=clipboard")
        else:
            print("\n\033[91m  Server did not start in time.\033[0m")
            _show_debug_help()
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\n  ARIA stopped.")
        return

    # ── Default: server + browser ─────────────────────────────────────────────
    print("  Starting server...")
    proc = start_server_visible()

    if wait_for_server(timeout=45):
        print_access_info()
        webbrowser.open("http://localhost:8000")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\n  ARIA stopped.")
    else:
        print("\n\033[91m  Server did not start in 45 seconds.\033[0m")
        _show_debug_help()
        proc.terminate()


def _show_debug_help():
    log = ROOT / "logs" / "server_run.log"
    print("\n\033[1m  How to see the real error:\033[0m")
    print(f"  Run this in your terminal:\n")
    print(f"  \033[96m  {PY} server.py\033[0m\n")
    print(f"  It will print the exact error message.")
    if log.exists():
        print(f"\n  Or check the log:\n  \033[96m  {log}\033[0m\n")
        try:
            lines = log.read_text().strip().split("\n")
            last  = [l for l in lines if l.strip()][-10:]
            print("\033[2m  Last log lines:\033[0m")
            for l in last:
                print(f"    {l}")
        except Exception:
            pass
    print()
    print("  Common fixes:")
    print("  \033[96m  python run.py --check\033[0m    ← see what's not installed")
    print("  \033[96m  python install.py --repair\033[0m  ← fix missing packages")
    print()


if __name__ == "__main__":
    main()
