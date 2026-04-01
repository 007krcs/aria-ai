"""
ARIA Service Entry Point
=========================
Run this to start ARIA as a background service with system tray.

    python aria_service.py          # tray + server + hotkeys
    python aria_service.py --ui     # also open the desktop app
    python aria_service.py --server # server only (no tray)

This is the file that auto-start registers on Windows/Mac/Linux.
"""

import sys
import os
import time
import threading
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from system.service import (
    ServerManager, NotificationManager, ScreenAssistant,
    HotkeyManager, AutoStartInstaller, TrayApp
)
from rich.console import Console

console = Console()


def open_ui():
    """Open the Flet desktop app."""
    try:
        import subprocess
        subprocess.Popen(
            [sys.executable, str(PROJECT_ROOT / "app" / "aria_app.py")],
            cwd=str(PROJECT_ROOT),
        )
    except Exception as e:
        # Fallback: open browser
        import webbrowser
        webbrowser.open("http://localhost:8000")


def main():
    parser = argparse.ArgumentParser(description="ARIA Service")
    parser.add_argument("--ui",     action="store_true", help="Open desktop UI on start")
    parser.add_argument("--server", action="store_true", help="Server only, no tray")
    parser.add_argument("--install", action="store_true", help="Install auto-start")
    parser.add_argument("--uninstall", action="store_true", help="Remove auto-start")
    args = parser.parse_args()

    # Init components
    server    = ServerManager()
    notifier  = NotificationManager()
    screen    = ScreenAssistant(notifier)
    autostart = AutoStartInstaller()

    # Handle install/uninstall
    if args.install:
        result = autostart.install()
        console.print(f"Auto-start: {result}")
        return
    if args.uninstall:
        result = autostart.uninstall()
        console.print(f"Auto-start removed: {result}")
        return

    # Start server
    console.print("[bold blue]Starting ARIA...[/]")
    started = server.start()
    if not started:
        console.print("[red]Could not start server. Check logs/service.log[/]")

    # Open UI if requested
    if args.ui:
        threading.Timer(2.0, open_ui).start()

    # Server-only mode
    if args.server:
        console.print("[green]ARIA server running[/] — Ctrl+C to stop")
        try:
            while True:
                time.sleep(10)
                if not server.is_running():
                    console.print("[yellow]Server went down — restarting...[/]")
                    server.start()
        except KeyboardInterrupt:
            server.stop()
            console.print("[dim]ARIA stopped.[/]")
        return

    # Start hotkeys
    hotkeys = HotkeyManager(server, notifier, screen, on_open_ui=open_ui)
    threading.Thread(target=hotkeys.start, daemon=True).start()

    # Start system tray (blocks until quit)
    console.print("[green]ARIA running in system tray[/]")
    console.print("  Alt+Space       = open ARIA")
    console.print("  Alt+Shift+S     = screen OCR")
    console.print("  Alt+Shift+A     = analyse clipboard")
    console.print("  Right-click tray icon for more options")

    tray = TrayApp(server, notifier, autostart, on_open_ui=open_ui)
    tray.start()


if __name__ == "__main__":
    main()
