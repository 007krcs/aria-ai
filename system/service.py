"""
ARIA System Service
===================
Runs ARIA as a background system service.

Features:
- System tray icon with quick actions
- Global hotkey (Alt+Space) to open ARIA anywhere
- Desktop notifications for important events
- Auto-start on system login
- Screen capture + vision OCR on demand
- Clipboard monitoring
- Background server management

Cross-platform: Windows, Mac, Linux

Install dependencies:
    pip install pystray pillow plyer keyboard pyautogui
"""

import sys
import os
import time
import json
import subprocess
import threading
import platform
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLATFORM     = platform.system()   # Windows / Darwin / Linux
PID_FILE     = PROJECT_ROOT / "aria_server.pid"
LOG_FILE     = PROJECT_ROOT / "logs" / "service.log"
LOG_FILE.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SERVER PROCESS MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class ServerManager:
    """Start/stop/monitor the ARIA FastAPI server process."""

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        if self.is_running():
            return True
        try:
            log_fd = open(LOG_FILE, "a")
            self._process = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "server.py")],
                cwd=str(PROJECT_ROOT),
                stdout=log_fd,
                stderr=log_fd,
            )
            PID_FILE.write_text(str(self._process.pid))
            time.sleep(3)  # wait for server to start
            console.print(f"  [green]ARIA server started[/] (PID {self._process.pid})")
            return True
        except Exception as e:
            console.print(f"  [red]Server start failed: {e}[/]")
            return False

    def stop(self):
        if self._process:
            self._process.terminate()
            self._process = None
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text())
                if PLATFORM == "Windows":
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                else:
                    os.kill(pid, 15)
            except Exception:
                pass
            PID_FILE.unlink(missing_ok=True)

    def is_running(self) -> bool:
        try:
            import requests
            r = requests.get("http://localhost:8000/api/health", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def restart(self):
        self.stop()
        time.sleep(1)
        self.start()


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICATION MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class NotificationManager:
    """Send desktop notifications cross-platform."""

    def notify(self, title: str, message: str, timeout: int = 5):
        try:
            from plyer import notification
            notification.notify(
                title=f"ARIA — {title}",
                message=message[:200],
                app_name="ARIA",
                timeout=timeout,
            )
        except ImportError:
            # Fallback per platform
            if PLATFORM == "Darwin":
                subprocess.run([
                    "osascript", "-e",
                    f'display notification "{message}" with title "ARIA — {title}"'
                ], capture_output=True)
            elif PLATFORM == "Linux":
                subprocess.run(["notify-send", f"ARIA — {title}", message], capture_output=True)
            elif PLATFORM == "Windows":
                # Windows toast via PowerShell
                ps = (
                    f'$n=[Windows.UI.Notifications.ToastNotificationManager,Windows.UI.Notifications,ContentType=WindowsRuntime];'
                    f'$t=$n::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);'
                    f'$t.GetElementsByTagName("text")[0].AppendChild($t.CreateTextNode("{title}"));'
                    f'$t.GetElementsByTagName("text")[1].AppendChild($t.CreateTextNode("{message}"));'
                    f'$n::CreateToastNotifier("ARIA").Show([Windows.UI.Notifications.ToastNotification]::new($t))'
                )
                subprocess.run(["powershell", "-Command", ps], capture_output=True)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# SCREEN CAPTURE + VISION OCR
# ─────────────────────────────────────────────────────────────────────────────

class ScreenAssistant:
    """
    Captures the screen and sends to ARIA's vision OCR.
    Triggered by hotkey or tray menu.
    """

    def __init__(self, notifier: NotificationManager):
        self.notifier = notifier

    def capture_and_ask(self, question: str = "What is on my screen? Summarise it.") -> str:
        """Capture screen, ask ARIA's vision model about it."""
        try:
            import pyautogui
            from PIL import Image
            import io, base64

            screenshot = pyautogui.screenshot()
            buf        = io.BytesIO()
            screenshot.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            # Ask ARIA
            import requests
            r = requests.post(
                "http://localhost:8000/api/vision/ask",
                json={"image_b64": b64, "question": question},
                timeout=30,
            )
            if r.status_code == 200:
                answer = r.json().get("answer", "")
                self.notifier.notify("Screen Analysis", answer[:100])
                return answer
        except ImportError:
            return "pip install pyautogui pillow for screen capture"
        except Exception as e:
            return f"Screen capture error: {e}"
        return ""

    def capture_region(self, x: int, y: int, w: int, h: int) -> str:
        """Capture a specific screen region and extract text."""
        try:
            import pyautogui
            import io, base64
            screenshot = pyautogui.screenshot(region=(x, y, w, h))
            buf        = io.BytesIO()
            screenshot.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            import requests
            r = requests.post(
                "http://localhost:8000/api/vision/ocr",
                json={"image_b64": b64},
                timeout=30,
            )
            return r.json().get("text", "") if r.status_code == 200 else ""
        except Exception as e:
            return f"Region capture error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL HOTKEY MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class HotkeyManager:
    """
    Register global hotkeys that work even when ARIA is in background.
    Alt+Space  — open ARIA quick ask
    Alt+Shift+S — screenshot OCR
    Alt+Shift+A — read clipboard
    """

    def __init__(self, server: ServerManager, notifier: NotificationManager,
                 screen: ScreenAssistant, on_open_ui: callable = None):
        self.server    = server
        self.notifier  = notifier
        self.screen    = screen
        self.on_open_ui = on_open_ui
        self._active   = False

    def start(self):
        self._active = True
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()

    def stop(self):
        self._active = False

    def _listen(self):
        try:
            import keyboard
            keyboard.add_hotkey("alt+space",          self._open_quick_ask)
            keyboard.add_hotkey("alt+shift+s",        self._screen_ocr)
            keyboard.add_hotkey("alt+shift+a",        self._ask_clipboard)
            keyboard.add_hotkey("alt+shift+q",        self._quick_question)
            console.print("  [green]Hotkeys registered:[/] Alt+Space=open, Alt+Shift+S=screen, Alt+Shift+A=clipboard")
            keyboard.wait()
        except ImportError:
            console.print("  [dim]pip install keyboard for global hotkeys[/]")
        except Exception as e:
            console.print(f"  [yellow]Hotkey error: {e}[/]")

    def _open_quick_ask(self):
        if self.on_open_ui:
            threading.Thread(target=self.on_open_ui, daemon=True).start()

    def _screen_ocr(self):
        self.notifier.notify("Screen OCR", "Analysing screen...")
        result = self.screen.capture_and_ask("Extract all text from this screen")
        self.notifier.notify("Screen OCR Result", result[:150] if result else "No text found")

    def _ask_clipboard(self):
        try:
            import pyperclip
            text = pyperclip.paste()
            if not text:
                return
            import requests
            r = requests.post(
                "http://localhost:8000/api/chat/stream",
                json={"message": f"Analyse this: {text[:500]}"},
                timeout=60,
            )
            self.notifier.notify("Clipboard Analysis", "Done — check ARIA")
        except Exception as e:
            self.notifier.notify("Error", str(e)[:80])

    def _quick_question(self):
        # Shows a simple input dialog
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            q = simpledialog.askstring("ARIA Quick Ask", "Ask ARIA:")
            root.destroy()
            if q:
                import requests
                r = requests.post(
                    "http://localhost:8000/api/chat/stream",
                    json={"message": q},
                    timeout=60,
                )
                answer = ""
                for line in r.iter_lines():
                    if line and line.startswith(b"data: "):
                        d = json.loads(line[6:])
                        if d.get("type") == "token":
                            answer += d["text"]
                self.notifier.notify("ARIA", answer[:150])
        except Exception as e:
            self.notifier.notify("Error", str(e)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-START INSTALLER
# ─────────────────────────────────────────────────────────────────────────────

class AutoStartInstaller:
    """
    Installs ARIA to run automatically at system login.
    No admin rights needed on Windows and Mac.
    Linux requires user systemd service.
    """

    def __init__(self):
        self.script = PROJECT_ROOT / "aria_service.py"

    def install(self) -> dict:
        """Install ARIA to auto-start on login."""
        try:
            if PLATFORM == "Windows":
                return self._install_windows()
            elif PLATFORM == "Darwin":
                return self._install_macos()
            else:
                return self._install_linux()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def uninstall(self) -> dict:
        """Remove ARIA from auto-start."""
        try:
            if PLATFORM == "Windows":
                return self._uninstall_windows()
            elif PLATFORM == "Darwin":
                return self._uninstall_macos()
            else:
                return self._uninstall_linux()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def is_installed(self) -> bool:
        try:
            if PLATFORM == "Windows":
                r = subprocess.run(
                    ["reg", "query", r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run", "/v", "ARIA"],
                    capture_output=True
                )
                return r.returncode == 0
            elif PLATFORM == "Darwin":
                plist = Path.home() / "Library/LaunchAgents/com.aria.service.plist"
                return plist.exists()
            else:
                service = Path.home() / ".config/systemd/user/aria.service"
                return service.exists()
        except Exception:
            return False

    def _install_windows(self) -> dict:
        """Add to Windows registry Run key — no admin needed."""
        import winreg
        cmd = f'"{sys.executable}" "{PROJECT_ROOT / "aria_service.py"}"'
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "ARIA", 0, winreg.REG_SZ, cmd)
        winreg.CloseKey(key)
        return {"success": True, "method": "Windows Registry Run key", "command": cmd}

    def _uninstall_windows(self) -> dict:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.DeleteValue(key, "ARIA")
        winreg.CloseKey(key)
        return {"success": True}

    def _install_macos(self) -> dict:
        """Install as macOS LaunchAgent — runs at login, no admin needed."""
        plist_dir  = Path.home() / "Library/LaunchAgents"
        plist_dir.mkdir(exist_ok=True)
        plist_path = plist_dir / "com.aria.service.plist"
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.aria.service</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{PROJECT_ROOT / "aria_service.py"}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>{LOG_FILE}</string>
</dict>
</plist>"""
        plist_path.write_text(plist_content)
        subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True)
        return {"success": True, "method": "macOS LaunchAgent", "plist": str(plist_path)}

    def _uninstall_macos(self) -> dict:
        plist_path = Path.home() / "Library/LaunchAgents/com.aria.service.plist"
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            plist_path.unlink()
        return {"success": True}

    def _install_linux(self) -> dict:
        """Install as systemd user service — no root needed."""
        service_dir = Path.home() / ".config/systemd/user"
        service_dir.mkdir(parents=True, exist_ok=True)
        service_path = service_dir / "aria.service"
        service_content = f"""[Unit]
Description=ARIA Personal AI Assistant
After=network.target

[Service]
ExecStart={sys.executable} {PROJECT_ROOT / "aria_service.py"}
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
"""
        service_path.write_text(service_content)
        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
        subprocess.run(["systemctl", "--user", "enable", "aria"], capture_output=True)
        subprocess.run(["systemctl", "--user", "start",  "aria"], capture_output=True)
        return {"success": True, "method": "systemd user service", "service": str(service_path)}

    def _uninstall_linux(self) -> dict:
        subprocess.run(["systemctl", "--user", "stop",    "aria"], capture_output=True)
        subprocess.run(["systemctl", "--user", "disable", "aria"], capture_output=True)
        service = Path.home() / ".config/systemd/user/aria.service"
        if service.exists():
            service.unlink()
        return {"success": True}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM TRAY APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class TrayApp:
    """
    System tray icon with full context menu.
    Runs in background — click to open ARIA, right-click for options.
    """

    def __init__(self, server: ServerManager, notifier: NotificationManager,
                 autostart: AutoStartInstaller, on_open_ui: callable = None):
        self.server    = server
        self.notifier  = notifier
        self.autostart = autostart
        self.on_open_ui = on_open_ui
        self._tray     = None

    def start(self):
        try:
            import pystray
            from PIL import Image, ImageDraw

            icon_img = self._create_icon()
            menu     = pystray.Menu(
                pystray.MenuItem("Open ARIA",          self._open_ui, default=True),
                pystray.MenuItem("Quick Ask",          self._quick_ask),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Scan Screen",        self._scan_screen),
                pystray.MenuItem("Scan Clipboard",     self._scan_clipboard),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Restart Server",     self._restart_server),
                pystray.MenuItem(
                    "Run at Login",
                    self._toggle_autostart,
                    checked=lambda item: self.autostart.is_installed(),
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit ARIA",          self._quit),
            )
            self._tray = pystray.Icon("ARIA", icon_img, "ARIA Personal AI", menu)
            self.notifier.notify("ARIA Started", "Running in background. Alt+Space to open.")
            self._tray.run()

        except ImportError:
            console.print("  [dim]pip install pystray pillow for system tray[/]")
            # Fallback: just keep server running
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                pass

    def _create_icon(self):
        """Create a simple ARIA icon programmatically."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            img  = Image.new("RGB", (64, 64), color=(18, 18, 28))
            draw = ImageDraw.Draw(img)
            # Draw "A" letter with purple accent
            draw.ellipse([4, 4, 60, 60], fill=(30, 30, 50), outline=(124, 106, 247), width=2)
            draw.text((18, 16), "AR", fill=(124, 106, 247))
            draw.text((18, 34), "IA", fill=(61, 214, 140))
            return img
        except Exception:
            from PIL import Image
            return Image.new("RGB", (64, 64), color=(124, 106, 247))

    def _open_ui(self, icon=None, item=None):
        if self.on_open_ui:
            threading.Thread(target=self.on_open_ui, daemon=True).start()
        else:
            import webbrowser
            webbrowser.open("http://localhost:8000")

    def _quick_ask(self, icon=None, item=None):
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.lift()
            q = simpledialog.askstring("ARIA", "Ask anything:", parent=root)
            root.destroy()
            if q:
                import requests
                full_answer = ""
                r = requests.post(
                    "http://localhost:8000/api/chat/stream",
                    json={"message": q}, stream=True, timeout=60
                )
                for line in r.iter_lines():
                    if line and line.startswith(b"data: "):
                        d = json.loads(line[6:])
                        if d.get("type") == "token":
                            full_answer += d["text"]
                self.notifier.notify("ARIA Answer", full_answer[:200])
        except Exception as e:
            self.notifier.notify("Error", str(e)[:80])

    def _scan_screen(self, icon=None, item=None):
        self.notifier.notify("ARIA", "Scanning screen...")
        try:
            import pyautogui, io, base64, requests
            shot = pyautogui.screenshot()
            buf  = io.BytesIO()
            shot.save(buf, "PNG")
            b64  = base64.b64encode(buf.getvalue()).decode()
            r    = requests.post("http://localhost:8000/api/vision/ocr",
                                 json={"image_b64": b64}, timeout=45)
            text = r.json().get("text", "")[:200]
            self.notifier.notify("Screen Text", text or "No text detected")
        except Exception as e:
            self.notifier.notify("Error", str(e)[:80])

    def _scan_clipboard(self, icon=None, item=None):
        try:
            import pyperclip
            text = pyperclip.paste()
            if not text.strip():
                self.notifier.notify("ARIA", "Clipboard is empty")
                return
            import requests
            r = requests.post(
                "http://localhost:8000/api/chat/stream",
                json={"message": f"Analyse and summarise: {text[:1000]}"}, timeout=60
            )
            answer = ""
            for line in r.iter_lines():
                if line and line.startswith(b"data: "):
                    d = json.loads(line[6:])
                    if d.get("type") == "token":
                        answer += d["text"]
            self.notifier.notify("Clipboard Analysis", answer[:200])
        except Exception as e:
            self.notifier.notify("Error", str(e)[:80])

    def _restart_server(self, icon=None, item=None):
        self.notifier.notify("ARIA", "Restarting server...")
        threading.Thread(target=self.server.restart, daemon=True).start()

    def _toggle_autostart(self, icon=None, item=None):
        if self.autostart.is_installed():
            self.autostart.uninstall()
            self.notifier.notify("ARIA", "Removed from startup")
        else:
            self.autostart.install()
            self.notifier.notify("ARIA", "Added to startup — runs at login")

    def _quit(self, icon=None, item=None):
        self.server.stop()
        if self._tray:
            self._tray.stop()
        sys.exit(0)
