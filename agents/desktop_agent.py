"""
ARIA Desktop Agent — Full Windows Desktop Control
===================================================
Gives ARIA the ability to:
  - Open / close / switch any application
  - Read what's on screen (OCR)
  - Type text, click, scroll, hotkeys
  - Read / write / search files
  - Run shell commands
  - List running apps and windows
  - Take screenshots

Called by OmegaOrchestrator when intent is desktop/system control.
Also exposed via /api/desktop/* endpoints for direct use.
"""

import os
import re
import time
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_import(name):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


# ── APP NAME → EXECUTABLE MAP ─────────────────────────────────────────────────

APP_MAP = {
    # Browsers
    "chrome":       "chrome.exe",
    "google chrome":"chrome.exe",
    "firefox":      "firefox.exe",
    "edge":         "msedge.exe",
    "microsoft edge":"msedge.exe",
    # Productivity
    "notepad":      "notepad.exe",
    "word":         "winword.exe",
    "excel":        "excel.exe",
    "powerpoint":   "powerpnt.exe",
    "outlook":      "outlook.exe",
    "teams":        "teams.exe",
    "onenote":      "onenote.exe",
    # Dev tools
    "vscode":       "code.exe",
    "vs code":      "code.exe",
    "visual studio code": "code.exe",
    "terminal":     "wt.exe",
    "cmd":          "cmd.exe",
    "powershell":   "powershell.exe",
    "git bash":     "git-bash.exe",
    # Media
    "vlc":          "vlc.exe",
    "spotify":      "spotify.exe",
    "photos":       "ms-photos:",
    # System
    "task manager": "taskmgr.exe",
    "calculator":   "calc.exe",
    "paint":        "mspaint.exe",
    "snipping tool":"snippingtool.exe",
    "settings":     "ms-settings:",
    "file explorer":"explorer.exe",
    "explorer":     "explorer.exe",
    "control panel":"control.exe",
    # Communication
    "whatsapp":     "whatsapp.exe",
    "telegram":     "telegram.exe",
    "slack":        "slack.exe",
    "zoom":         "zoom.exe",
    "discord":      "discord.exe",
}


class DesktopAgent:
    """
    Full Windows desktop control for ARIA.
    All methods return a dict: {"ok": bool, "result": str, "data": any}
    """

    def __init__(self):
        self._pyautogui  = _safe_import("pyautogui")
        self._pywinauto  = _safe_import("pywinauto")
        self._psutil     = _safe_import("psutil")
        self._keyboard   = _safe_import("keyboard")
        self._PIL        = _safe_import("PIL.ImageGrab")

        # Safety: pyautogui pause between actions
        if self._pyautogui:
            self._pyautogui.FAILSAFE = True
            self._pyautogui.PAUSE    = 0.3

    # ── OPEN APPLICATION ──────────────────────────────────────────────────────

    def open_app(self, name: str, args: str = "") -> dict:
        """Open an application by name or full path."""
        name_lower = name.lower().strip()

        # 1. Look up known app map
        exe = APP_MAP.get(name_lower)

        # 2. Try direct path
        if not exe and (name.endswith(".exe") or "/" in name or "\\" in name):
            exe = name

        # 3. Search PATH
        if not exe:
            found = shutil.which(name_lower) or shutil.which(name_lower + ".exe")
            if found:
                exe = found

        # 4. Common install locations
        if not exe:
            search_dirs = [
                r"C:\Program Files",
                r"C:\Program Files (x86)",
                os.path.expanduser("~\\AppData\\Local\\Programs"),
                os.path.expanduser("~\\AppData\\Roaming"),
            ]
            for d in search_dirs:
                for root, dirs, files in os.walk(d):
                    for f in files:
                        if f.lower() == name_lower + ".exe" or f.lower() == name_lower:
                            exe = os.path.join(root, f)
                            break
                    if exe:
                        break
                if exe:
                    break

        if not exe:
            return {"ok": False, "result": f"Could not find '{name}'. Try giving the full path or install it first."}

        try:
            cmd = [exe] + (args.split() if args else [])
            subprocess.Popen(cmd, shell=True)
            time.sleep(0.8)
            return {"ok": True, "result": f"Opened {name}"}
        except Exception as e:
            return {"ok": False, "result": f"Failed to open {name}: {e}"}

    # ── CLOSE APPLICATION ─────────────────────────────────────────────────────

    def close_app(self, name: str) -> dict:
        """Close an application by name."""
        if not self._psutil:
            return {"ok": False, "result": "psutil not available"}

        name_lower = name.lower().replace(".exe", "")
        killed = []

        for proc in self._psutil.process_iter(["pid", "name"]):
            try:
                pname = proc.info["name"].lower().replace(".exe", "")
                if name_lower in pname or pname in name_lower:
                    proc.terminate()
                    killed.append(proc.info["name"])
            except Exception:
                pass

        if killed:
            return {"ok": True, "result": f"Closed: {', '.join(set(killed))}"}
        return {"ok": False, "result": f"No running process found matching '{name}'"}

    # ── LIST RUNNING APPS ─────────────────────────────────────────────────────

    def list_apps(self) -> dict:
        """List all running applications (with visible windows)."""
        try:
            import win32gui, win32process
            windows = []
            def enum_cb(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title.strip():
                        windows.append(title)
            win32gui.EnumWindows(enum_cb, None)
            unique = sorted(set(windows))
            return {"ok": True, "result": "\n".join(unique[:30]), "data": unique}
        except Exception:
            # Fallback: psutil
            if self._psutil:
                procs = sorted(set(
                    p.name() for p in self._psutil.process_iter(["name"])
                    if p.name().endswith(".exe")
                ))
                return {"ok": True, "result": "\n".join(procs[:30]), "data": procs}
            return {"ok": False, "result": "Cannot list apps"}

    # ── FOCUS / SWITCH WINDOW ─────────────────────────────────────────────────

    def focus_window(self, title: str) -> dict:
        """Bring a window to focus by partial title match."""
        try:
            import pygetwindow as gw
            wins = gw.getWindowsWithTitle(title)
            if not wins:
                # Partial match
                all_wins = gw.getAllTitles()
                matched = [w for w in all_wins if title.lower() in w.lower()]
                if matched:
                    wins = gw.getWindowsWithTitle(matched[0])
            if wins:
                w = wins[0]
                w.restore()
                w.activate()
                return {"ok": True, "result": f"Focused: {w.title}"}
            return {"ok": False, "result": f"No window with title '{title}'"}
        except Exception as e:
            return {"ok": False, "result": f"Focus failed: {e}"}

    # ── TAKE SCREENSHOT ───────────────────────────────────────────────────────

    def screenshot(self, save_path: str = "") -> dict:
        """Take a screenshot and optionally save it."""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            if save_path:
                img.save(save_path)
                return {"ok": True, "result": f"Screenshot saved to {save_path}", "data": save_path}
            else:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(os.path.expanduser("~"), "Desktop", f"aria_screenshot_{ts}.png")
                img.save(path)
                return {"ok": True, "result": f"Screenshot saved: {path}", "data": path}
        except Exception as e:
            return {"ok": False, "result": f"Screenshot failed: {e}"}

    # ── READ SCREEN (OCR) ─────────────────────────────────────────────────────

    def read_screen(self, region=None) -> dict:
        """OCR the current screen and return the text."""
        try:
            from PIL import ImageGrab
            import pytesseract

            # Try to find tesseract
            possible = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                shutil.which("tesseract"),
            ]
            for p in possible:
                if p and os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break

            img  = ImageGrab.grab(bbox=region)
            text = pytesseract.image_to_string(img)
            text = text.strip()
            if not text:
                return {"ok": True, "result": "Screen appears to have no readable text (graphical content)."}
            return {"ok": True, "result": text[:3000], "data": text}
        except Exception as e:
            return {"ok": False, "result": f"Screen read failed: {e}. Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"}

    # ── TYPE TEXT ─────────────────────────────────────────────────────────────

    def type_text(self, text: str, interval: float = 0.03) -> dict:
        """Type text at the current cursor position."""
        if not self._pyautogui:
            return {"ok": False, "result": "pyautogui not available"}
        try:
            self._pyautogui.typewrite(text, interval=interval)
            return {"ok": True, "result": f"Typed: {text[:50]}{'...' if len(text)>50 else ''}"}
        except Exception as e:
            # Fallback: pyperclip paste for unicode
            try:
                import pyperclip
                pyperclip.copy(text)
                self._pyautogui.hotkey("ctrl", "v")
                return {"ok": True, "result": f"Pasted: {text[:50]}"}
            except Exception:
                return {"ok": False, "result": f"Type failed: {e}"}

    # ── CLICK ─────────────────────────────────────────────────────────────────

    def click(self, x: int = None, y: int = None, button: str = "left",
              image: str = None) -> dict:
        """Click at coordinates or find and click an image on screen."""
        if not self._pyautogui:
            return {"ok": False, "result": "pyautogui not available"}
        try:
            if image:
                loc = self._pyautogui.locateCenterOnScreen(image, confidence=0.8)
                if loc:
                    self._pyautogui.click(loc, button=button)
                    return {"ok": True, "result": f"Clicked image at {loc}"}
                return {"ok": False, "result": f"Image '{image}' not found on screen"}
            elif x is not None and y is not None:
                self._pyautogui.click(x, y, button=button)
                return {"ok": True, "result": f"Clicked ({x}, {y})"}
            else:
                self._pyautogui.click(button=button)
                return {"ok": True, "result": "Clicked at current position"}
        except Exception as e:
            return {"ok": False, "result": f"Click failed: {e}"}

    # ── HOTKEY ────────────────────────────────────────────────────────────────

    def hotkey(self, *keys) -> dict:
        """Press a keyboard shortcut (e.g. 'ctrl', 'c')."""
        if not self._pyautogui:
            return {"ok": False, "result": "pyautogui not available"}
        try:
            self._pyautogui.hotkey(*keys)
            return {"ok": True, "result": f"Pressed: {'+'.join(keys)}"}
        except Exception as e:
            return {"ok": False, "result": f"Hotkey failed: {e}"}

    # ── SCROLL ────────────────────────────────────────────────────────────────

    def scroll(self, clicks: int = 3, direction: str = "down") -> dict:
        """Scroll the mouse wheel."""
        if not self._pyautogui:
            return {"ok": False, "result": "pyautogui not available"}
        try:
            amount = -clicks if direction == "down" else clicks
            self._pyautogui.scroll(amount)
            return {"ok": True, "result": f"Scrolled {direction} {abs(clicks)} clicks"}
        except Exception as e:
            return {"ok": False, "result": f"Scroll failed: {e}"}

    # ── FILE OPERATIONS ───────────────────────────────────────────────────────

    def read_file(self, path: str, max_chars: int = 8000) -> dict:
        """Read content of any text file."""
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return {"ok": False, "result": f"File not found: {path}"}
            if p.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                return {"ok": False, "result": "File too large (>10MB). Open it in an editor."}
            text = p.read_text(encoding="utf-8", errors="replace")
            snippet = text[:max_chars]
            note = f"\n\n[Showing first {max_chars} chars of {len(text)} total]" if len(text) > max_chars else ""
            return {"ok": True, "result": snippet + note, "data": text}
        except Exception as e:
            return {"ok": False, "result": f"Read failed: {e}"}

    def write_file(self, path: str, content: str, mode: str = "w") -> dict:
        """Write content to a file (mode: 'w' overwrite, 'a' append)."""
        try:
            p = Path(path).expanduser()
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            return {"ok": True, "result": f"Written {len(content)} chars to {path}"}
        except Exception as e:
            return {"ok": False, "result": f"Write failed: {e}"}

    def search_files(self, query: str, directory: str = "~", ext: str = "") -> dict:
        """Search for files by name."""
        try:
            base = Path(directory).expanduser()
            results = []
            pattern = f"*{query}*" + (f".{ext}" if ext else "")
            for p in base.rglob(pattern):
                results.append(str(p))
                if len(results) >= 20:
                    break
            if results:
                return {"ok": True, "result": "\n".join(results), "data": results}
            return {"ok": True, "result": f"No files matching '{query}' found in {directory}"}
        except Exception as e:
            return {"ok": False, "result": f"Search failed: {e}"}

    def list_directory(self, path: str = "~") -> dict:
        """List files and folders in a directory."""
        try:
            p = Path(path).expanduser()
            if not p.exists():
                return {"ok": False, "result": f"Directory not found: {path}"}
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            lines = []
            for item in items[:50]:
                prefix = "📁 " if item.is_dir() else "📄 "
                size   = f"  ({item.stat().st_size // 1024}KB)" if item.is_file() else ""
                lines.append(f"{prefix}{item.name}{size}")
            return {"ok": True, "result": "\n".join(lines), "data": [str(i) for i in items]}
        except Exception as e:
            return {"ok": False, "result": f"List failed: {e}"}

    # ── RUN COMMAND ───────────────────────────────────────────────────────────

    def run_command(self, cmd: str, timeout: int = 30, shell: bool = True) -> dict:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                cmd, shell=shell, capture_output=True, text=True,
                timeout=timeout, encoding="utf-8", errors="replace"
            )
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            combined = out + ("\n\nSTDERR:\n" + err if err else "")
            return {
                "ok":     result.returncode == 0,
                "result": combined[:3000] or "(no output)",
                "code":   result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "result": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"ok": False, "result": f"Command failed: {e}"}

    # ── NATURAL LANGUAGE DISPATCHER ───────────────────────────────────────────

    def execute_nl(self, instruction: str) -> dict:
        """
        Parse a natural language desktop instruction and execute it.
        e.g. "open Chrome", "close Notepad", "type hello world",
             "read my Desktop/notes.txt", "list files in Downloads"
        """
        q = instruction.lower().strip()

        # Open app
        m = re.match(r"open\s+(.+)", q)
        if m:
            return self.open_app(m.group(1).strip())

        # Close app
        m = re.match(r"close\s+(.+)|kill\s+(.+)|stop\s+(.+)", q)
        if m:
            name = (m.group(1) or m.group(2) or m.group(3)).strip()
            return self.close_app(name)

        # List apps / windows
        if any(x in q for x in ["list apps", "running apps", "open apps", "what apps", "list windows"]):
            return self.list_apps()

        # Screenshot
        if any(x in q for x in ["screenshot", "screen shot", "capture screen"]):
            return self.screenshot()

        # Read screen
        if any(x in q for x in ["read screen", "what's on screen", "what is on screen", "ocr"]):
            return self.read_screen()

        # Type text
        m = re.match(r"type\s+(.+)", q)
        if m:
            return self.type_text(m.group(1).strip())

        # Hotkey
        m = re.match(r"press\s+(.+)|hotkey\s+(.+)", q)
        if m:
            keys_str = (m.group(1) or m.group(2)).strip()
            keys = [k.strip() for k in re.split(r"[+\s]+", keys_str)]
            return self.hotkey(*keys)

        # Read file
        m = re.match(r"read\s+(?:file\s+)?(.+)", q)
        if m:
            path = m.group(1).strip()
            if any(c in path for c in ["/", "\\", "."]):
                return self.read_file(path)

        # List directory
        m = re.match(r"list\s+(?:files?\s+(?:in\s+)?)?(.+)|show\s+files?\s+in\s+(.+)", q)
        if m:
            path = (m.group(1) or m.group(2)).strip()
            return self.list_directory(path)

        # Run command
        m = re.match(r"run\s+(?:command\s+)?(.+)|execute\s+(.+)", q)
        if m:
            cmd = (m.group(1) or m.group(2)).strip()
            return self.run_command(cmd)

        # Focus window
        m = re.match(r"(?:focus|switch to|go to)\s+(.+)", q)
        if m:
            return self.focus_window(m.group(1).strip())

        return {
            "ok": False,
            "result": f"I understood '{instruction}' as a desktop action but couldn't match a specific operation. "
                      "Try: 'open Chrome', 'close Notepad', 'read ~/Desktop/file.txt', 'list files in Downloads', "
                      "'take screenshot', 'type hello world', 'press ctrl+c'"
        }
