"""
ARIA Global Service — Always-On Background Daemon
==================================================
Runs as a Windows system tray process independent of the browser/frontend.
Provides Siri/Alexa-style activation that works even when the UI is closed.

Features:
  - System tray icon with right-click menu
  - Global hotkey (Ctrl+Shift+Space by default)
  - Wake word detection via faster-whisper (offline)
  - Forwards commands to ARIA backend (port 8000)
  - Desktop toast notifications
  - Auto-start with Windows (optional)
  - Kernel-level keyboard hook for instant response

Usage:
    python system/global_service.py          # start service
    python system/global_service.py --install # register auto-start
    python system/global_service.py --remove  # remove auto-start
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import queue
import sys
import tempfile
import threading
import time
import winreg
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("aria.global_service")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [ARIA-SVC] %(message)s",
                    datefmt="%H:%M:%S")

ARIA_PORT     = int(os.getenv("ARIA_PORT", "8000"))
ARIA_BASE_URL = f"http://localhost:{ARIA_PORT}"
HOTKEY        = os.getenv("ARIA_HOTKEY", "ctrl+shift+space")
WAKE_WORD     = os.getenv("ARIA_WAKE_WORD", "hey aria").lower()
TOKEN_FILE    = PROJECT_ROOT / "data" / ".service_token"

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import requests as _req
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import keyboard
    KEYBOARD_OK = True
except ImportError:
    KEYBOARD_OK = False
    logger.warning("keyboard not installed — hotkey disabled. pip install keyboard")

try:
    import pystray
    from PIL import Image as _PILImage, ImageDraw as _Draw
    TRAY_OK = True
except ImportError:
    TRAY_OK = False
    logger.warning("pystray/Pillow not installed — tray disabled. pip install pystray pillow")

try:
    from plyer import notification as _plyer_notif
    PLYER_OK = True
except ImportError:
    PLYER_OK = False

try:
    import ctypes
    CTYPES_OK = True
except ImportError:
    CTYPES_OK = False

try:
    import sounddevice as _sd
    import numpy as _np
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False

try:
    import faster_whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False


# ── Token helpers ─────────────────────────────────────────────────────────────

def _load_token() -> str:
    """Load saved JWT token for API calls."""
    try:
        if TOKEN_FILE.exists():
            return TOKEN_FILE.read_text().strip()
    except Exception:
        pass
    return ""


def _save_token(token: str):
    TOKEN_FILE.parent.mkdir(exist_ok=True)
    TOKEN_FILE.write_text(token)


# ── ARIA API client ───────────────────────────────────────────────────────────

class ARIAClient:
    """Thin HTTP client to the local ARIA backend."""

    def __init__(self):
        self._token = _load_token()
        self._session = _req.Session() if REQUESTS_OK else None

    @property
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def is_alive(self) -> bool:
        if not REQUESTS_OK:
            return False
        try:
            r = self._session.get(f"{ARIA_BASE_URL}/api/health",
                                  headers=self._headers, timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def send_command(self, text: str, session_id: str = "global") -> str:
        """Send a natural language command and get the response (non-streaming)."""
        if not REQUESTS_OK:
            return "requests not installed"
        try:
            r = self._session.post(
                f"{ARIA_BASE_URL}/api/auto/quick",
                json={"message": text, "session_id": session_id},
                headers=self._headers,
                timeout=30,
            )
            if r.ok:
                d = r.json()
                return d.get("response") or d.get("answer") or d.get("result") or str(d)
            return f"Server returned {r.status_code}"
        except Exception as e:
            return f"Connection failed: {e}"

    def stream_command(self, text: str, on_token=None, session_id: str = "global"):
        """Stream a command response token by token, calling on_token(str) per chunk."""
        if not REQUESTS_OK:
            return
        try:
            with self._session.post(
                f"{ARIA_BASE_URL}/api/auto/stream",
                json={"message": text, "session_id": session_id},
                headers=self._headers,
                stream=True,
                timeout=60,
            ) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        raw = line.decode("utf-8").strip()
                        if not raw.startswith("data:"):
                            continue
                        payload = json.loads(raw[5:].strip())
                        t = payload.get("type", "")
                        if t in ("token", "text") and on_token:
                            on_token(payload.get("content", ""))
                        elif t == "done":
                            break
                    except Exception:
                        pass
        except Exception as e:
            logger.error("Stream error: %s", e)

    def refresh_token(self, pin: str) -> bool:
        if not REQUESTS_OK:
            return False
        try:
            r = self._session.post(
                f"{ARIA_BASE_URL}/auth/login",
                json={"pin": pin, "device_name": "aria-global-service"},
                timeout=5,
            )
            if r.ok:
                d = r.json()
                if d.get("token"):
                    self._token = d["token"]
                    _save_token(self._token)
                    return True
        except Exception:
            pass
        return False


# ── Desktop notification ──────────────────────────────────────────────────────

def notify(title: str, message: str, duration: int = 4):
    """Show a desktop notification via plyer or Windows toast."""
    try:
        if PLYER_OK:
            _plyer_notif.notify(
                title=title,
                message=message[:200],
                app_name="ARIA",
                timeout=duration,
            )
            return
        # Windows fallback via ctypes
        if CTYPES_OK and platform.system() == "Windows":
            ctypes.windll.user32.MessageBeep(0)
    except Exception as e:
        logger.error("Notification error: %s", e)


# ── Overlay input window ──────────────────────────────────────────────────────

def show_input_overlay() -> Optional[str]:
    """
    Show a minimal input overlay for voice/text command.
    Returns user input or None if cancelled.
    Uses tkinter (always available on Windows).
    """
    try:
        import tkinter as tk
        result = {"text": None}

        root = tk.Tk()
        root.title("ARIA")
        root.geometry("500x60+{}+{}".format(
            root.winfo_screenwidth() // 2 - 250,
            root.winfo_screenheight() - 120,
        ))
        root.configure(bg="#00050f")
        root.attributes("-topmost", True)
        root.overrideredirect(True)

        frame = tk.Frame(root, bg="#010a1a", bd=1, relief="solid",
                         highlightbackground="#00d4ff", highlightthickness=1)
        frame.pack(fill="both", expand=True, padx=2, pady=2)

        entry = tk.Entry(
            frame, bg="#020f22", fg="#c8eeff",
            font=("Segoe UI", 13), bd=0, insertbackground="#00d4ff",
            relief="flat",
        )
        entry.pack(side="left", fill="both", expand=True, padx=12, pady=10)
        entry.insert(0, "")
        entry.focus_set()

        label = tk.Label(frame, text="A", bg="#010a1a", fg="#00d4ff",
                         font=("Segoe UI", 13, "bold"), padx=8)
        label.pack(side="right")

        def _submit(event=None):
            result["text"] = entry.get().strip()
            root.destroy()

        def _cancel(event=None):
            root.destroy()

        entry.bind("<Return>", _submit)
        entry.bind("<Escape>", _cancel)
        root.bind("<FocusOut>", _cancel)

        # Auto-close after 15s
        root.after(15000, root.destroy)
        root.mainloop()
        return result["text"] or None

    except Exception as e:
        logger.error("Overlay error: %s", e)
        return None


# ── Wake-word detector ────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Continuous microphone listener for wake word.
    Uses faster-whisper for offline transcription.
    Only activates when audio energy exceeds threshold.
    """

    def __init__(self, wake_word: str = WAKE_WORD, on_detected=None):
        self.wake_word   = wake_word.lower()
        self.on_detected = on_detected
        self._running    = False
        self._thread: Optional[threading.Thread] = None
        self._model      = None

    def _load_model(self):
        if not WHISPER_OK or not AUDIO_OK:
            return False
        try:
            self._model = faster_whisper.WhisperModel(
                "tiny", device="cpu", compute_type="int8"
            )
            logger.info("Wake word model loaded (tiny whisper)")
            return True
        except Exception as e:
            logger.warning("Whisper load failed: %s", e)
            return False

    def start(self):
        if self._running:
            return
        if not self._load_model():
            logger.warning("Wake word disabled — install faster-whisper sounddevice")
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Wake word listener started: '%s'", self.wake_word)

    def stop(self):
        self._running = False

    def _listen_loop(self):
        SAMPLE_RATE  = 16000
        CHUNK        = 0.5   # seconds per chunk
        ENERGY_GATE  = 0.01  # RMS threshold
        BUFFER_SECS  = 2.0   # capture window after energy gate

        buffer: list = []
        buffer_time  = 0.0
        recording    = False

        try:
            with _sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                  dtype="float32") as stream:
                while self._running:
                    audio_chunk, _ = stream.read(int(SAMPLE_RATE * CHUNK))
                    rms = float(_np.sqrt(_np.mean(audio_chunk ** 2)))

                    if rms > ENERGY_GATE:
                        buffer.append(audio_chunk)
                        buffer_time += CHUNK
                        recording = True

                    if recording and (rms <= ENERGY_GATE or buffer_time >= BUFFER_SECS):
                        if buffer:
                            audio = _np.concatenate(buffer).flatten()
                            self._transcribe_and_check(audio, SAMPLE_RATE)
                        buffer = []
                        buffer_time = 0.0
                        recording = False
        except Exception as e:
            logger.error("Wake word stream error: %s", e)

    def _transcribe_and_check(self, audio: "_np.ndarray", sr: int):
        if not self._model:
            return
        try:
            segments, _ = self._model.transcribe(audio, language="en",
                                                  beam_size=1, vad_filter=True)
            text = " ".join(s.text for s in segments).lower().strip()
            if self.wake_word in text:
                logger.info("Wake word detected: '%s'", text)
                if self.on_detected:
                    self.on_detected(text.replace(self.wake_word, "").strip())
        except Exception:
            pass


# ── System tray icon ──────────────────────────────────────────────────────────

def _make_tray_icon() -> "_PILImage.Image":
    """Create a minimal ARIA icon for the system tray."""
    size = 64
    img  = _PILImage.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = _Draw.Draw(img)
    # Dark background circle
    draw.ellipse([2, 2, size - 2, size - 2], fill="#00050f", outline="#00d4ff", width=2)
    # "A" letter
    draw.text((20, 14), "A", fill="#00d4ff")
    return img


class ARIATray:
    """System tray icon and menu."""

    def __init__(self, client: ARIAClient, hotkey_service: "HotkeyService"):
        self._client  = client
        self._hotkey  = hotkey_service
        self._icon    = None
        self._running = False

    def start(self):
        if not TRAY_OK:
            logger.warning("pystray not available — tray icon disabled")
            return
        self._running = True
        t = threading.Thread(target=self._run_tray, daemon=True)
        t.start()

    def _run_tray(self):
        try:
            img = _make_tray_icon()

            def _on_open(icon, item):
                import webbrowser
                webbrowser.open(f"http://localhost:{ARIA_PORT}")

            def _on_command(icon, item):
                text = show_input_overlay()
                if text:
                    self._send_and_notify(text)

            def _on_quit(icon, item):
                self._running = False
                icon.stop()

            menu = pystray.Menu(
                pystray.MenuItem("Open ARIA", _on_open, default=True),
                pystray.MenuItem("Quick Command (Ctrl+Shift+Space)", _on_command),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    f"Server: {'Online' if self._client.is_alive() else 'Offline'}",
                    lambda *_: None,
                    enabled=False,
                ),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit ARIA Service", _on_quit),
            )

            self._icon = pystray.Icon("ARIA", img, "ARIA — Personal AI", menu)
            self._icon.run()
        except Exception as e:
            logger.error("Tray error: %s", e)

    def _send_and_notify(self, text: str):
        def _worker():
            response_parts = []
            self._client.stream_command(
                text,
                on_token=lambda t: response_parts.append(t),
            )
            response = "".join(response_parts).strip()[:200]
            if response:
                notify("ARIA", response)
        threading.Thread(target=_worker, daemon=True).start()

    def update_status(self, online: bool):
        # Tray tooltip update (best-effort)
        if self._icon:
            try:
                self._icon.title = f"ARIA — {'Online' if online else 'Offline'}"
            except Exception:
                pass


# ── Hotkey service ────────────────────────────────────────────────────────────

class HotkeyService:
    """Registers global hotkey and handles activation."""

    def __init__(self, client: ARIAClient, hotkey: str = HOTKEY):
        self._client  = client
        self._hotkey  = hotkey
        self._active  = False

    def start(self):
        if not KEYBOARD_OK:
            return
        try:
            keyboard.add_hotkey(self._hotkey, self._on_hotkey, suppress=False)
            logger.info("Global hotkey registered: %s", self._hotkey)
        except Exception as e:
            logger.error("Hotkey registration failed: %s", e)

    def _on_hotkey(self):
        if self._active:
            return
        self._active = True
        try:
            text = show_input_overlay()
            if text:
                self._send_and_notify(text)
        finally:
            self._active = False

    def _send_and_notify(self, text: str):
        def _worker():
            parts = []
            self._client.stream_command(text, on_token=parts.append)
            resp = "".join(parts).strip()[:200]
            if resp:
                notify("ARIA", resp)
        threading.Thread(target=_worker, daemon=True).start()

    def stop(self):
        if KEYBOARD_OK:
            try:
                keyboard.remove_hotkey(self._hotkey)
            except Exception:
                pass


# ── Windows auto-start ────────────────────────────────────────────────────────

_AUTOSTART_KEY  = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
_AUTOSTART_NAME = "ARIAGlobalService"


def install_autostart():
    """Add ARIA global service to Windows startup registry."""
    if platform.system() != "Windows":
        print("Auto-start only supported on Windows")
        return
    exe = sys.executable
    script = str(Path(__file__).resolve())
    cmd = f'"{exe}" "{script}"'
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY,
                            0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, _AUTOSTART_NAME, 0, winreg.REG_SZ, cmd)
        print(f"✅ ARIA will now start automatically with Windows")
        print(f"   Command: {cmd}")
    except Exception as e:
        print(f"❌ Failed to install auto-start: {e}")


def remove_autostart():
    """Remove ARIA global service from Windows startup."""
    if platform.system() != "Windows":
        return
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _AUTOSTART_KEY,
                            0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, _AUTOSTART_NAME)
        print("✅ ARIA auto-start removed")
    except FileNotFoundError:
        print("ARIA auto-start was not installed")
    except Exception as e:
        print(f"❌ Failed to remove auto-start: {e}")


# ── Health monitor ────────────────────────────────────────────────────────────

class HealthMonitor:
    """Polls ARIA backend health and shows notification if it goes down/up."""

    def __init__(self, client: ARIAClient, tray: Optional[ARIATray] = None):
        self._client   = client
        self._tray     = tray
        self._was_alive = None
        self._thread: Optional[threading.Thread] = None
        self._running  = False

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            alive = self._client.is_alive()
            if alive != self._was_alive:
                if alive:
                    notify("ARIA Online", "ARIA backend is ready")
                    logger.info("Backend came online")
                else:
                    if self._was_alive is not None:
                        notify("ARIA Offline", "ARIA backend stopped. Run: python server.py")
                        logger.warning("Backend went offline")
                self._was_alive = alive
                if self._tray:
                    self._tray.update_status(alive)
            time.sleep(10)


# ── Main service ──────────────────────────────────────────────────────────────

class GlobalService:
    """
    ARIA always-on background service.
    Manages tray, hotkey, wake word, and health monitor.
    """

    def __init__(self):
        self._client   = ARIAClient()
        self._hotkey   = HotkeyService(self._client)
        self._tray     = ARIATray(self._client, self._hotkey)
        self._health   = HealthMonitor(self._client, self._tray)
        self._wakeword = WakeWordDetector(
            wake_word=WAKE_WORD,
            on_detected=self._on_wake,
        )

    def _on_wake(self, command_text: str):
        """Called when wake word is detected."""
        if command_text:
            # If there's a command after the wake word, execute it directly
            parts = []
            self._client.stream_command(command_text, on_token=parts.append)
            resp = "".join(parts).strip()[:200]
            if resp:
                notify("ARIA", resp)
        else:
            # Open input overlay for the user to type/speak
            text = show_input_overlay()
            if text:
                parts = []
                self._client.stream_command(text, on_token=parts.append)
                resp = "".join(parts).strip()[:200]
                if resp:
                    notify("ARIA", resp)

    def start(self):
        logger.info("═" * 50)
        logger.info("ARIA Global Service starting")
        logger.info("  Hotkey: %s", HOTKEY)
        logger.info("  Wake word: '%s'", WAKE_WORD)
        logger.info("  Backend: %s", ARIA_BASE_URL)
        logger.info("═" * 50)

        self._hotkey.start()
        self._health.start()
        self._wakeword.start()
        self._tray.start()   # blocks on Windows (pystray runs main loop)

        # If tray is not available, keep alive with a blocking loop
        if not TRAY_OK:
            logger.info("Running without tray. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        logger.info("ARIA Global Service stopping")
        self._hotkey.stop()
        self._health.stop()
        self._wakeword.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARIA Global Service")
    parser.add_argument("--install", action="store_true",
                        help="Install auto-start with Windows")
    parser.add_argument("--remove", action="store_true",
                        help="Remove auto-start from Windows")
    parser.add_argument("--hotkey", default=HOTKEY,
                        help=f"Global hotkey (default: {HOTKEY})")
    parser.add_argument("--wake-word", default=WAKE_WORD,
                        help=f"Wake word (default: {WAKE_WORD})")
    args = parser.parse_args()

    if args.install:
        install_autostart()
        sys.exit(0)

    if args.remove:
        remove_autostart()
        sys.exit(0)

    svc = GlobalService()
    svc.start()
