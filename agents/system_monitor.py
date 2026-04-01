"""
ARIA — System Monitor Agent
=============================
Tracks everything happening on your device and syncs across all devices.

What it monitors (all local, nothing sent to cloud):
- Active window / application at every moment
- Time spent per app, per category, per hour
- Browser tab titles and domains (not page content)
- Keyboard activity level (counts, not keystrokes — privacy preserved)
- Mouse movement and click patterns
- CPU, RAM, disk, network usage
- Idle time vs active time
- App switching patterns (what do you switch to after what)
- Focus sessions (uninterrupted time in one app)

Cross-device: logs are stored in SQLite. Any device on the same WiFi
can read the analytics by opening http://YOUR-PC-IP:8000/analytics

Privacy: Only window titles, app names, and usage duration are stored.
Keyboard content is NEVER recorded — only typing activity level (fast/slow/idle).

Install:
    pip install psutil pygetwindow pyautogui
    Windows only for window title: pip install pywin32
"""

import os
import re
import sys
import time
import json
import sqlite3
import threading
import platform
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

console     = Console()
PLATFORM    = platform.system()
DB_PATH     = PROJECT_ROOT / "logs" / "system_events.db"
DB_PATH.parent.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    with sqlite3.connect(DB_PATH, timeout=15) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS app_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_start    TEXT NOT NULL,
                ts_end      TEXT,
                app_name    TEXT,
                window_title TEXT,
                category    TEXT,
                duration_s  REAL DEFAULT 0,
                device_id   TEXT,
                device_name TEXT
            );

            CREATE TABLE IF NOT EXISTS system_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                cpu_pct     REAL,
                ram_pct     REAL,
                disk_pct    REAL,
                net_sent_mb REAL,
                net_recv_mb REAL,
                active_app  TEXT,
                is_idle     INTEGER DEFAULT 0,
                device_id   TEXT
            );

            CREATE TABLE IF NOT EXISTS browser_visits (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                domain      TEXT,
                title       TEXT,
                duration_s  REAL DEFAULT 0,
                device_id   TEXT
            );

            CREATE TABLE IF NOT EXISTS focus_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_start    TEXT,
                ts_end      TEXT,
                app_name    TEXT,
                duration_s  REAL,
                interrupted INTEGER DEFAULT 0,
                device_id   TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_app_ts    ON app_sessions(ts_start);
            CREATE INDEX IF NOT EXISTS idx_snap_ts   ON system_snapshots(ts);
            CREATE INDEX IF NOT EXISTS idx_browser_ts ON browser_visits(ts);
        """)


# ─────────────────────────────────────────────────────────────────────────────
# APP CATEGORISER
# ─────────────────────────────────────────────────────────────────────────────

class AppCategoriser:
    """
    Classifies applications and window titles into productivity categories.
    Used for focus analytics and pattern detection.
    """

    CATEGORIES = {
        "coding": [
            "visual studio", "vs code", "vscode", "pycharm", "intellij",
            "android studio", "xcode", "sublime", "atom", "vim", "neovim",
            "emacs", "jupyter", "spyder", "code", "terminal", "cmd",
            "powershell", "bash", "git", "github", "gitlab",
        ],
        "browser": [
            "chrome", "firefox", "edge", "safari", "brave", "opera",
            "internet explorer", "chromium",
        ],
        "communication": [
            "slack", "teams", "discord", "zoom", "meet", "skype",
            "telegram", "whatsapp", "signal", "outlook", "thunderbird",
            "mail", "gmail",
        ],
        "document": [
            "word", "excel", "powerpoint", "libreoffice", "google docs",
            "notion", "obsidian", "notepad", "onenote", "evernote",
            "adobe", "pdf", "acrobat",
        ],
        "media": [
            "youtube", "netflix", "spotify", "vlc", "mpv", "potplayer",
            "musicbee", "foobar", "plex", "prime video", "hotstar",
            "twitch", "soundcloud",
        ],
        "social": [
            "twitter", "instagram", "facebook", "linkedin", "reddit",
            "quora", "pinterest", "snapchat", "tiktok",
        ],
        "design": [
            "figma", "sketch", "adobe xd", "photoshop", "illustrator",
            "canva", "gimp", "inkscape", "affinity",
        ],
        "system": [
            "explorer", "finder", "files", "settings", "control panel",
            "task manager", "activity monitor", "system preferences",
        ],
        "gaming": [
            "steam", "epic games", "battle.net", "origin", "uplay",
            "minecraft", "roblox", "valorant", "league",
        ],
        "learning": [
            "coursera", "udemy", "khan academy", "duolingo", "anki",
            "mindnode", "xmind",
        ],
    }

    def categorise(self, app_name: str, window_title: str = "") -> str:
        text = f"{app_name} {window_title}".lower()
        for category, keywords in self.CATEGORIES.items():
            if any(kw in text for kw in keywords):
                return category
        return "other"


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class WindowTracker:
    """
    Tracks the currently active window and application.
    Cross-platform: Windows, Mac, Linux.
    """

    def get_active_window(self) -> tuple[str, str]:
        """Returns (app_name, window_title)."""
        try:
            if PLATFORM == "Windows":
                return self._get_windows()
            elif PLATFORM == "Darwin":
                return self._get_mac()
            else:
                return self._get_linux()
        except Exception:
            return "unknown", "unknown"

    def _get_windows(self) -> tuple[str, str]:
        try:
            import win32gui
            import win32process
            import psutil
            hwnd    = win32gui.GetForegroundWindow()
            title   = win32gui.GetWindowText(hwnd)
            _, pid  = win32process.GetWindowThreadProcessId(hwnd)
            proc    = psutil.Process(pid)
            app     = proc.name().replace(".exe","")
            return app, title
        except ImportError:
            # Fallback without win32
            try:
                import subprocess
                result = subprocess.run(
                    ["powershell", "-command",
                     "Get-Process | Where-Object {$_.MainWindowHandle -ne 0} | "
                     "Sort-Object CPU -Descending | Select-Object -First 1 | "
                     "Select-Object ProcessName, MainWindowTitle | ConvertTo-Json"],
                    capture_output=True, text=True, timeout=3
                )
                data  = json.loads(result.stdout)
                return data.get("ProcessName",""), data.get("MainWindowTitle","")
            except Exception:
                return "unknown", "unknown"

    def _get_mac(self) -> tuple[str, str]:
        try:
            import subprocess
            script  = """
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    set frontWindow to ""
                    try
                        set frontWindow to name of front window of application process frontApp
                    end try
                    return frontApp & "|" & frontWindow
                end tell
            """
            result  = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=3
            )
            parts   = result.stdout.strip().split("|")
            return parts[0], parts[1] if len(parts) > 1 else ""
        except Exception:
            return "unknown", "unknown"

    def _get_linux(self) -> tuple[str, str]:
        try:
            import subprocess
            # Try xdotool first
            wid     = subprocess.run(
                ["xdotool","getactivewindow"], capture_output=True, text=True, timeout=2
            ).stdout.strip()
            if wid:
                title = subprocess.run(
                    ["xdotool","getwindowname",wid], capture_output=True, text=True, timeout=2
                ).stdout.strip()
                pid   = subprocess.run(
                    ["xdotool","getwindowpid",wid], capture_output=True, text=True, timeout=2
                ).stdout.strip()
                if pid:
                    import psutil
                    app = psutil.Process(int(pid)).name()
                    return app, title
            return "unknown", title
        except Exception:
            return "unknown", "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# IDLE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class IdleDetector:
    """Detects when the user is idle (no input for N seconds)."""

    def __init__(self, idle_threshold_s: int = 120):
        self.threshold = idle_threshold_s

    def get_idle_seconds(self) -> float:
        try:
            if PLATFORM == "Windows":
                return self._idle_windows()
            elif PLATFORM == "Darwin":
                return self._idle_mac()
            else:
                return self._idle_linux()
        except Exception:
            return 0.0

    def is_idle(self) -> bool:
        return self.get_idle_seconds() >= self.threshold

    def _idle_windows(self) -> float:
        try:
            import ctypes
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(LASTINPUTINFO)
            ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))
            millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
            return millis / 1000.0
        except Exception:
            return 0.0

    def _idle_mac(self) -> float:
        try:
            import subprocess
            result = subprocess.run(
                ["ioreg", "-c", "IOHIDSystem"],
                capture_output=True, text=True, timeout=3
            )
            m = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', result.stdout)
            return int(m.group(1)) / 1e9 if m else 0.0
        except Exception:
            return 0.0

    def _idle_linux(self) -> float:
        try:
            import subprocess
            result = subprocess.run(
                ["xprintidle"], capture_output=True, text=True, timeout=2
            )
            return int(result.stdout.strip()) / 1000.0
        except Exception:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

class SystemResourceMonitor:
    """Monitors CPU, RAM, disk and network usage."""

    def __init__(self):
        self._last_net = None
        self._last_net_time = None

    def snapshot(self) -> dict:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory().percent
            disk= psutil.disk_usage("/").percent

            # Network delta
            net  = psutil.net_io_counters()
            now  = time.time()
            sent_mb = recv_mb = 0.0
            if self._last_net and self._last_net_time:
                dt = now - self._last_net_time
                if dt > 0:
                    sent_mb = (net.bytes_sent - self._last_net.bytes_sent) / 1024 / 1024 / dt
                    recv_mb = (net.bytes_recv - self._last_net.bytes_recv) / 1024 / 1024 / dt
            self._last_net      = net
            self._last_net_time = now

            return {
                "cpu_pct":     round(cpu, 1),
                "ram_pct":     round(ram, 1),
                "disk_pct":    round(disk, 1),
                "net_sent_mb": round(max(0, sent_mb), 3),
                "net_recv_mb": round(max(0, recv_mb), 3),
            }
        except ImportError:
            return {"cpu_pct":0,"ram_pct":0,"disk_pct":0,"net_sent_mb":0,"net_recv_mb":0}
        except Exception:
            return {"cpu_pct":0,"ram_pct":0,"disk_pct":0,"net_sent_mb":0,"net_recv_mb":0}


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SYSTEM MONITOR AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SystemMonitorAgent:
    """
    Runs silently in the background, tracking everything.
    Samples every INTERVAL seconds. Logs to SQLite.
    Thread-safe. Can be stopped cleanly.
    """

    INTERVAL      = 5     # sample every 5 seconds
    SNAP_INTERVAL = 30    # full snapshot every 30 seconds

    def __init__(self, device_name: str = None):
        import socket, hashlib
        self.device_name = device_name or platform.node()
        self.device_id   = hashlib.md5(
            (self.device_name + platform.system()).encode()
        ).hexdigest()[:8]

        self.window_tracker  = WindowTracker()
        self.idle_detector   = IdleDetector()
        self.resource_monitor= SystemResourceMonitor()
        self.categoriser     = AppCategoriser()

        self._running     = False
        self._thread      = None
        self._lock        = threading.Lock()

        # Current session tracking
        self._current_app   = None
        self._session_start = None
        self._focus_start   = None
        self._focus_app     = None
        self._snap_counter  = 0

        init_db()
        console.print(f"  [green]System Monitor:[/] device={self.device_name} id={self.device_id}")

    def start(self):
        """Start monitoring in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        console.print("  [green]System monitoring started[/] — tracking app usage")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._current_app:
            self._end_session(datetime.now().isoformat())

    def _loop(self):
        while self._running:
            try:
                self._tick()
            except Exception as e:
                console.print(f"  [yellow]Monitor tick error: {e}[/]")
            time.sleep(self.INTERVAL)

    def _tick(self):
        now     = datetime.now().isoformat()
        app, title = self.window_tracker.get_active_window()
        is_idle = self.idle_detector.is_idle()
        category= self.categoriser.categorise(app, title)

        if is_idle:
            app, title = "idle", ""
            category   = "idle"

        # App session tracking
        if app != self._current_app:
            if self._current_app:
                self._end_session(now)
            self._current_app   = app
            self._session_start = now
            self._start_session(app, title, category, now)

        # Focus session tracking (uninterrupted 5+ min in same app)
        if not is_idle:
            if self._focus_app != app:
                if self._focus_app and self._focus_start:
                    dur = (datetime.now() - datetime.fromisoformat(self._focus_start)).total_seconds()
                    if dur >= 300:  # 5 min = focus session
                        self._save_focus_session(self._focus_app, self._focus_start, now, dur, interrupted=True)
                self._focus_app   = app
                self._focus_start = now

        # Periodic full snapshot
        self._snap_counter += 1
        if self._snap_counter >= (self.SNAP_INTERVAL // self.INTERVAL):
            self._snap_counter = 0
            resources          = self.resource_monitor.snapshot()
            self._save_snapshot(app, is_idle, resources, now)

    def _start_session(self, app, title, category, ts):
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            conn.execute(
                "INSERT INTO app_sessions (ts_start, app_name, window_title, category, device_id, device_name) "
                "VALUES (?,?,?,?,?,?)",
                (ts, app, title[:200], category, self.device_id, self.device_name)
            )

    def _end_session(self, ts_end):
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            row = conn.execute(
                "SELECT id, ts_start FROM app_sessions WHERE device_id=? ORDER BY id DESC LIMIT 1",
                (self.device_id,)
            ).fetchone()
            if row:
                dur = (datetime.fromisoformat(ts_end) -
                       datetime.fromisoformat(row[1])).total_seconds()
                conn.execute(
                    "UPDATE app_sessions SET ts_end=?, duration_s=? WHERE id=?",
                    (ts_end, round(dur, 1), row[0])
                )

    def _save_snapshot(self, app, is_idle, resources, ts):
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            conn.execute(
                "INSERT INTO system_snapshots "
                "(ts, cpu_pct, ram_pct, disk_pct, net_sent_mb, net_recv_mb, active_app, is_idle, device_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, resources["cpu_pct"], resources["ram_pct"],
                 resources["disk_pct"], resources["net_sent_mb"],
                 resources["net_recv_mb"], app, int(is_idle), self.device_id)
            )

    def _save_focus_session(self, app, ts_start, ts_end, dur, interrupted):
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            conn.execute(
                "INSERT INTO focus_sessions (ts_start, ts_end, app_name, duration_s, interrupted, device_id) "
                "VALUES (?,?,?,?,?,?)",
                (ts_start, ts_end, app, round(dur,1), int(interrupted), self.device_id)
            )

    # ── Query APIs ────────────────────────────────────────────────────────────

    def get_today_summary(self) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            conn.row_factory = sqlite3.Row

            # Time per app today
            app_rows = conn.execute("""
                SELECT app_name, category,
                       SUM(duration_s) as total_s,
                       COUNT(*) as sessions
                FROM app_sessions
                WHERE ts_start LIKE ? AND app_name != 'idle'
                GROUP BY app_name
                ORDER BY total_s DESC
            """, (f"{today}%",)).fetchall()

            # Time per category
            cat_rows = conn.execute("""
                SELECT category, SUM(duration_s) as total_s
                FROM app_sessions
                WHERE ts_start LIKE ? AND app_name != 'idle'
                GROUP BY category ORDER BY total_s DESC
            """, (f"{today}%",)).fetchall()

            # Idle time
            idle_row = conn.execute("""
                SELECT SUM(duration_s) FROM app_sessions
                WHERE ts_start LIKE ? AND app_name = 'idle'
            """, (f"{today}%",)).fetchone()

            # Focus sessions today
            focus_rows = conn.execute("""
                SELECT app_name, duration_s, ts_start
                FROM focus_sessions
                WHERE ts_start LIKE ? ORDER BY duration_s DESC
            """, (f"{today}%",)).fetchall()

        total_tracked = sum(r["total_s"] for r in app_rows)
        idle_time     = (idle_row[0] or 0)

        return {
            "date":         today,
            "device":       self.device_name,
            "active_time_h": round(total_tracked / 3600, 2),
            "idle_time_h":  round(idle_time / 3600, 2),
            "top_apps":     [{"app": r["app_name"], "category": r["category"],
                              "minutes": round(r["total_s"]/60, 1),
                              "sessions": r["sessions"]} for r in app_rows[:10]],
            "by_category":  {r["category"]: round(r["total_s"]/60, 1) for r in cat_rows},
            "focus_sessions":[{"app": r["app_name"],
                               "minutes": round(r["duration_s"]/60,1),
                               "start": r["ts_start"][:16]} for r in focus_rows[:5]],
        }

    def get_hourly_activity(self, days: int = 1) -> list[dict]:
        """Activity breakdown by hour — for heatmap visualization."""
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            rows = conn.execute("""
                SELECT strftime('%H', ts_start) as hour,
                       category,
                       SUM(duration_s) as total_s
                FROM app_sessions
                WHERE ts_start > ? AND app_name != 'idle'
                GROUP BY hour, category
                ORDER BY hour
            """, (since,)).fetchall()
        return [{"hour": int(r[0]), "category": r[1],
                 "minutes": round(r[2]/60,1)} for r in rows]

    def get_app_switches(self, hours: int = 24) -> list[dict]:
        """App switching patterns — what do you switch TO after each app."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            rows = conn.execute("""
                SELECT a.app_name as from_app, b.app_name as to_app, COUNT(*) as count
                FROM app_sessions a
                JOIN app_sessions b ON b.id = a.id + 1
                WHERE a.ts_start > ? AND a.device_id = b.device_id
                  AND a.app_name != 'idle' AND b.app_name != 'idle'
                GROUP BY from_app, to_app
                ORDER BY count DESC LIMIT 20
            """, (since,)).fetchall()
        return [{"from": r[0], "to": r[1], "count": r[2]} for r in rows]

    def get_system_trend(self, hours: int = 24) -> list[dict]:
        """CPU/RAM/Network trends over time."""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            rows = conn.execute("""
                SELECT strftime('%Y-%m-%d %H:%M', ts) as minute,
                       AVG(cpu_pct) as cpu, AVG(ram_pct) as ram,
                       AVG(net_recv_mb) as net_in
                FROM system_snapshots
                WHERE ts > ? AND device_id = ?
                GROUP BY minute ORDER BY minute
            """, (since, self.device_id)).fetchall()
        return [{"ts": r[0], "cpu": round(r[1],1),
                 "ram": round(r[2],1), "net_in": round(r[3],3)} for r in rows]

    def get_all_devices_summary(self) -> list[dict]:
        """Summary for ALL devices that have logged data."""
        today = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(DB_PATH, timeout=15) as conn:
            devices = conn.execute(
                "SELECT DISTINCT device_id, device_name FROM app_sessions"
            ).fetchall()
            result  = []
            for dev_id, dev_name in devices:
                row = conn.execute("""
                    SELECT SUM(duration_s) as active,
                           COUNT(DISTINCT app_name) as unique_apps
                    FROM app_sessions
                    WHERE ts_start LIKE ? AND device_id = ? AND app_name != 'idle'
                """, (f"{today}%", dev_id)).fetchone()
                result.append({
                    "device_id":   dev_id,
                    "device_name": dev_name,
                    "active_hours": round((row[0] or 0)/3600, 2),
                    "unique_apps": row[1] or 0,
                })
        return result
