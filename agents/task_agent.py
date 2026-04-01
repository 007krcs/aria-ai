"""
ARIA — Task Agent
==================
Handles all scheduled and triggered tasks:
- Alarms (exact time)
- Reminders (time-based or condition-based)
- Price alerts (monitor stock/crypto, trigger when condition met)
- Birthday wishes (auto-send on the date)
- Email drafting and sending
- Recurring tasks

All tasks are persisted in SQLite.
All tasks survive server restarts.
All tasks fire notification + voice alert + optional message.
"""

import re
import json
import time
import sqlite3
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console      = Console()
DB_PATH      = PROJECT_ROOT / "logs" / "tasks.db"
DB_PATH.parent.mkdir(exist_ok=True)


def init_task_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL,
                title       TEXT,
                data        TEXT NOT NULL,
                trigger_ts  TEXT,
                repeat      TEXT DEFAULT 'none',
                status      TEXT DEFAULT 'pending',
                created_ts  TEXT,
                fired_ts    TEXT,
                device_id   TEXT
            );

            CREATE TABLE IF NOT EXISTS price_alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT,
                condition   TEXT,
                target      REAL,
                message     TEXT,
                status      TEXT DEFAULT 'active',
                created_ts  TEXT,
                fired_ts    TEXT
            );

            CREATE TABLE IF NOT EXISTS contacts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT,
                phone       TEXT,
                email       TEXT,
                birthday    TEXT,
                notes       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_task_status ON tasks(status, trigger_ts);
            CREATE INDEX IF NOT EXISTS idx_alert_status ON price_alerts(status);
        """)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

class TaskScheduler:
    """
    Background task runner.
    Checks every 30 seconds for tasks that need to fire.
    """

    def __init__(self, notifier=None, bus=None, engine=None):
        self.notifier = notifier
        self.bus      = bus
        self.engine   = engine
        self._running = False
        self._thread  = None
        init_task_db()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        console.print("  [green]Task scheduler running[/]")

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                self._check_tasks()
                self._check_price_alerts()
                self._check_birthdays()
            except Exception as e:
                console.print(f"  [yellow]Scheduler error: {e}[/]")
            time.sleep(30)

    def _check_tasks(self):
        """Fire any tasks whose trigger time has passed."""
        now = datetime.now().isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            tasks = conn.execute("""
                SELECT * FROM tasks
                WHERE status = 'pending'
                  AND trigger_ts <= ?
                ORDER BY trigger_ts
            """, (now,)).fetchall()

            for task in tasks:
                self._fire_task(dict(task), conn)

    def _fire_task(self, task: dict, conn):
        """Execute a task and mark it fired."""
        data  = json.loads(task["data"])
        title = task["title"] or task["type"]

        console.print(f"  [green]Task fired:[/] {title}")

        # Notify
        if self.notifier:
            self.notifier.notify(f"ARIA — {title}", data.get("message",""))

        # Publish to bus
        if self.bus:
            from agents.agent_bus import Event
            self.bus.publish(Event(
                f"task_fired",
                {"task": task, "data": data},
                "scheduler"
            ))

        # Handle special task types
        if task["type"] == "birthday_wish":
            self._send_birthday_wish(data)

        # Mark as fired
        now = datetime.now().isoformat()
        if task["repeat"] == "none":
            conn.execute(
                "UPDATE tasks SET status='fired', fired_ts=? WHERE id=?",
                (now, task["id"])
            )
        else:
            # Reschedule for next occurrence
            next_ts = self._next_occurrence(task["trigger_ts"], task["repeat"])
            conn.execute(
                "UPDATE tasks SET trigger_ts=?, fired_ts=? WHERE id=?",
                (next_ts, now, task["id"])
            )

    def _next_occurrence(self, trigger_ts: str, repeat: str) -> str:
        dt = datetime.fromisoformat(trigger_ts)
        if repeat == "daily":
            dt += timedelta(days=1)
        elif repeat == "weekly":
            dt += timedelta(weeks=1)
        elif repeat == "monthly":
            dt = dt.replace(month=dt.month % 12 + 1)
        elif repeat == "yearly":
            dt = dt.replace(year=dt.year + 1)
        return dt.isoformat()

    def _check_price_alerts(self):
        """Check if any price alert conditions have been met."""
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            alerts = conn.execute(
                "SELECT * FROM price_alerts WHERE status = 'active'"
            ).fetchall()

        for alert in alerts:
            self._check_single_alert(dict(alert))

    def _check_single_alert(self, alert: dict):
        """Check a single price alert."""
        symbol    = alert["symbol"]
        target    = alert["target"]
        condition = alert["condition"]  # "above", "below", "equals"

        price = self._get_price(symbol)
        if price is None:
            return

        triggered = (
            (condition == "above"  and price >= target) or
            (condition == "below"  and price <= target) or
            (condition == "equals" and abs(price - target) < target * 0.01)
        )

        if triggered:
            msg = f"{symbol} is now {price:.2f} (target was {target:.2f})"
            console.print(f"  [green]Price alert:[/] {msg}")

            if self.notifier:
                self.notifier.notify("ARIA — Price Alert", msg)

            if self.bus:
                from agents.agent_bus import Event
                self.bus.publish(Event(
                    "price_alert_triggered",
                    {"symbol": symbol, "price": price, "target": target, "message": msg},
                    "task_scheduler"
                ))

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE price_alerts SET status='fired', fired_ts=? WHERE id=?",
                    (datetime.now().isoformat(), alert["id"])
                )

    def _get_price(self, symbol: str) -> Optional[float]:
        """Get current price of a stock/crypto symbol."""
        try:
            import requests
            # Use Yahoo Finance (free, no key)
            r = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            )
            data  = r.json()
            price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
            return float(price)
        except Exception:
            return None

    def _check_birthdays(self):
        """Check for birthdays today and queue wishes."""
        today = datetime.now().strftime("%m-%d")
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            contacts = conn.execute(
                "SELECT * FROM contacts WHERE birthday LIKE ?",
                (f"%-{today}",)
            ).fetchall()

        for contact in contacts:
            # Check if already wished today
            today_full = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(DB_PATH) as conn:
                already = conn.execute("""
                    SELECT id FROM tasks
                    WHERE type='birthday_wish'
                      AND data LIKE ?
                      AND fired_ts LIKE ?
                """, (f'%{contact["name"]}%', f'{today_full}%')).fetchone()

            if not already:
                if self.notifier:
                    self.notifier.notify(
                        "Birthday Today!",
                        f"Today is {contact['name']}'s birthday!"
                    )

    def _send_birthday_wish(self, data: dict):
        """Generate and optionally send a birthday message."""
        if not self.engine:
            return
        contact = data.get("name","")
        prompt  = (
            f"Write a warm, genuine birthday message for {contact}. "
            f"3-4 sentences. Friendly and personal. No emojis."
        )
        try:
            message = self.engine.generate(prompt, temperature=0.7)
            console.print(f"  [green]Birthday wish for {contact}:[/] {message[:80]}...")
            if self.bus:
                from agents.agent_bus import Event
                self.bus.publish(Event(
                    "birthday_wish_ready",
                    {"contact": contact, "message": message, "data": data},
                    "scheduler"
                ))
        except Exception:
            pass

    # ── Task creation APIs ────────────────────────────────────────────────────

    def add_alarm(
        self,
        time_str:  str,
        label:     str = "ARIA Alarm",
        repeat:    str = "none",
    ) -> dict:
        """Add an alarm. time_str: 'tomorrow 7am', '15:30', 'in 2 hours'"""
        trigger = self._parse_trigger_time(time_str)
        if not trigger:
            return {"success": False, "error": f"Could not parse time: {time_str}"}

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO tasks (type, title, data, trigger_ts, repeat, created_ts) "
                "VALUES (?,?,?,?,?,?)",
                ("alarm", label,
                 json.dumps({"label": label, "message": f"Alarm: {label}"}),
                 trigger, repeat, datetime.now().isoformat())
            )

        return {
            "success":    True,
            "task_id":    cursor.lastrowid,
            "fires_at":   trigger,
            "label":      label,
            "repeat":     repeat,
        }

    def add_reminder(
        self,
        text:      str,
        time_str:  str,
        repeat:    str = "none",
    ) -> dict:
        trigger = self._parse_trigger_time(time_str)
        if not trigger:
            return {"success": False, "error": f"Could not parse time: {time_str}"}

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO tasks (type, title, data, trigger_ts, repeat, created_ts) "
                "VALUES (?,?,?,?,?,?)",
                ("reminder", text[:100],
                 json.dumps({"message": text}),
                 trigger, repeat, datetime.now().isoformat())
            )

        return {
            "success":  True,
            "task_id":  cursor.lastrowid,
            "fires_at": trigger,
            "text":     text,
        }

    def add_price_alert(
        self,
        symbol:    str,
        target:    float,
        condition: str = "above",
        message:   str = "",
    ) -> dict:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO price_alerts (symbol, condition, target, message, created_ts) "
                "VALUES (?,?,?,?,?)",
                (symbol.upper(), condition, target,
                 message or f"Alert: {symbol} {condition} {target}",
                 datetime.now().isoformat())
            )

        # Get current price for reference
        current = self._get_price(symbol.upper())

        return {
            "success":      True,
            "alert_id":     cursor.lastrowid,
            "symbol":       symbol.upper(),
            "condition":    condition,
            "target":       target,
            "current_price": current,
            "message":      f"Watching {symbol.upper()} — will notify when {condition} {target}",
        }

    def add_birthday(
        self,
        name:     str,
        date_str: str,
        phone:    str = "",
        email:    str = "",
    ) -> dict:
        """Add a contact birthday. date_str: '25 December', '12-25', 'December 25'"""
        # Parse date to MM-DD
        bday = self._parse_birthday(date_str)

        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "INSERT INTO contacts (name, phone, email, birthday) VALUES (?,?,?,?)",
                (name, phone, email, bday)
            )

        return {
            "success": True,
            "name":    name,
            "birthday": bday,
            "note":    "ARIA will notify you on their birthday and prepare a wish",
        }

    def list_upcoming(self, days: int = 7) -> list[dict]:
        """List tasks firing in the next N days."""
        future = (datetime.now() + timedelta(days=days)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, type, title, trigger_ts, repeat, status
                FROM tasks
                WHERE status = 'pending' AND trigger_ts <= ?
                ORDER BY trigger_ts
            """, (future,)).fetchall()
        return [dict(r) for r in rows]

    def list_price_alerts(self) -> list[dict]:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM price_alerts WHERE status = 'active'"
            ).fetchall()
        alerts = []
        for r in rows:
            a    = dict(r)
            a["current_price"] = self._get_price(a["symbol"])
            alerts.append(a)
        return alerts

    # ── Time parsing ──────────────────────────────────────────────────────────

    def _parse_trigger_time(self, text: str) -> Optional[str]:
        text = text.strip().lower()
        now  = datetime.now()

        # "in X minutes/hours"
        m = re.search(r"in\s+(\d+)\s+(minute|hour|day)", text)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            delta   = {"minute": timedelta(minutes=n),
                       "hour":   timedelta(hours=n),
                       "day":    timedelta(days=n)}[unit]
            return (now + delta).isoformat()

        # "tomorrow at X"
        if "tomorrow" in text:
            base = now + timedelta(days=1)
            t    = self._parse_time_of_day(text)
            if t:
                return base.replace(hour=t[0], minute=t[1], second=0).isoformat()

        # "tonight at X"
        if "tonight" in text:
            t = self._parse_time_of_day(text)
            if t:
                return now.replace(hour=t[0], minute=t[1], second=0).isoformat()

        # "at X:XX am/pm"
        t = self._parse_time_of_day(text)
        if t:
            dt = now.replace(hour=t[0], minute=t[1], second=0)
            if dt < now:
                dt += timedelta(days=1)
            return dt.isoformat()

        return None

    def _parse_time_of_day(self, text: str) -> Optional[tuple[int, int]]:
        m = re.search(r"(\d{1,2})[:.](\d{2})\s*(am|pm)?", text)
        if m:
            h, mn = int(m.group(1)), int(m.group(2))
            if m.group(3) == "pm" and h < 12: h += 12
            elif m.group(3) == "am" and h == 12: h = 0
            return h, mn
        m = re.search(r"(\d{1,2})\s*(am|pm)", text)
        if m:
            h = int(m.group(1))
            if m.group(2) == "pm" and h < 12: h += 12
            elif m.group(2) == "am" and h == 12: h = 0
            return h, 0
        return None

    def _parse_birthday(self, date_str: str) -> str:
        months = {
            "jan":"01","feb":"02","mar":"03","apr":"04",
            "may":"05","jun":"06","jul":"07","aug":"08",
            "sep":"09","oct":"10","nov":"11","dec":"12",
            "january":"01","february":"02","march":"03","april":"04",
            "june":"06","july":"07","august":"08",
            "september":"09","october":"10","november":"11","december":"12",
        }
        date_str = date_str.strip().lower()
        m = re.search(r"(\d{1,2})[-/](\d{1,2})", date_str)
        if m:
            return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
        for name, num in months.items():
            if name in date_str:
                day_m = re.search(r"(\d{1,2})", date_str)
                day   = day_m.group(1) if day_m else "1"
                return f"{num}-{int(day):02d}"
        return date_str


# ─────────────────────────────────────────────────────────────────────────────
# EMAIL AGENT
# ─────────────────────────────────────────────────────────────────────────────

class EmailAgent:
    """
    Drafts and sends emails via SMTP.
    Supports Gmail (app password) and any SMTP server.
    Uses LLM to draft email bodies from instructions.
    """

    def __init__(self, engine=None):
        self.engine = engine
        self._smtp_config = self._load_config()

    def _load_config(self) -> dict:
        config_file = PROJECT_ROOT / ".env"
        config      = {}
        try:
            for line in config_file.read_text().split("\n"):
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    config[k.strip()] = v.strip()
        except Exception:
            pass
        return {
            "smtp_host":  config.get("SMTP_HOST","smtp.gmail.com"),
            "smtp_port":  int(config.get("SMTP_PORT","587")),
            "email":      config.get("EMAIL_ADDRESS",""),
            "password":   config.get("EMAIL_PASSWORD",""),  # Gmail app password
        }

    def draft(self, to: str, subject: str, instruction: str) -> dict:
        """
        Draft an email body from an instruction.
        e.g. instruction = "reply saying I'll be available Thursday"
        """
        if not self.engine:
            return {"subject": subject, "body": instruction, "drafted": False}

        prompt = (
            f"Write a professional email.\n"
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"Instruction: {instruction}\n\n"
            f"Write only the email body (no subject line, no headers):\n"
        )
        try:
            body = self.engine.generate(prompt, temperature=0.4)
            return {"subject": subject, "body": body, "to": to, "drafted": True}
        except Exception as e:
            return {"subject": subject, "body": instruction, "drafted": False, "error": str(e)}

    def send(self, to: str, subject: str, body: str) -> dict:
        """Send an email via SMTP."""
        cfg = self._smtp_config
        if not cfg["email"] or not cfg["password"]:
            return {
                "success": False,
                "error": "Email not configured. Add EMAIL_ADDRESS and EMAIL_PASSWORD to .env",
                "note":  "For Gmail: create an App Password at myaccount.google.com/apppasswords",
            }

        try:
            msg             = MIMEMultipart()
            msg["From"]     = cfg["email"]
            msg["To"]       = to
            msg["Subject"]  = subject
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as server:
                server.starttls()
                server.login(cfg["email"], cfg["password"])
                server.send_message(msg)

            console.print(f"  [green]Email sent:[/] to {to}")
            return {"success": True, "to": to, "subject": subject}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def draft_and_send(self, to: str, subject: str, instruction: str) -> dict:
        """Draft an email and send it."""
        draft  = self.draft(to, subject, instruction)
        result = self.send(to, draft["subject"], draft["body"])
        return {**draft, **result}
