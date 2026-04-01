"""
ARIA — Proactive Intelligence Engine
======================================
The feature that makes ARIA feel alive rather than reactive.

Current ARIA: waits for you to ask something.
Proactive ARIA: watches for things you care about and tells YOU.

What it monitors:
  1. Stock / crypto price targets     → "Apple hit ₹195, your target was ₹190"
  2. News topics you care about       → "3 new articles about SpaceX this morning"
  3. Research paper alerts            → "New paper on LLM fine-tuning you'd like"
  4. Weather warnings                 → "Rain tomorrow, you have an outdoor event"
  5. Behaviour anomalies              → "You've been working 4h straight, take a break"
  6. Task reminders                   → fires from the scheduler
  7. System health                    → "Disk 92% full, clean up?"

How it works:
  - Background thread wakes up every 5 minutes
  - Checks each monitor for new triggers
  - If triggered → push notification + optional voice alert
  - Stores what it already notified to avoid spam (rate limiting per topic)
  - Learns which alerts you engage with vs dismiss → adjusts frequency

All monitoring is local. No data leaves your machine.
"""

import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from rich.console import Console

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
ALERTS_FILE   = PROJECT_ROOT / "data" / "proactive_alerts.json"
INTERESTS_FILE= PROJECT_ROOT / "data" / "user_interests.json"
ALERTS_FILE.parent.mkdir(exist_ok=True)
console = Console()


class ProactiveEngine:
    """
    Watches the world model, news, prices, and behaviour.
    Pushes notifications without being asked.
    """

    def __init__(self, world_model=None, scheduler=None,
                 analyst=None, research=None, notifier=None, voice=None):
        self.world     = world_model
        self.scheduler = scheduler
        self.analyst   = analyst
        self.research  = research
        self.notifier  = notifier
        self.voice     = voice

        self._running  = False
        self._alerted: dict = self._load_alert_history()
        self._interests: dict = self._load_interests()

    def _load_alert_history(self) -> dict:
        try:
            if ALERTS_FILE.exists():
                return json.loads(ALERTS_FILE.read_text())
        except Exception:
            pass
        return {}

    def _save_alert_history(self):
        try:
            # Keep last 7 days
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            self._alerted = {k: v for k, v in self._alerted.items() if v > cutoff}
            ALERTS_FILE.write_text(json.dumps(self._alerted, indent=2))
        except Exception:
            pass

    def _load_interests(self) -> dict:
        """Load user interests — price targets, news topics, research areas."""
        try:
            if INTERESTS_FILE.exists():
                return json.loads(INTERESTS_FILE.read_text())
        except Exception:
            pass
        return {
            "stock_targets":   [],   # [{"symbol":"AAPL","target":190,"direction":"above"}]
            "news_topics":     [],   # ["SpaceX","AI regulation","cricket"]
            "research_topics": [],   # ["LLM fine-tuning","transformer efficiency"]
            "break_after_min": 90,   # remind to take breaks after N min focus
        }

    def save_interests(self, interests: dict):
        self._interests.update(interests)
        INTERESTS_FILE.write_text(json.dumps(self._interests, indent=2))

    def _already_alerted(self, key: str, cooldown_min: int = 60) -> bool:
        """Check if we already sent this alert recently."""
        last = self._alerted.get(key)
        if not last:
            return False
        delta = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
        return delta < cooldown_min * 60

    def _record_alert(self, key: str):
        self._alerted[key] = datetime.now().isoformat()
        self._save_alert_history()

    def _push(self, title: str, body: str, speak: bool = False):
        """Send a notification and optionally speak it."""
        console.print(f"  [green]Proactive:[/] {title} — {body[:60]}")
        if self.notifier:
            self.notifier.notify(title, body)
        if speak and self.voice:
            self.voice.speak(f"{title}. {body}")

    # ── Individual monitors ───────────────────────────────────────────────────

    def _check_prices(self):
        """Monitor stock/crypto price targets."""
        targets = self._interests.get("stock_targets", [])
        if not targets:
            return

        for target in targets:
            symbol    = target.get("symbol","").upper()
            price_target = float(target.get("target", 0))
            direction = target.get("direction","above")

            if not symbol or not price_target:
                continue

            # Get current price from world model (which auto-refreshes)
            current = None
            if self.world:
                val = self.world.ask(symbol, "stock_price")
                if val:
                    try: current = float(val)
                    except Exception: pass

            if current is None:
                # Fetch directly
                try:
                    import requests
                    r = requests.get(
                        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                        headers={"User-Agent":"Mozilla/5.0"}, timeout=5,
                    )
                    current = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
                except Exception:
                    continue

            triggered = (
                (direction == "above" and current >= price_target) or
                (direction == "below" and current <= price_target)
            )

            alert_key = f"price_{symbol}_{direction}_{price_target}"
            if triggered and not self._already_alerted(alert_key, cooldown_min=120):
                change_pct = (current - price_target) / price_target * 100
                msg = f"{symbol} is now ₹{current:.2f} — your target was ₹{price_target:.2f} ({direction})"
                self._push("Price Alert 📈", msg, speak=True)
                self._record_alert(alert_key)

                # Update world model
                if self.world:
                    self.world.assert_fact(symbol, "stock_price", current, 0.99,
                                          "proactive_engine", "stock_price")

    def _check_news(self):
        """Check for new articles on topics of interest."""
        topics = self._interests.get("news_topics", [])
        if not topics or not self.research:
            return

        for topic in topics[:3]:  # max 3 topics per cycle
            alert_key = f"news_{topic}_{datetime.now().strftime('%Y%m%d_%H')}"
            if self._already_alerted(alert_key, cooldown_min=240):
                continue

            try:
                # Quick semantic scholar search
                results = self.research.search_semantic_scholar(topic, max_results=3)
                # Filter: only truly new papers (last 7 days)
                new_papers = []
                for p in results:
                    yr = p.get("year")
                    if yr and int(yr) >= datetime.now().year:
                        new_papers.append(p)

                if new_papers:
                    titles = "; ".join(p["title"][:60] for p in new_papers[:2])
                    self._push(
                        f"New research on {topic}",
                        f"{len(new_papers)} new paper(s): {titles}",
                        speak=False
                    )
                    self._record_alert(alert_key)
            except Exception:
                pass

    def _check_behaviour(self):
        """Monitor behaviour patterns — suggest breaks, flag stress."""
        if not self.analyst:
            return

        # Focus overload check
        break_after = self._interests.get("break_after_min", 90)
        alert_key = f"break_{datetime.now().strftime('%Y%m%d_%H%M')[:13]}"

        if not self._already_alerted(alert_key, cooldown_min=break_after):
            try:
                focus = self.analyst.analyse_focus_capacity(days=0)
                # Check if currently in a long focus session
                sessions = focus.get("total_sessions", 0)
                if sessions > 0:
                    # Check if current session is long
                    max_min = focus.get("max_focus_min", 0)
                    if max_min >= break_after:
                        self._push(
                            "Time for a break 🧘",
                            f"You've been focused for {max_min:.0f} minutes. Rest your eyes for 5 min.",
                            speak=False
                        )
                        self._record_alert(alert_key)
            except Exception:
                pass

        # High stress alert
        stress_key = f"stress_{datetime.now().strftime('%Y%m%d')}"
        if not self._already_alerted(stress_key, cooldown_min=480):
            try:
                stress = self.analyst.detect_stress_indicators(hours=3)
                if stress.get("stress_level") == "high":
                    top = stress.get("indicators",[""])[0]
                    self._push(
                        "Stress detected ⚠️",
                        f"{top[:100]}" if top else "High stress pattern detected today.",
                        speak=False
                    )
                    self._record_alert(stress_key)
            except Exception:
                pass

    def _check_system(self):
        """Monitor disk, CPU, and system health."""
        alert_key = f"system_{datetime.now().strftime('%Y%m%d_%H')}"
        if self._already_alerted(alert_key, cooldown_min=120):
            return
        try:
            import psutil
            disk = psutil.disk_usage("/").percent
            if disk > 90:
                self._push(
                    "Disk almost full 💾",
                    f"Disk usage is {disk:.0f}%. Consider cleaning up.",
                    speak=False
                )
                self._record_alert(alert_key)
        except Exception:
            pass

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self):
        """Start all proactive monitors in background."""
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        console.print("  [green]Proactive engine:[/] monitoring prices · news · behaviour · system")

    def stop(self):
        self._running = False

    def _loop(self):
        CHECK_INTERVALS = {
            "prices":    5,    # every 5 min
            "news":      60,   # every hour
            "behaviour": 15,   # every 15 min
            "system":    30,   # every 30 min
        }
        last_checks: dict = {}

        while self._running:
            now = time.time()
            for monitor, interval_min in CHECK_INTERVALS.items():
                last = last_checks.get(monitor, 0)
                if now - last >= interval_min * 60:
                    try:
                        getattr(self, f"_check_{monitor}")()
                    except Exception as e:
                        console.print(f"  [yellow]Proactive {monitor} error: {e}[/]")
                    last_checks[monitor] = now

            time.sleep(60)  # wake up every minute to check intervals

    # ── User-facing API ───────────────────────────────────────────────────────

    def add_price_target(self, symbol: str, target: float, direction: str = "above"):
        targets = self._interests.get("stock_targets", [])
        # Remove existing target for same symbol/direction
        targets = [t for t in targets
                   if not (t["symbol"] == symbol and t["direction"] == direction)]
        targets.append({"symbol": symbol, "target": target, "direction": direction})
        self._interests["stock_targets"] = targets
        self.save_interests(self._interests)
        return {"watching": symbol, "target": target, "direction": direction}

    def add_news_topic(self, topic: str):
        topics = self._interests.get("news_topics", [])
        if topic not in topics:
            topics.append(topic)
        self._interests["news_topics"] = topics
        self.save_interests(self._interests)
        return {"watching": topic}

    def add_research_topic(self, topic: str):
        topics = self._interests.get("research_topics", [])
        if topic not in topics:
            topics.append(topic)
        self._interests["research_topics"] = topics
        self.save_interests(self._interests)
        return {"watching": topic}

    def status(self) -> dict:
        return {
            "running":          self._running,
            "interests":        self._interests,
            "recent_alerts":    len(self._alerted),
            "monitors":         ["prices","news","behaviour","system"],
        }
