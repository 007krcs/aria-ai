"""
ARIA — Agent Event Bus
========================
The nervous system of ARIA's multi-agent system.
Agents publish events and subscribe to results.
All communication is async, real-time, and in-memory.

Example flow:
  User says: "Remind me when Apple hits ₹180"

  1. IntentRouter publishes: {type: "price_alert", symbol: "AAPL", target: 180}
  2. SearchAgent subscribes to "price_alert" → starts monitoring
  3. SchedulerAgent subscribes to "price_alert" → persists the alert
  4. When price hits: SearchAgent publishes {type: "alert_triggered", ...}
  5. CommsAgent subscribes to "alert_triggered" → sends notification
  6. VoiceAgent subscribes to "alert_triggered" → speaks the alert

Every agent is independent. They don't call each other directly.
They only communicate via the bus — fully decoupled.
"""

import re
import asyncio
import json
import time
import uuid
import threading
from collections import defaultdict
from datetime import datetime
from typing import Callable, Any, Optional
from rich.console import Console

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# EVENT
# ─────────────────────────────────────────────────────────────────────────────

class Event:
    """A message on the agent bus."""
    def __init__(
        self,
        event_type: str,
        data:       dict,
        source:     str = "system",
        reply_to:   str = None,
    ):
        self.id         = str(uuid.uuid4())[:8]
        self.type       = event_type
        self.data       = data
        self.source     = source
        self.reply_to   = reply_to          # ID of event this is responding to
        self.ts         = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "id":       self.id,
            "type":     self.type,
            "data":     self.data,
            "source":   self.source,
            "reply_to": self.reply_to,
            "ts":       self.ts,
        }

    def __repr__(self):
        return f"Event({self.type}, source={self.source}, id={self.id})"


# ─────────────────────────────────────────────────────────────────────────────
# AGENT BUS
# ─────────────────────────────────────────────────────────────────────────────

class AgentBus:
    """
    Central event bus for all ARIA agents.
    Thread-safe. Supports both sync and async subscribers.
    Keeps an audit log of all events.

    Usage:
        bus = AgentBus()

        # Subscribe
        def on_alarm(event):
            print(f"Alarm: {event.data}")
        bus.subscribe("alarm_triggered", on_alarm)

        # Publish
        bus.publish(Event("alarm_triggered", {"time": "07:00"}))

        # Request-response pattern
        result = bus.request("search_web", {"query": "Apple stock"}, timeout=10)
    """

    def __init__(self):
        self._subscribers:  dict[str, list[Callable]] = defaultdict(list)
        self._wildcard:     list[Callable]             = []   # receives ALL events
        self._history:      list[dict]                 = []
        self._lock          = threading.Lock()
        self._max_history   = 500

    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to a specific event type."""
        with self._lock:
            self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: Callable):
        """Subscribe to ALL events (for logging, monitoring)."""
        with self._lock:
            self._wildcard.append(handler)

    def unsubscribe(self, event_type: str, handler: Callable):
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)

    def publish(self, event: Event) -> int:
        """
        Publish an event to all subscribers.
        Returns number of handlers that received it.
        """
        # Log
        with self._lock:
            self._history.append(event.to_dict())
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            handlers = list(self._subscribers.get(event.type, []))
            wildcards = list(self._wildcard)

        count = 0
        for handler in handlers + wildcards:
            try:
                handler(event)
                count += 1
            except Exception as e:
                console.print(f"  [yellow]Bus handler error ({handler.__name__}): {e}[/]")

        return count

    def request(
        self,
        event_type: str,
        data:       dict,
        timeout:    float = 10.0,
        source:     str   = "system",
    ) -> Optional[dict]:
        """
        Publish an event and wait for a response.
        Response must be published with reply_to = original event ID.
        """
        event    = Event(event_type, data, source)
        result   = [None]
        received = threading.Event()

        def on_response(resp_event: Event):
            if resp_event.reply_to == event.id:
                result[0] = resp_event.data
                received.set()

        self.subscribe(f"{event_type}_response", on_response)
        self.publish(event)

        got_response = received.wait(timeout=timeout)
        self.unsubscribe(f"{event_type}_response", on_response)

        return result[0] if got_response else None

    def get_history(self, event_type: str = None, limit: int = 50) -> list[dict]:
        """Get recent event history, optionally filtered by type."""
        with self._lock:
            history = list(self._history)
        if event_type:
            history = [e for e in history if e["type"] == event_type]
        return history[-limit:]

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_events":    len(self._history),
                "subscribers":     {k: len(v) for k, v in self._subscribers.items()},
                "event_types":     list(set(e["type"] for e in self._history[-100:])),
            }


# ─────────────────────────────────────────────────────────────────────────────
# INTENT ROUTER
# Classifies natural language → specific agent action
# ─────────────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Classifies user intent and routes to the right agent(s).
    No model needed for most commands — pure pattern matching.
    Falls back to LLM for ambiguous cases.
    """

    # Intent patterns: (regex, event_type, field_extractors)
    PATTERNS = [
        # Phone calls
        (r"\bcall\b.+",                         "make_call",      {"contact": r"call\s+(.+?)(?:\s+on|$)"}),
        (r"\bdial\b.+",                         "make_call",      {"contact": r"dial\s+(.+)"}),
        (r"\bphone\b.+",                        "make_call",      {"contact": r"phone\s+(.+)"}),

        # Messages
        (r"\b(send|write|compose)\s+(a\s+)?message\b", "send_message", {}),
        (r"\b(text|whatsapp|sms)\b.+",         "send_message",   {"contact": r"(?:text|whatsapp|sms)\s+(.+?)(?:\s+saying|\s+that|\s+with|$)"}),
        (r"\b(wish|wish\s+happy)\s+birthday\b", "send_birthday",  {"contact": r"(?:wish|to)\s+(.+?)(?:\s+happy|$)"}),

        # Email
        (r"\b(send|reply|write|compose)\s+(an?\s+)?email\b", "send_email", {}),
        (r"\breply\s+to\b.+",                  "send_email",     {"contact": r"reply\s+to\s+(.+)"}),

        # Alarms & Reminders
        (r"\bset\s+(an?\s+)?alarm\b",           "set_alarm",      {"time": r"alarm\s+(?:at\s+)?(.+?)(?:\s+for|$)"}),
        (r"\bwake\s+me\b",                      "set_alarm",      {"time": r"wake\s+me\s+(?:up\s+)?(?:at\s+)?(.+)"}),
        (r"\bremind\s+me\b",                    "set_reminder",   {"text": r"remind\s+me\s+(?:to\s+)?(.+?)(?:\s+at\s+|\s+in\s+|$)",
                                                                   "time": r"(?:at|in)\s+(.+?)$"}),

        # Search & research
        (r"\b(search|look up|find|google)\s+(.+)",  "web_search", {"query": r"(?:search|look up|find|google)\s+(.+)"}),
        (r"\bmarket research\s+(?:for\s+)?(.+)",    "market_research", {"company": r"market research\s+(?:for\s+)?(.+)"}),
        (r"\b(news|latest news)\b",             "get_news",       {"topic": r"(?:news|latest news)(?:\s+about\s+|\s+on\s+)?(.+)?"}),

        # Stock / price alerts
        (r"\b(?:stock|price|share)\s+(?:of\s+)?(.+)",  "stock_price",  {"symbol": r"(?:stock|price|share)\s+(?:of\s+)?(.+?)(?:\s+price|$)"}),
        (r"\bremind\s+me\s+when\s+(.+)\s+(?:hits|reaches|goes to)\s+(.+)",
                                                "price_alert",    {"symbol": r"when\s+(.+?)\s+(?:hits|reaches)",
                                                                   "target": r"(?:hits|reaches|goes to)\s+(.+)"}),

        # Media
        (r"\bplay\s+(.+)\s+on\s+youtube\b",    "play_youtube",   {"query": r"play\s+(.+)\s+on\s+youtube"}),
        (r"\bopen\s+(.+)\s+on\s+youtube\b",    "play_youtube",   {"query": r"open\s+(.+)\s+on\s+youtube"}),
        (r"\bwatch\s+(.+)",                     "play_youtube",   {"query": r"watch\s+(.+)"}),
        (r"\bplay\s+(?:song\s+|music\s+)?(.+)", "play_music",    {"query": r"play\s+(?:song\s+|music\s+)?(.+)"}),
        (r"\bspotify\b.+",                      "play_music",     {"query": r"spotify\s+(.+)"}),

        # App control
        (r"\bopen\s+(.+)",                      "open_app",       {"app": r"open\s+(.+)"}),
        (r"\blaunch\s+(.+)",                    "open_app",       {"app": r"launch\s+(.+)"}),
        (r"\bclose\s+(.+)",                     "close_app",      {"app": r"close\s+(.+)"}),

        # Screenshot / screen
        (r"\b(screenshot|screen capture)\b",    "take_screenshot", {}),
        (r"\bwhat.s on (?:my )?screen\b",       "read_screen",    {}),
    ]

    def __init__(self, engine=None):
        self.engine  = engine
        import re as _re
        self._re     = _re
        self._compiled = [(self._re.compile(p, self._re.IGNORECASE), t, f)
                         for p, t, f in self.PATTERNS]

    def route(self, text: str) -> list[dict]:
        """
        Parse text into one or more action events.
        Returns list of {event_type, data} dicts.
        """
        text    = text.strip()
        results = []

        for pattern, event_type, field_extractors in self._compiled:
            if pattern.search(text):
                data = {"raw_text": text}
                for field, extractor_pattern in field_extractors.items():
                    m = self._re.search(extractor_pattern, text, self._re.IGNORECASE)
                    if m:
                        data[field] = m.group(1).strip()
                results.append({"event_type": event_type, "data": data})

        # Deduplicate by event_type
        seen   = set()
        unique = []
        for r in results:
            if r["event_type"] not in seen:
                seen.add(r["event_type"])
                unique.append(r)

        # LLM fallback for unrecognised intent
        if not unique and self.engine:
            return self._llm_route(text)

        return unique or [{"event_type": "general_query", "data": {"query": text}}]

    def _llm_route(self, text: str) -> list[dict]:
        """Use LLM to classify intent when patterns don't match."""
        prompt = (
            f"Classify this user command into one of these action types:\n"
            f"make_call, send_message, send_email, set_alarm, set_reminder,\n"
            f"web_search, market_research, stock_price, price_alert,\n"
            f"play_youtube, play_music, open_app, take_screenshot, general_query\n\n"
            f"Command: {text}\n\n"
            f"Return JSON: {{\"event_type\": \"...\", \"data\": {{...}}}}\n"
            f"JSON:"
        )
        try:
            raw  = self.engine.generate(prompt, temperature=0.1)
            import re
            raw  = re.sub(r"```\w*\n?|```","", raw).strip()
            data = json.loads(raw)
            return [data]
        except Exception:
            return [{"event_type": "general_query", "data": {"query": text}}]
