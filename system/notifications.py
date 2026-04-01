"""
ARIA — Notification Bridge
============================
Connects ARIA's internal alert system to:
  1. Tauri native OS notifications (desktop app)
  2. Browser Web Notifications API (PWA / browser)
  3. WebSocket push (any connected client)

How it works:
  Server side:  NotificationManager puts alerts in an asyncio.Queue
  Client side:  React subscribes to GET /api/notifications/stream (SSE)
                On each notification, calls Tauri or Web Notifications API

The SSE stream is a long-lived HTTP connection that stays open.
The client reconnects automatically if it drops.
This replaces polling — notifications arrive within milliseconds.

Notification sources that feed into this:
  - ProactiveEngine  (price alerts, news, behaviour)
  - TaskScheduler    (alarms, reminders, birthdays)
  - SelfImprovementEngine (security issues, model updates)
  - SystemMonitorAgent (disk full, CPU spike)
"""

import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NotificationManager:
    """
    Central hub for all ARIA notifications.
    Thread-safe. Works with both sync (agents) and async (FastAPI) code.
    """

    def __init__(self):
        self._queues:   list[asyncio.Queue] = []   # one per connected SSE client
        self._history:  list[dict]          = []   # last 50 notifications
        self._loop:     asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop (call once at server startup)."""
        self._loop = loop

    def notify(self, title: str, body: str, type_: str = "info",
               action: str = None, data: dict = None):
        """
        Send a notification to all connected clients.
        Can be called from any thread — sync agents, background threads, etc.
        """
        notification = {
            "id":     f"notif_{int(datetime.now().timestamp()*1000)}",
            "title":  title,
            "body":   body,
            "type":   type_,     # info | success | warning | error
            "action": action,    # optional deeplink: "/voice", "/analytics", etc.
            "data":   data or {},
            "ts":     datetime.now().isoformat(),
        }

        # Store in history
        self._history.append(notification)
        if len(self._history) > 50:
            self._history = self._history[-50:]

        # Push to all SSE clients
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._broadcast(notification), self._loop
            )

        from rich.console import Console
        Console().print(f"  [green]Notify:[/] {title} — {body[:60]}")

    async def _broadcast(self, notification: dict):
        """Push notification to all connected SSE clients."""
        dead = []
        for q in self._queues:
            try:
                await q.put(notification)
            except Exception:
                dead.append(q)
        for q in dead:
            try: self._queues.remove(q)
            except ValueError: pass

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """
        FastAPI SSE generator. Each call creates a new subscriber.
        Yields server-sent events as JSON.
        """
        q = asyncio.Queue(maxsize=50)
        self._queues.append(q)

        # Send last 5 notifications immediately (catch up)
        for n in self._history[-5:]:
            yield f"data: {json.dumps(n)}\n\n"

        # Keep-alive ping every 30s + real notifications as they arrive
        try:
            while True:
                try:
                    notification = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(notification)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive
                    yield f"data: {json.dumps({'type': 'ping', 'ts': datetime.now().isoformat()})}\n\n"
        except GeneratorExit:
            pass
        finally:
            try: self._queues.remove(q)
            except ValueError: pass

    def get_history(self, limit: int = 20) -> list[dict]:
        return self._history[-limit:]


# Global instance — imported by server.py
notification_manager = NotificationManager()
