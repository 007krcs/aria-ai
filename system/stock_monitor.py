"""
ARIA Stock Monitor — 24/7 Background Investment Watchlist
==========================================================

Runs as a background asyncio task inside the ARIA server process.
Every N minutes it:
  1. Re-runs InvestmentTimingAgent.analyze() for each watched symbol
  2. Compares current signal to previous signal
  3. Fires push notification when:
     - Signal changes (HOLD → INVEST_NOW, WATCH → EXIT_NOW, etc.)
     - Score crosses a threshold (>60 or <-60)
     - RSI/MACD crossover detected
  4. Persists watchlist + last signals in data/watchlist.json

Usage (server.py):
    from system.stock_monitor import StockMonitor
    monitor = StockMonitor(timing_agent=timing_agent, notification_manager=notif_mgr)
    monitor.start()   # starts background asyncio task
    monitor.add("RELIANCE")
    monitor.add("TCS")
    monitor.remove("RELIANCE")
    monitor.list()    # returns current watchlist
    monitor.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.json"
WATCHLIST_FILE.parent.mkdir(exist_ok=True)


# ── Signal change severity mapping ───────────────────────────────────────────
SIGNAL_RANK = {
    "INVEST_NOW":  3,
    "ACCUMULATE":  2,
    "WATCH":       1,
    "HOLD":        0,
    "REDUCE":     -1,
    "SELL":       -2,
    "EXIT_NOW":   -3,
}

# Notify only when rank changes by this much
NOTIFY_RANK_DELTA = 1   # any signal change
NOTIFY_SCORE_THRESHOLD = 60  # absolute invest/exit zone


@dataclass
class WatchedSymbol:
    symbol:         str
    asset_type:     str = "auto"
    last_signal:    Optional[str] = None
    last_score:     Optional[float] = None
    last_checked:   Optional[str] = None
    push_enabled:   bool = True
    check_count:    int = 0
    notify_count:   int = 0


@dataclass
class MonitorState:
    symbols:      dict[str, WatchedSymbol] = field(default_factory=dict)
    interval_min: int = 15    # how often to check each symbol
    running:      bool = False
    started_at:   Optional[str] = None
    total_checks: int = 0
    total_alerts: int = 0


class StockMonitor:
    """
    Background 24/7 watchlist monitor.
    Runs inside the ARIA server event loop.
    """

    def __init__(
        self,
        timing_agent=None,          # InvestmentTimingAgent instance
        notification_manager=None,  # ARIA NotificationManager
        interval_minutes: int = 15,
    ):
        self._agent  = timing_agent
        self._notif  = notification_manager
        self._state  = MonitorState(interval_min=interval_minutes)
        self._task: Optional[asyncio.Task] = None
        self._load_watchlist()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start the background monitoring loop."""
        if self._task and not self._task.done():
            logger.info("StockMonitor already running")
            return
        self._state.running   = True
        self._state.started_at = datetime.now().isoformat()
        try:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._monitor_loop())
            logger.info("StockMonitor started (interval=%dm, symbols=%d)",
                        self._state.interval_min, len(self._state.symbols))
        except RuntimeError:
            # No running event loop — will be started via asyncio.run_coroutine_threadsafe
            logger.warning("StockMonitor: no running loop — call monitor.start() after server startup")

    def stop(self):
        """Stop the monitoring loop."""
        self._state.running = False
        if self._task:
            self._task.cancel()
        logger.info("StockMonitor stopped")

    def add(self, symbol: str, asset_type: str = "auto", push: bool = True) -> dict:
        """Add a symbol to the watchlist."""
        key = symbol.upper().strip()
        if key in self._state.symbols:
            return {"ok": False, "message": f"{key} already in watchlist"}
        self._state.symbols[key] = WatchedSymbol(
            symbol=key, asset_type=asset_type, push_enabled=push
        )
        self._save_watchlist()
        logger.info("StockMonitor: added %s", key)
        return {"ok": True, "message": f"{key} added to watchlist", "symbol": key}

    def remove(self, symbol: str) -> dict:
        """Remove a symbol from the watchlist."""
        key = symbol.upper().strip()
        if key not in self._state.symbols:
            return {"ok": False, "message": f"{key} not in watchlist"}
        del self._state.symbols[key]
        self._save_watchlist()
        return {"ok": True, "message": f"{key} removed from watchlist"}

    def list_symbols(self) -> list[dict]:
        """Return current watchlist with last signals."""
        return [
            {
                "symbol":       s.symbol,
                "asset_type":   s.asset_type,
                "last_signal":  s.last_signal,
                "last_score":   s.last_score,
                "last_checked": s.last_checked,
                "push_enabled": s.push_enabled,
                "check_count":  s.check_count,
                "notify_count": s.notify_count,
            }
            for s in self._state.symbols.values()
        ]

    def status(self) -> dict:
        """Return monitor health status."""
        return {
            "running":        self._state.running,
            "started_at":     self._state.started_at,
            "interval_min":   self._state.interval_min,
            "symbol_count":   len(self._state.symbols),
            "total_checks":   self._state.total_checks,
            "total_alerts":   self._state.total_alerts,
            "symbols":        self.list_symbols(),
        }

    def set_interval(self, minutes: int):
        """Change check interval (takes effect on next cycle)."""
        self._state.interval_min = max(1, minutes)
        self._save_watchlist()

    # ── Background loop ───────────────────────────────────────────────────────

    async def _monitor_loop(self):
        """Main background loop — runs until stopped."""
        logger.info("StockMonitor loop starting")
        while self._state.running:
            if not self._state.symbols:
                await asyncio.sleep(60)
                continue

            symbols = list(self._state.symbols.keys())
            logger.debug("StockMonitor checking %d symbols: %s", len(symbols), symbols)

            for sym in symbols:
                if not self._state.running:
                    break
                try:
                    await self._check_symbol(sym)
                    self._state.total_checks += 1
                    # Brief pause between symbols to not hammer Yahoo Finance
                    await asyncio.sleep(3)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    logger.error("StockMonitor error checking %s: %s", sym, e)

            self._save_watchlist()
            # Wait for next cycle
            sleep_seconds = self._state.interval_min * 60
            logger.debug("StockMonitor sleeping %ds until next cycle", sleep_seconds)
            await asyncio.sleep(sleep_seconds)

    async def _check_symbol(self, symbol: str):
        """Check one symbol, compare with last signal, fire notification if needed."""
        if not self._agent:
            return

        entry = self._state.symbols.get(symbol)
        if not entry:
            return

        try:
            signal = await self._agent.analyze(symbol, entry.asset_type)
        except Exception as e:
            logger.warning("StockMonitor: failed to analyze %s: %s", symbol, e)
            return

        prev_signal = entry.last_signal
        prev_score  = entry.last_score or 0.0

        # Update state
        entry.last_signal  = signal.signal
        entry.last_score   = signal.timing_score
        entry.last_checked = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry.check_count += 1

        # Determine if push notification needed
        should_notify = self._should_notify(
            prev_signal, signal.signal,
            prev_score,  signal.timing_score,
            signal.push_trigger,
            entry.push_enabled,
        )

        if should_notify:
            await self._fire_notification(entry, signal, prev_signal)
            entry.notify_count += 1
            self._state.total_alerts += 1

        logger.info(
            "StockMonitor %s: %s (score=%.1f, prev=%s, notify=%s)",
            symbol, signal.signal, signal.timing_score, prev_signal, should_notify
        )

    def _should_notify(
        self,
        prev_signal: Optional[str],
        curr_signal: str,
        prev_score:  float,
        curr_score:  float,
        push_trigger: bool,
        push_enabled: bool,
    ) -> bool:
        """Decide whether to fire a push notification."""
        if not push_enabled:
            return False

        # Always notify on first check if signal is actionable
        if prev_signal is None:
            return abs(curr_score) >= NOTIFY_SCORE_THRESHOLD

        # Signal rank changed meaningfully
        prev_rank = SIGNAL_RANK.get(prev_signal, 0)
        curr_rank = SIGNAL_RANK.get(curr_signal, 0)
        if abs(curr_rank - prev_rank) >= NOTIFY_RANK_DELTA and curr_rank != 0:
            return True

        # Score crossed invest/exit zone
        if abs(prev_score) < NOTIFY_SCORE_THRESHOLD <= abs(curr_score):
            return True

        # Agent-detected trigger (MACD/RSI crossover etc.)
        if push_trigger:
            return True

        return False

    async def _fire_notification(self, entry: WatchedSymbol, signal, prev_signal: Optional[str]):
        """Send push notification via ARIA notification manager."""
        # Determine urgency
        urgency = "info"
        if signal.signal in ("INVEST_NOW", "EXIT_NOW"):
            urgency = "warning"
        elif signal.signal in ("ACCUMULATE", "SELL"):
            urgency = "info"

        emoji_map = {
            "INVEST_NOW": "🟢", "ACCUMULATE": "🔵", "WATCH": "🔵",
            "HOLD": "⚪", "REDUCE": "🟡", "SELL": "🟠", "EXIT_NOW": "🔴",
        }
        emoji = emoji_map.get(signal.signal, "📈")

        title = f"{emoji} {entry.symbol} — {signal.signal}"

        lines = [
            f"Score: {signal.timing_score:+.1f}/100  |  Confidence: {signal.confidence:.0f}%",
            f"Price: ₹{signal.current_price:.2f}" if ".NS" in entry.symbol or ".BO" in entry.symbol
            else f"Price: {signal.current_price:.2f}",
        ]
        if signal.stop_loss and signal.targets:
            lines.append(f"SL: {signal.stop_loss:.2f}  |  T1: {signal.targets[0]:.2f}")
        if prev_signal and prev_signal != signal.signal:
            lines.append(f"Changed from: {prev_signal} → {signal.signal}")
        if signal.why_moving:
            lines.append(f"Why: {signal.why_moving[0]}")

        body = "\n".join(lines)

        if self._notif:
            try:
                await self._notif.send(title=title, body=body, type=urgency)
                logger.info("StockMonitor: notification sent for %s (%s)", entry.symbol, signal.signal)
            except Exception as e:
                logger.error("StockMonitor notification error: %s", e)
        else:
            logger.info("PUSH NOTIFICATION (no manager): %s — %s", title, body)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_watchlist(self):
        """Save watchlist to data/watchlist.json."""
        try:
            data = {
                "interval_min": self._state.interval_min,
                "symbols": {
                    k: {
                        "symbol":       v.symbol,
                        "asset_type":   v.asset_type,
                        "last_signal":  v.last_signal,
                        "last_score":   v.last_score,
                        "last_checked": v.last_checked,
                        "push_enabled": v.push_enabled,
                        "check_count":  v.check_count,
                        "notify_count": v.notify_count,
                    }
                    for k, v in self._state.symbols.items()
                },
            }
            WATCHLIST_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("StockMonitor: failed to save watchlist: %s", e)

    def _load_watchlist(self):
        """Load watchlist from data/watchlist.json."""
        try:
            if not WATCHLIST_FILE.exists():
                return
            data = json.loads(WATCHLIST_FILE.read_text())
            self._state.interval_min = data.get("interval_min", 15)
            for sym, entry in data.get("symbols", {}).items():
                self._state.symbols[sym] = WatchedSymbol(**{
                    k: v for k, v in entry.items()
                    if k in WatchedSymbol.__dataclass_fields__
                })
            logger.info("StockMonitor: loaded %d symbols from watchlist", len(self._state.symbols))
        except Exception as e:
            logger.error("StockMonitor: failed to load watchlist: %s", e)
