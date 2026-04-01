"""
ARIA Offline Mode Manager
Detects internet connectivity and manages graceful degradation to local-only mode.
No external dependencies beyond stdlib + requests (already in requirements).
"""

import threading
import time
import socket
import logging
from enum import Enum
from typing import Callable, List, Optional
from datetime import datetime

try:
    import requests as _requests
    _REQUESTS = True
except ImportError:
    _requests = None
    _REQUESTS = False

logger = logging.getLogger("aria.offline")


class ConnectivityState(str, Enum):
    ONLINE      = "online"       # full internet access
    DEGRADED    = "degraded"     # partial — some sources reachable
    OFFLINE     = "offline"      # no internet
    UNKNOWN     = "unknown"      # not yet checked


# Probe targets: lightweight, reliable, globally distributed
_PROBES = [
    ("8.8.8.8",         53,  "Google DNS"),
    ("1.1.1.1",         53,  "Cloudflare DNS"),
    ("208.67.222.222",  53,  "OpenDNS"),
    ("9.9.9.9",         53,  "Quad9 DNS"),
]

_HTTP_PROBES = [
    "https://www.google.com/generate_204",
    "https://connectivity-check.ubuntu.com",
    "https://captive.apple.com/hotspot-detect.html",
]


class OfflineModeManager:
    """
    Monitors internet connectivity and exposes state + callbacks.

    Usage:
        mgr = OfflineModeManager()
        mgr.on_state_change(lambda old, new: print(f"{old} -> {new}"))
        mgr.start()
        if mgr.is_online:
            ...
    """

    def __init__(
        self,
        check_interval: int = 30,
        fast_interval: int = 5,
        tcp_timeout: float = 2.0,
        http_timeout: float = 4.0,
    ):
        self.check_interval   = check_interval   # seconds between checks when stable
        self.fast_interval    = fast_interval    # seconds between checks during recovery
        self.tcp_timeout      = tcp_timeout
        self.http_timeout     = http_timeout

        self._state           = ConnectivityState.UNKNOWN
        self._prev_state      = ConnectivityState.UNKNOWN
        self._lock            = threading.Lock()
        self._callbacks: List[Callable] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event      = threading.Event()
        self._last_check: Optional[datetime] = None
        self._consecutive_failures = 0
        self._reachable_count = 0   # how many probes passed last check

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> ConnectivityState:
        with self._lock:
            return self._state

    @property
    def is_online(self) -> bool:
        return self.state in (ConnectivityState.ONLINE, ConnectivityState.DEGRADED)

    @property
    def is_fully_online(self) -> bool:
        return self.state == ConnectivityState.ONLINE

    @property
    def is_offline(self) -> bool:
        return self.state == ConnectivityState.OFFLINE

    def on_state_change(self, callback: Callable[[str, str], None]) -> None:
        """Register a callback: fn(old_state: str, new_state: str)"""
        self._callbacks.append(callback)

    def check_now(self) -> ConnectivityState:
        """Run a connectivity check immediately (blocking, thread-safe)."""
        new_state = self._do_check()
        self._update_state(new_state)
        return new_state

    def start(self, run_immediately: bool = True) -> None:
        """Start background monitoring thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        if run_immediately:
            self.check_now()
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="aria-offline-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Offline monitor started — state={self._state.value}")

    def stop(self) -> None:
        """Stop background monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def status_dict(self) -> dict:
        with self._lock:
            return {
                "state":               self._state.value,
                "is_online":           self.is_online,
                "reachable_probes":    self._reachable_count,
                "total_probes":        len(_PROBES),
                "consecutive_failures": self._consecutive_failures,
                "last_check":          self._last_check.isoformat() if self._last_check else None,
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_check(self) -> ConnectivityState:
        """Run TCP + optional HTTP probes. Returns new state."""
        passed = 0
        for host, port, name in _PROBES:
            try:
                with socket.create_connection((host, port), timeout=self.tcp_timeout):
                    passed += 1
            except OSError:
                pass

        with self._lock:
            self._reachable_count = passed
        self._last_check = datetime.now()

        if passed == 0:
            return ConnectivityState.OFFLINE
        if passed < len(_PROBES) // 2:
            return ConnectivityState.DEGRADED

        # All or most DNS probes pass — do a quick HTTP captive-portal check
        if _REQUESTS:
            for url in _HTTP_PROBES:
                try:
                    r = _requests.get(url, timeout=self.http_timeout, allow_redirects=False)
                    if r.status_code in (200, 204, 301, 302):
                        return ConnectivityState.ONLINE
                except Exception:
                    continue
            # HTTP all failed but DNS worked — probably captive portal / restricted
            return ConnectivityState.DEGRADED

        return ConnectivityState.ONLINE  # requests not installed, trust DNS

    def _update_state(self, new_state: ConnectivityState) -> None:
        with self._lock:
            old_state = self._state
            self._state = new_state
            if new_state == ConnectivityState.OFFLINE:
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

        if new_state != old_state:
            self._prev_state = old_state
            logger.info(f"Connectivity: {old_state.value} -> {new_state.value}")
            for cb in self._callbacks:
                try:
                    cb(old_state.value, new_state.value)
                except Exception as e:
                    logger.warning(f"Offline callback error: {e}")

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            # Use fast interval during degraded/offline to recover quickly
            interval = (
                self.fast_interval
                if self._state in (ConnectivityState.OFFLINE, ConnectivityState.DEGRADED)
                else self.check_interval
            )
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                break
            try:
                new_state = self._do_check()
                self._update_state(new_state)
            except Exception as e:
                logger.error(f"Offline monitor error: {e}")


# ── Agent capability registry for offline degradation ────────────────────────

class OfflineCapabilityFilter:
    """
    Maps each ARIA capability to whether it needs internet.
    Agents can query this to decide whether to attempt network calls.
    """

    # capability -> requires_internet
    _CAPS = {
        "web_search":          True,
        "research_search":     True,
        "news":                True,
        "stock_data":          True,
        "weather":             True,
        "translation":         True,
        "pubmed_search":       True,
        "fda_drugs":           True,
        "llm_cloud":           True,     # cloud LLMs
        "llm_local":           False,    # Ollama
        "memory":              False,
        "voice_stt":           False,    # Whisper local
        "voice_tts":           False,    # pyttsx3/edge-tts cached
        "computer_control":    False,
        "file_management":     False,
        "code_execution":      False,
        "calendar":            False,
        "system_monitor":      False,
        "automation":          False,
        "media_control":       False,
        "ocr":                 False,
        "document_processing": False,
        "chat":                False,    # local LLM only
    }

    def __init__(self, manager: OfflineModeManager):
        self._mgr = manager

    def can_use(self, capability: str) -> bool:
        """Returns True if capability is usable in current connectivity state."""
        needs_net = self._CAPS.get(capability, True)
        if not needs_net:
            return True
        return self._mgr.is_online

    def available_capabilities(self) -> dict:
        return {
            cap: self.can_use(cap)
            for cap in self._CAPS
        }

    def offline_message(self, capability: str) -> str:
        if self.can_use(capability):
            return ""
        state = self._mgr.state.value
        return (
            f"'{capability}' requires internet access but ARIA is currently {state}. "
            "Using local fallback if available."
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[OfflineModeManager] = None
_filter:  Optional[OfflineCapabilityFilter] = None


def get_manager() -> OfflineModeManager:
    global _manager
    if _manager is None:
        _manager = OfflineModeManager()
    return _manager


def get_filter() -> OfflineCapabilityFilter:
    global _filter
    if _filter is None:
        _filter = OfflineCapabilityFilter(get_manager())
    return _filter


def is_online() -> bool:
    return get_manager().is_online


def can_use(capability: str) -> bool:
    return get_filter().can_use(capability)
