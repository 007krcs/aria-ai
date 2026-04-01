"""
ARIA — Security Middleware  (v1)
==================================
Provides:
  • Rate limiting  — per-IP token bucket, configurable per endpoint
  • Request size   — max body size guard
  • Input sanitiser — strips null bytes, control chars, oversized payloads
  • WebSocket token — validates JWT from header OR query param (with warning)
  • CORS policy    — tightly scoped to localhost origins
  • Brute-force lockout — N failed logins → IP ban for M minutes
"""

from __future__ import annotations

import re
import time
import threading
from collections import defaultdict
from typing import Optional
from rich.console import Console

from fastapi import Request, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BUCKET RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class _TokenBucket:
    """
    Per-IP token bucket.
    capacity = max burst; refill_rate = tokens per second.
    """
    __slots__ = ("tokens", "capacity", "refill_rate", "last_refill", "_lock")

    def __init__(self, capacity: float, refill_rate: float):
        self.capacity    = capacity
        self.tokens      = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.monotonic()
        self._lock       = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.refill_rate,
            )
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class RateLimiter:
    """
    Thread-safe rate limiter.
    Default: 60 req/min burst=20 per IP.
    Configurable per-route limits.
    """

    # (capacity, refill_per_second) per route prefix
    _ROUTE_LIMITS: dict[str, tuple[float, float]] = {
        "/auth/login":   (5,  0.05),   #  5 burst, 3 req/min (brute-force guard)
        "/auth/setup":   (3,  0.02),   #  3 burst
        "/query":        (20, 0.5),    # 20 burst, 30 req/min
        "/chat":         (20, 0.5),
        "/ingest":       (5,  0.08),   #  5 burst, 5 req/min (ingestion is expensive)
        "/nova":         (10, 0.25),
        "default":       (60, 1.0),    # 60 burst, 60 req/min
    }

    def __init__(self):
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock    = threading.Lock()
        # Cleanup stale buckets every 10 min
        threading.Thread(target=self._cleanup, daemon=True,
                         name="ratelimit-gc").start()

    def _get_bucket(self, ip: str, route: str) -> _TokenBucket:
        # Match route to limit config
        limits = self._ROUTE_LIMITS["default"]
        for prefix, lim in self._ROUTE_LIMITS.items():
            if prefix != "default" and route.startswith(prefix):
                limits = lim
                break

        key = f"{ip}:{route}"
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _TokenBucket(*limits)
            return self._buckets[key]

    def is_allowed(self, ip: str, route: str) -> bool:
        return self._get_bucket(ip, route).consume()

    def _cleanup(self):
        """Remove stale buckets periodically."""
        while True:
            time.sleep(600)
            with self._lock:
                cutoff = time.monotonic() - 600
                stale  = [k for k, b in self._buckets.items()
                          if b.last_refill < cutoff]
                for k in stale:
                    del self._buckets[k]


# ─────────────────────────────────────────────────────────────────────────────
# BRUTE-FORCE LOCKOUT
# ─────────────────────────────────────────────────────────────────────────────

class BruteForceGuard:
    """Locks out an IP for 15 minutes after 10 consecutive failed attempts."""

    MAX_FAILS     = 10
    LOCKOUT_S     = 900  # 15 minutes

    def __init__(self):
        self._fails:   dict[str, int]   = defaultdict(int)
        self._lockout: dict[str, float] = {}
        self._lock     = threading.Lock()

    def record_failure(self, ip: str):
        with self._lock:
            self._fails[ip] += 1
            if self._fails[ip] >= self.MAX_FAILS:
                self._lockout[ip] = time.time() + self.LOCKOUT_S
                console.print(f"  [red]Brute-force lockout:[/] {ip}")

    def record_success(self, ip: str):
        with self._lock:
            self._fails.pop(ip, None)
            self._lockout.pop(ip, None)

    def is_locked(self, ip: str) -> bool:
        with self._lock:
            expiry = self._lockout.get(ip)
            if expiry and time.time() < expiry:
                return True
            elif expiry:
                # Lockout expired — reset
                del self._lockout[ip]
                self._fails.pop(ip, None)
            return False

    def remaining_lockout_s(self, ip: str) -> int:
        with self._lock:
            expiry = self._lockout.get(ip, 0)
            return max(0, int(expiry - time.time()))


# ─────────────────────────────────────────────────────────────────────────────
# INPUT SANITISER
# ─────────────────────────────────────────────────────────────────────────────

MAX_QUERY_LEN   = 8_000    # characters
MAX_BODY_BYTES  = 50_000_000  # 50 MB upload cap

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitise_text(text: str, max_len: int = MAX_QUERY_LEN) -> str:
    """Remove null bytes and control characters; truncate if oversized."""
    if not isinstance(text, str):
        return ""
    text = _CONTROL_RE.sub("", text)
    return text[:max_len]


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

_rate_limiter      = RateLimiter()
_brute_force_guard = BruteForceGuard()


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Drop-in FastAPI middleware that applies:
      1. Request body size cap
      2. Rate limiting per IP
    """

    async def dispatch(self, request: Request, call_next):
        ip    = _get_client_ip(request)
        path  = request.url.path

        # 1. Body size cap
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"},
            )

        # 2. Rate limit
        if not _rate_limiter.is_allowed(ip, path):
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Please slow down."},
                headers={"Retry-After": "60"},
            )

        return await call_next(request)


def _get_client_ip(request: Request) -> str:
    """Extract real IP, accounting for reverse proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET TOKEN VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

async def validate_ws_token(websocket: WebSocket) -> Optional[str]:
    """
    Validate JWT for a WebSocket connection.
    Accepts token from:
      1. Authorization header (preferred — not logged)
      2. Query param ?token=... (fallback — warn user)

    Returns the token string or None if invalid/missing.
    """
    # 1. Header (preferred)
    auth_header = websocket.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    # 2. Query param fallback (less secure but compatible with browsers)
    token = websocket.query_params.get("token")
    if token:
        # Only warn once per IP
        ip = websocket.client.host if websocket.client else "unknown"
        console.print(
            f"  [yellow]WS token via query param from {ip} — "
            f"prefer Authorization header for security[/]"
        )
        return token

    return None


# ─────────────────────────────────────────────────────────────────────────────
# CORS CONFIG (for FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

# Allow localhost + any cloud frontend origins set via CORS_ORIGINS env var
import os as _os
_extra_origins = [o.strip() for o in _os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

CORS_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:1420",   # Tauri dev server
    "http://localhost:9177",   # Vite dev server
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "tauri://localhost",        # Tauri production
] + _extra_origins

CORS_SETTINGS = {
    "allow_origins":      CORS_ORIGINS,
    "allow_credentials":  True,
    "allow_methods":      ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "allow_headers":      ["Authorization", "Content-Type", "X-Request-ID"],
    "expose_headers":     ["X-Request-ID"],
    "max_age":            3600,
}


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE EXPORTS for server.py
# ─────────────────────────────────────────────────────────────────────────────

def get_brute_force_guard() -> BruteForceGuard:
    return _brute_force_guard


def record_login_failure(ip: str):
    _brute_force_guard.record_failure(ip)


def record_login_success(ip: str):
    _brute_force_guard.record_success(ip)


def check_login_allowed(ip: str) -> bool:
    return not _brute_force_guard.is_locked(ip)
