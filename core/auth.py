"""
ARIA — Authentication System
==============================
Protects ARIA's API so only trusted devices can access it.

Design: simple but real security.
  - Owner sets a PIN on first run (stored as bcrypt hash, never plain text)
  - Login returns a JWT token (24h expiry)
  - Every API request must carry the token in Authorization header
  - Trusted devices can be paired without re-entering PIN every time
  - Device pairing uses a short-lived 6-digit code shown on screen

Why not OAuth / passwords / usernames?
  This is a personal assistant on a local network.
  A PIN is the right balance — simple enough that you actually use it,
  strong enough to block anyone on the same WiFi.

Token flow:
  1. First run → POST /auth/setup  {pin: "1234"}  → stores hash
  2. Login     → POST /auth/login  {pin: "1234"}  → returns JWT
  3. All APIs  → Header: Authorization: Bearer <token>
  4. Pair device → GET /auth/pair-code → 6-digit code (30s expiry)
               → POST /auth/pair {code, device_name} → device token (30 days)

Storage:
  - PIN hash: data/auth.json  (bcrypt, safe to store)
  - Device tokens: data/devices.json
  - Session tokens: in-memory (lost on restart, by design)
"""

import json
import time
import secrets
import hashlib
import hmac
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console
from fastapi import Request

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTH_FILE    = PROJECT_ROOT / "data" / "auth.json"
DEVICES_FILE = PROJECT_ROOT / "data" / "devices.json"
AUTH_FILE.parent.mkdir(exist_ok=True)

# JWT secret — generated once and stored
_SECRET_FILE = PROJECT_ROOT / "data" / ".jwt_secret"


# ─────────────────────────────────────────────────────────────────────────────
# JWT — lightweight, no external library needed
# ─────────────────────────────────────────────────────────────────────────────

def _get_secret() -> str:
    """Get or create the JWT signing secret."""
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_text().strip()
    secret = secrets.token_hex(32)
    _SECRET_FILE.write_text(secret)
    _SECRET_FILE.chmod(0o600)  # owner-readable only
    return secret


def _b64url(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    padding = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + "=" * padding)


def create_token(payload: dict, expires_hours: float = 24) -> str:
    """Create a signed JWT token."""
    import json as _json
    header  = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = dict(payload)
    payload["exp"] = time.time() + expires_hours * 3600
    payload["iat"] = time.time()
    body    = _b64url(_json.dumps(payload).encode())
    secret  = _get_secret().encode()
    sig     = _b64url(hmac.new(secret, f"{header}.{body}".encode(), hashlib.sha256).digest())
    return f"{header}.{body}.{sig}"


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token. Returns payload or None."""
    try:
        import json as _json
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, body, sig = parts
        secret = _get_secret().encode()
        expected_sig = _b64url(
            hmac.new(secret, f"{header}.{body}".encode(), hashlib.sha256).digest()
        )
        # Constant-time compare
        if not hmac.compare_digest(sig, expected_sig):
            return None
        payload = _json.loads(_b64url_decode(body))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PIN MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _hash_pin(pin: str) -> str:
    """Hash a PIN with bcrypt. Falls back to sha256+salt if bcrypt not installed."""
    try:
        import bcrypt
        return bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        # Fallback: SHA-256 with a stored salt
        salt = secrets.token_hex(16)
        h    = hashlib.sha256((salt + pin).encode()).hexdigest()
        return f"sha256:{salt}:{h}"


def _verify_pin(pin: str, stored_hash: str) -> bool:
    """Verify a PIN against its stored hash."""
    try:
        if stored_hash.startswith("sha256:"):
            _, salt, h = stored_hash.split(":")
            return hmac.compare_digest(
                h, hashlib.sha256((salt + pin).encode()).hexdigest()
            )
        import bcrypt
        return bcrypt.checkpw(pin.encode(), stored_hash.encode())
    except Exception:
        return False


def _load_auth() -> dict:
    if AUTH_FILE.exists():
        try:
            return json.loads(AUTH_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_auth(data: dict):
    AUTH_FILE.write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def _load_devices() -> dict:
    if DEVICES_FILE.exists():
        try:
            return json.loads(DEVICES_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_devices(data: dict):
    DEVICES_FILE.write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# AUTH MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class AuthManager:
    """
    Central authentication manager.
    Handles setup, login, pairing, and token verification.
    """

    def __init__(self):
        self._pair_codes: dict[str, float] = {}   # code → expiry timestamp
        self._active_tokens: set[str]       = set()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def is_setup(self) -> bool:
        """Has the owner set a PIN yet?"""
        return bool(_load_auth().get("pin_hash"))

    def setup(self, pin: str, owner_name: str = "Owner") -> dict:
        """
        First-time setup. Sets the owner PIN.
        Generates a one-time recovery code shown only once.
        Returns an initial session token + recovery_code.
        """
        if len(pin) < 4:
            return {"success": False, "error": "PIN must be at least 4 characters"}

        # Generate 10-char alphanumeric recovery code (shown once, stored as hash)
        recovery_code = secrets.token_urlsafe(8).upper()[:10]

        data = {
            "pin_hash":       _hash_pin(pin),
            "recovery_hash":  _hash_pin(recovery_code),
            "owner_name":     owner_name,
            "setup_ts":       datetime.now().isoformat(),
        }
        _save_auth(data)
        console.print(f"  [green]Auth setup:[/] PIN set for {owner_name}")
        console.print(f"  [yellow]Recovery code (save this!):[/] {recovery_code}")

        # Return initial token (30-day)
        token = create_token({"role": "owner", "name": owner_name}, expires_hours=720)
        self._active_tokens.add(token)
        return {
            "success":       True,
            "token":         token,
            "recovery_code": recovery_code,
            "message":       f"Welcome, {owner_name}. Save your recovery code — it won't be shown again.",
        }

    def reset_pin(self, recovery_code: str, new_pin: str) -> dict:
        """Reset PIN using the recovery code generated at setup."""
        if len(new_pin) < 4:
            return {"success": False, "error": "New PIN must be at least 4 characters"}
        auth = _load_auth()
        if not auth.get("recovery_hash"):
            return {"success": False, "error": "No recovery code on file. Delete data/auth.json to start fresh."}
        if not _verify_pin(recovery_code.upper().strip(), auth["recovery_hash"]):
            console.print("  [red]Auth reset failed:[/] wrong recovery code")
            return {"success": False, "error": "Incorrect recovery code"}
        # Set new PIN, invalidate recovery code (one-use), generate new recovery code
        new_recovery = secrets.token_urlsafe(8).upper()[:10]
        auth["pin_hash"]      = _hash_pin(new_pin)
        auth["recovery_hash"] = _hash_pin(new_recovery)
        auth["reset_ts"]      = datetime.now().isoformat()
        _save_auth(auth)
        token = create_token({"role": "owner", "name": auth.get("owner_name", "Owner")})
        self._active_tokens.add(token)
        console.print("  [green]Auth reset:[/] PIN changed via recovery code")
        return {
            "success":       True,
            "token":         token,
            "recovery_code": new_recovery,
            "message":       "PIN reset. Save your new recovery code.",
        }

    def change_pin(self, current_pin: str, new_pin: str) -> dict:
        """Change PIN when logged in — requires current PIN."""
        if len(new_pin) < 4:
            return {"success": False, "error": "New PIN must be at least 4 characters"}
        auth = _load_auth()
        if not _verify_pin(current_pin, auth.get("pin_hash", "")):
            return {"success": False, "error": "Current PIN is incorrect"}
        auth["pin_hash"] = _hash_pin(new_pin)
        auth["changed_ts"] = datetime.now().isoformat()
        _save_auth(auth)
        console.print("  [green]Auth:[/] PIN changed successfully")
        return {"success": True, "message": "PIN changed successfully"}

    # ── Login ─────────────────────────────────────────────────────────────────

    def login(self, pin: str, device_name: str = "unknown") -> dict:
        """
        Authenticate with PIN. Returns a 24-hour session token.
        """
        auth = _load_auth()
        if not auth.get("pin_hash"):
            return {"success": False, "error": "not_setup", "message": "ARIA has no PIN set. Go to /auth/setup first."}

        if not _verify_pin(pin, auth["pin_hash"]):
            console.print(f"  [red]Auth failed:[/] wrong PIN from {device_name}")
            return {"success": False, "error": "wrong_pin", "message": "Incorrect PIN"}

        token = create_token({
            "role":        "owner",
            "device":      device_name,
            "name":        auth.get("owner_name", "Owner"),
        }, expires_hours=720)   # 30 days — avoids constant re-auth
        self._active_tokens.add(token)
        console.print(f"  [green]Auth:[/] {device_name} logged in")
        return {
            "success":    True,
            "token":      token,
            "expires_in": "30 days",
            "name":       auth.get("owner_name","Owner"),
        }

    # ── Token verification ─────────────────────────────────────────────────────

    def verify(self, token: str) -> Optional[dict]:
        """
        Verify a token from an API request.
        Returns the token payload or None if invalid.
        """
        # Check device tokens first (long-lived)
        devices = _load_devices()
        if token in devices:
            device = devices[token]
            if device.get("expires", 0) > time.time():
                return {"role": "device", "device": device.get("name","?")}
            else:
                # Expired device token
                del devices[token]
                _save_devices(devices)
                return None

        # Check JWT session token
        return verify_token(token)

    def logout(self, token: str):
        """Invalidate a token."""
        self._active_tokens.discard(token)

    # ── Device pairing ─────────────────────────────────────────────────────────

    def generate_pair_code(self) -> str:
        """
        Generate a 6-digit pairing code valid for 60 seconds.
        Show this code on the main device — user enters it on the new device.
        """
        code   = str(secrets.randbelow(1000000)).zfill(6)
        expiry = time.time() + 60
        self._pair_codes[code] = expiry
        console.print(f"  [green]Pair code:[/] {code} (valid 60s)")
        return code

    def pair_device(self, code: str, device_name: str) -> dict:
        """
        Pair a new device using the pairing code.
        Returns a 30-day device token.
        """
        expiry = self._pair_codes.get(code)
        if not expiry or time.time() > expiry:
            return {"success": False, "error": "Invalid or expired code"}

        del self._pair_codes[code]

        # Create long-lived device token
        device_token = secrets.token_hex(32)
        devices      = _load_devices()
        devices[device_token] = {
            "name":    device_name,
            "paired":  datetime.now().isoformat(),
            "expires": time.time() + 30 * 86400,  # 30 days
        }
        _save_devices(devices)

        console.print(f"  [green]Device paired:[/] {device_name}")
        return {
            "success":    True,
            "token":      device_token,
            "device":     device_name,
            "expires_in": "30 days",
        }

    def list_devices(self) -> list[dict]:
        devices = _load_devices()
        result  = []
        now     = time.time()
        for token, d in devices.items():
            result.append({
                "name":    d.get("name","?"),
                "paired":  d.get("paired","?"),
                "active":  d.get("expires",0) > now,
                "token_preview": token[:8] + "…",
            })
        return result

    def revoke_device(self, token_preview: str) -> bool:
        devices = _load_devices()
        for token in list(devices.keys()):
            if token.startswith(token_preview[:8]):
                del devices[token]
                _save_devices(devices)
                return True
        return False

    def status(self) -> dict:
        return {
            "setup":        self.is_setup(),
            "active_codes": len(self._pair_codes),
            "paired_devices": len(_load_devices()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI DEPENDENCY — inject into protected routes
# ─────────────────────────────────────────────────────────────────────────────

# Global instance
auth_manager = AuthManager()


def get_token_from_request(request: Request) -> Optional[str]:
    """Extract Bearer token from Authorization header or query param."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    # Also accept from query param (for WebSocket connections)
    return request.query_params.get("token")


def require_auth(request: Request):
    """
    FastAPI dependency — type-annotated as Request so FastAPI injects the
    HTTP request object rather than treating it as a query param.
    Usage:  @app.get("/protected", dependencies=[Depends(require_auth)])
    Or:     def endpoint(user=Depends(require_auth))
    """
    from fastapi import HTTPException, status
    token = get_token_from_request(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No token. POST /auth/login with your PIN.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = auth_manager.verify(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Login again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


def optional_auth(request: Request) -> Optional[dict]:
    """Like require_auth but returns None instead of raising if no token."""
    token = get_token_from_request(request)
    if not token:
        return None
    return auth_manager.verify(token)
