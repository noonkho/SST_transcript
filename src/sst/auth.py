"""API-key + browser-session auth.

Server-side sessions (in-memory; survive an in-process port restart, cleared on
a full process restart — acceptable).

Security note: every authenticated user is full-access (can read the HF token,
change the API key, etc.). Fine for a trusted LAN; if roles are ever needed,
gate `/api/config` mutations to localhost.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time

from .config import config

SESSION_TTL = 24 * 3600          # 24h -> "login again next day"
COOKIE_NAME = "sst_session"

_sessions: dict[str, float] = {}   # token -> expiry epoch
_lock = threading.Lock()

# brute-force cap for /login (per client IP)
_fails: dict[str, list[float]] = {}
MAX_FAILS = 8
FAIL_WINDOW = 300                  # 5 min


def key_ok(candidate: str) -> bool:
    if not config.api_key:
        return False
    return hmac.compare_digest(candidate.strip(), config.api_key)   # constant-time


def new_session() -> str:
    token = secrets.token_urlsafe(32)
    with _lock:
        _sessions[token] = time.time() + SESSION_TTL
    return token


def session_valid(token: str | None) -> bool:
    if not token:
        return False
    with _lock:
        exp = _sessions.get(token)
        if exp is None:
            return False
        if exp < time.time():
            _sessions.pop(token, None)
            return False
        return True


def drop_session(token: str | None):
    if token:
        with _lock:
            _sessions.pop(token, None)


def is_local(host: str | None) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


def login_rate_ok(ip: str) -> bool:
    now = time.time()
    with _lock:
        hits = [t for t in _fails.get(ip, []) if now - t < FAIL_WINDOW]
        _fails[ip] = hits
        return len(hits) < MAX_FAILS


def record_fail(ip: str):
    with _lock:
        _fails.setdefault(ip, []).append(time.time())
