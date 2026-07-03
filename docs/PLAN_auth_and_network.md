# Implementation plan — LAN exposure, auth, live port change, network info

Handoff spec for the executor. Follow exactly. Keep the existing macOS-style UI
conventions. Verify in the preview after each backend+frontend slice.

## Decisions (already made — do not re-ask)
- Auth = **API key + browser login page**. API callers send `Authorization: Bearer <key>`.
- **Localhost is exempt** (127.0.0.1 / ::1 never challenged — no lockout).
- Scope = **everything** (web UI pages + `/v1` + `/api` + `/docs`).
- Browser sessions expire after **24h** ("login again next day").
- API key UI = hidden field with **reveal / regenerate / copy** (like the mockup).
- Port change is saved then the **web server restarts in-process** (models stay in RAM).
- No reverse proxy (trusted LAN, 6–15 users).

---

## 1. Config — `src/sst/config.py`
Add fields to `AppConfig`:
```python
auth_enabled: bool = False
api_key: str = ""            # >= 4 printable chars, no whitespace
```
Add helper:
```python
import re
_API_KEY_RE = re.compile(r"^\S{4,}$")   # >=4 non-whitespace chars

@staticmethod
def valid_api_key(key: str) -> bool:
    return bool(key) and bool(_API_KEY_RE.match(key)) and key.isprintable()
```
`port` already exists. Keep `host="0.0.0.0"` (already binds all interfaces = reachable over WiFi/Ethernet).

---

## 2. Auth module — new file `src/sst/auth.py`
Server-side sessions (in-memory; survive an in-process port restart, cleared on
full process restart — acceptable).

```python
import hmac, secrets, threading, time
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
```

---

## 3. Auth middleware + login routes — `src/sst/server.py`
Add a Starlette `@app.middleware("http")` **before** the routes are hit.

Allow-list (no auth needed, even when enabled):
- `GET /login`, `POST /login`, `POST /logout`
- `GET /favicon.ico`, anything under `/static/`
Everything else is gated when `config.auth_enabled`.

Gate logic per request:
```python
from starlette.responses import JSONResponse, RedirectResponse
from .auth import COOKIE_NAME, is_local, session_valid, key_ok

PUBLIC_PREFIXES = ("/static/",)
PUBLIC_PATHS = {"/login", "/logout", "/favicon.ico"}

@app.middleware("http")
async def auth_gate(request, call_next):
    if not config.auth_enabled:
        return await call_next(request)
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    if is_local(request.client.host if request.client else None):
        return await call_next(request)
    # 1) API bearer key
    authz = request.headers.get("authorization", "")
    if authz.startswith("Bearer ") and key_ok(authz[7:]):
        return await call_next(request)
    # 2) browser session cookie
    if session_valid(request.cookies.get(COOKIE_NAME)):
        return await call_next(request)
    # unauthenticated
    accepts_html = "text/html" in request.headers.get("accept", "")
    if accepts_html and request.method == "GET":
        return RedirectResponse(f"/login?next={path}", status_code=303)
    return JSONResponse({"detail": "authentication required"}, status_code=401,
                        headers={"WWW-Authenticate": "Bearer"})
```

Login routes:
```python
from .auth import (new_session, drop_session, key_ok, COOKIE_NAME,
                   login_rate_ok, record_fail, SESSION_TTL)

@app.get("/login", response_class=HTMLResponse)
def login_page():
    return (STATIC_DIR / "login.html").read_text()

@app.post("/login")
def login(body: dict, request: Request):
    ip = request.client.host if request.client else "?"
    if not login_rate_ok(ip):
        raise HTTPException(429, "Too many attempts. Wait a few minutes.")
    if not key_ok(body.get("key", "")):
        record_fail(ip)
        raise HTTPException(401, "Wrong key")
    token = new_session()
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE_NAME, token, max_age=SESSION_TTL,
                    httponly=True, samesite="lax")   # no Secure: LAN is http
    return resp

@app.post("/logout")
def logout(request: Request):
    drop_session(request.cookies.get(COOKIE_NAME))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME)
    return resp
```
Import `Request` from fastapi. Middleware must be added right after `app = FastAPI(...)`.

**Security note to preserve in code comment:** every authenticated user is full-
access (can read HF token, change key, etc.). Fine for a trusted LAN; if roles are
ever needed, gate `/api/config` mutations to localhost.

---

## 4. Live port change + in-process web-server restart — `src/sst/__main__.py`
Replace the single `uvicorn.run(...)` with a supervisor that rebuilds the uvicorn
server when a restart is requested. The Python process (and `manager` with models
in RAM) stays alive; only the HTTP listener rebinds.

```python
import threading, uvicorn
from .config import config
from .server import app

class WebServer:
    def __init__(self):
        self._server: uvicorn.Server | None = None
        self._stop = False

    def request_restart(self):
        if self._server:
            self._server.should_exit = True   # unblocks .run()

    def run_forever(self):
        last_good_port = config.port
        while not self._stop:
            cfg = uvicorn.Config(app, host=config.host, port=config.port, log_level="info")
            self._server = uvicorn.Server(cfg)
            try:
                self._server.run()             # blocks until should_exit
                last_good_port = config.port
            except OSError as e:               # port busy / invalid -> revert
                print(f"[web] cannot bind port {config.port}: {e}; reverting to {last_good_port}")
                config.port = last_good_port
                config.save()
                continue

web = WebServer()

def main():
    web.run_forever()
```
Expose `web` so the API can trigger a restart. Import lazily in server.py to avoid a
cycle:
```python
def _restart_web():
    from . import __main__ as m
    m.web.request_restart()
```
When port changes via `/api/config`, spawn a short delayed thread so the HTTP
response is flushed **before** the socket drops:
```python
threading.Timer(0.4, _restart_web).start()
```

---

## 5. Extend `/api/config` — `src/sst/server.py`
Handle new keys. Validate before saving.
```python
# api_key
if "api_key" in body:
    k = (body["api_key"] or "").strip()
    if k and not AppConfig.valid_api_key(k):
        raise HTTPException(400, "API key needs >=4 printable chars, no whitespace")
    config.api_key = k

# auth_enabled
if "auth_enabled" in body:
    want = bool(body["auth_enabled"])
    if want and not config.api_key:
        raise HTTPException(400, "Set an API key before enabling authentication")
    config.auth_enabled = want

# port
restart_needed = False
if "port" in body:
    try:
        p = int(body["port"])
        assert 1024 <= p <= 65535
    except (TypeError, ValueError, AssertionError):
        raise HTTPException(400, "Port must be 1024–65535")
    if p != config.port:
        config.port = p
        restart_needed = True
```
After `config.save()`, if `restart_needed`: `threading.Timer(0.4, _restart_web).start()`
and include `{"restart": True, "port": config.port}` in the response so the UI can
redirect the browser to the new port.

Add a **regenerate** helper endpoint (optional; UI can also just set a value):
```python
@app.post("/api/config/regenerate-key")
def regen_key():
    import secrets
    config.api_key = secrets.token_urlsafe(24)
    config.save()
    return {"api_key": config.api_key}
```
(This endpoint is itself gated by the middleware unless localhost — fine.)

---

## 6. Network info endpoint — `src/sst/server.py`
```python
import socket
def _primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # no packet sent; picks default-route IP
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

@app.get("/api/network")
def api_network():
    host = socket.gethostname()
    ips = set()
    try:
        ips.update(socket.gethostbyname_ex(host)[2])
    except Exception:
        pass
    ips.add(_primary_ip())
    ips = sorted(i for i in ips if not i.startswith("127."))
    return {
        "hostname": host,
        "mdns": f"{host}.local" if not host.endswith(".local") else host,
        "port": config.port,
        "addresses": ips,
        "auth_enabled": config.auth_enabled,
    }
```

Include `auth_enabled` in `/api/status.config` too (UI reads it there).

---

## 7. Frontend — Settings tab (`static/index.html` + `app.js` + `style.css`)
Add two cards to `#tab-settings`.

**API key card** (mirror the mockup: label, hidden input, reveal 👁 / regenerate ↻ /
copy ⧉ icon buttons, hint "≥4 printable chars, no whitespace"):
```html
<div class="card">
  <h2>Access &amp; security</h2>
  <label class="toggle-label">Require login for other devices
    <input type="checkbox" id="auth-enabled" class="toggle">
  </label>
  <div class="hint">When on, devices other than this computer must enter the API
    key (browser) or send it as a Bearer token (API). This machine (localhost) is
    never asked.</div>
  <div class="key-row" style="margin-top:14px">
    <label style="flex:1">API key
      <div class="key-input">
        <input type="password" id="api-key" placeholder="set a key…">
        <button class="icon-btn" id="key-reveal" title="Show/hide">👁</button>
        <button class="icon-btn" id="key-regen" title="Generate a strong key">↻</button>
        <button class="icon-btn" id="key-copy" title="Copy">⧉</button>
      </div>
    </label>
    <button class="btn primary" id="save-key">Save</button>
  </div>
  <div class="hint" id="auth-state"></div>
</div>
```

**Port card:**
```html
<div class="card">
  <h2>Server port</h2>
  <div class="options-row">
    <label>Port <input type="number" id="srv-port" min="1024" max="65535"></label>
    <button class="btn" id="save-port">Save &amp; restart</button>
  </div>
  <div class="hint">Saving restarts the web server on the new port (models stay
    loaded — no re-download). You'll be redirected; other devices must reconnect
    to the new port. If a login is active it will be required again on the new port.</div>
</div>
```

`app.js` behaviour:
- Load current values from `/api/status` (add `auth_enabled`, `api_key` is **never**
  returned in full — return only `has_api_key` bool; the input stays empty with a
  "key is set" hint). Reveal toggles input `type` between password/text **only for a
  value the user just typed**; it cannot reveal the stored key (server never sends it).
  - Regenerate ↻ calls `POST /api/config/regenerate-key`, drops the returned key into
    the input as text (revealed) so the user can copy it, and shows "unsaved — click Save".
  - Copy ⧉ uses `navigator.clipboard.writeText(input.value)`.
- Save key → `POST /api/config {api_key}`.
- Toggle auth → `POST /api/config {auth_enabled}`; if enabling with no key set, show the
  server's 400 message.
- Save port → `POST /api/config {port}`; on `{restart:true}` show "Restarting… redirecting"
  and after ~1.2s `window.location.port = newPort` (or full URL swap). Handle the brief
  fetch failure during rebind gracefully.

CSS: add `.key-input{display:flex;gap:6px;align-items:center}` `.key-input input{flex:1}`
reuse existing `.icon-btn`.

**Login page** — new `static/login.html`: minimal centered card, reuses `style.css`,
one password field + Login button, posts JSON to `/login`, on success reads `?next`
and redirects there (default `/`). Show inline error on 401/429.

---

## 8. Frontend — Dashboard "Share access" card
Add after the stat grid in `#tab-dashboard`. Populated from `/api/network`.
Show, with copy buttons:
- **This computer:** `http://localhost:<port>`
- **WiFi / Ethernet (other devices):** for each address → `http://<ip>:<port>`
  and the mDNS `http://<hostname>.local:<port>`
- **API base URL:** `http://<ip>:<port>/v1`
- If `auth_enabled`: a line "🔒 Login required — share the API key separately" and a
  ready-to-copy curl example:
  `curl -H "Authorization: Bearer <key>" http://<ip>:<port>/v1/models`
- If not: "🔓 Open on the LAN — anyone who can reach this address can use it. Enable
  login in Settings to restrict."
Poll `/api/network` when the Dashboard tab opens (IP/port can change).

---

## 9. README — new "## Network access & security" section
Document exactly (see prose the requester asked for):
- **Same machine:** `http://localhost:<port>`.
- **Other devices on WiFi/Ethernet:** same router/subnet, open `http://<LAN-IP>:<port>`
  (find IP on Dashboard). mDNS `http://<hostname>.local:<port>` on Apple/most OSes.
  Wired vs wireless doesn't matter — both use the LAN IP.
- **macOS firewall:** System Settings → Network → Firewall. Either keep it on and, on
  first launch, click **Allow incoming connections** for the Python/uv process, or add
  it under *Options…*. (Firewall filters by app, not port — allowing the process is
  enough.) Windows: allow the app on Private networks. Linux ufw: `sudo ufw allow <port>/tcp`.
- **Enable login:** Settings → Access & security → set API key → toggle on. localhost
  is never challenged.
- **API with the key:**
  ```bash
  curl -H "Authorization: Bearer YOUR_KEY" http://<LAN-IP>:<port>/v1/models
  ```
  OpenAI SDK: `OpenAI(base_url="http://<LAN-IP>:<port>/v1", api_key="YOUR_KEY")` — the
  SDK already sends the Bearer header, so no code change beyond the key.
  Browsers log in once at `/login`; the session cookie lasts 24h, then re-login.
- **Change port:** Settings → Server port → Save & restart. Models stay loaded.
- Note: LAN traffic is plain HTTP (fine inside a trusted network). For untrusted
  networks put Caddy/nginx in front for HTTPS — out of scope here.
- Update `client_example.py` header comment to mention `api_key="YOUR_KEY"`.

---

## 10. Verify (preview) — do all before done
1. Auth off: other-origin request (simulate by sending no cookie + a non-local
   `X-Forwarded`? no — just confirm open access) works.
2. Set key, enable auth: `curl` without key → 401; with `Authorization: Bearer <key>`
   → 200. `GET /` in browser (non-local simulated) → redirect to `/login`.
3. localhost still open with auth on (the preview IS localhost) — confirm UI works
   without login.
4. Login page: wrong key → 401 msg; 8 wrong → 429; right key → cookie set, `/` loads.
5. Regenerate/reveal/copy key buttons behave.
6. Change port → response `{restart:true}`; server rebinds; models NOT reloaded
   (check logs — no "loading STT model" line); UI redirects to new port; old port dead.
7. Dashboard share card shows LAN IP + URLs; curl example present when auth on.
8. Bad port (e.g. 80) → 400; busy port → server reverts, stays up.

## Files touched
- `src/sst/config.py` (fields + validator)
- `src/sst/auth.py` (new)
- `src/sst/__main__.py` (WebServer supervisor)
- `src/sst/server.py` (middleware, /login, /logout, /api/network, /api/config additions, status.auth_enabled)
- `src/sst/static/index.html` (settings cards, dashboard share card)
- `src/sst/static/login.html` (new)
- `src/sst/static/app.js` (settings + dashboard logic, login redirect on 401)
- `src/sst/static/style.css` (.key-input helpers)
- `client_example.py` (api_key note)
- `README.md` (Network access & security section)
