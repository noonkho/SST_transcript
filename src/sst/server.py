"""FastAPI application: OpenAI-compatible API + UI API + static web UI."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import mimetypes
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import SUPPORTED_EXTENSIONS, ffmpeg_available
from .auth import (
    COOKIE_NAME,
    SESSION_TTL,
    drop_session,
    is_local,
    key_ok,
    login_rate_ok,
    new_session,
    record_fail,
    session_valid,
)
from .config import AppConfig, config
from .formats import FORMATTERS
from .jobs import jobs
from .manager import manager
from .pipeline import run_transcription
from .registry import CATALOG, DIARIZATION_CATALOG, STT_CATALOG, classify_hf_model, find_entry

log = logging.getLogger("sst.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """Load configured models once at startup (in background) for 24/7 low latency."""
    def load():
        downloaded = manager.downloaded_repos()
        # First run: fetch the configured STT model and the ungated diarizer
        # automatically (progress is visible in the Models tab). The gated
        # pyannote model is only fetched once a HF token has been saved.
        if config.stt_model not in downloaded:
            log.info("first run: downloading %s in the background", config.stt_model)
            manager.start_download(config.stt_model)
        if "builtin/vad-ecapa-clustering" not in downloaded:
            manager.start_download("builtin/vad-ecapa-clustering")
        if config.diarization_model not in downloaded and config.hf_token:
            manager.start_download(config.diarization_model)
        try:
            manager.ensure_loaded()
            log.info("models preloaded: stt=%s diar=%s", manager.stt_repo, manager.diar_repo)
        except Exception as exc:  # noqa: BLE001
            log.warning("model preload deferred: %s", exc)
    threading.Thread(target=load, daemon=True).start()
    yield


app = FastAPI(
    title="SST — Local Speech-to-Text + Diarization",
    version=__version__,
    lifespan=_lifespan,
)


# ------------------------------------------------------------------- auth
# Allow-list of paths that never require auth, even when config.auth_enabled.
PUBLIC_PREFIXES = ("/static/",)
PUBLIC_PATHS = {"/login", "/logout", "/favicon.ico"}


@app.middleware("http")
async def auth_gate(request: Request, call_next):
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


def _restart_web():
    from . import __main__ as m
    m.web.request_restart()


# ------------------------------------------------------------------ helpers

def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "audio").suffix.lower() or ".bin"
    if suffix not in SUPPORTED_EXTENSIONS and suffix != ".bin":
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="sst_")
    with tmp:
        shutil.copyfileobj(file.file, tmp)
    return tmp.name


def _submit(file: UploadFile, params: dict):
    path = _save_upload(file)

    def work(job):
        # Audio is kept on disk (job.audio_path) for playback/editing; the
        # job store deletes it when the job is deleted or rotated out.
        return run_transcription(job, job.audio_path)

    return jobs.submit(
        filename=file.filename or "audio", params=params, fn=work, audio_src=path,
    )


def _attachment_headers(filename: str) -> dict:
    """Content-Disposition that survives non-ASCII filenames (RFC 5987)."""
    ascii_name = filename.encode("ascii", "replace").decode().replace('"', "_")
    quoted = urllib.parse.quote(filename)
    return {
        "Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quoted}"
    }


# ------------------------------------------------------- OpenAI-compatible

@app.post("/v1/audio/transcriptions")
def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form(""),
    response_format: str = Form("json"),
    prompt: str = Form(""),  # accepted for compatibility; unused
    temperature: float = Form(0.0),  # accepted for compatibility; unused
    diarize: bool = Form(True),
    num_speakers: int | None = Form(None),
    diarization_model: str = Form(""),
):
    """OpenAI-compatible transcription. Blocks until the result is ready.

    Extensions beyond OpenAI: `diarize`, `num_speakers`, `diarization_model`;
    every JSON response includes diarized `segments`.
    """
    if response_format not in FORMATTERS:
        raise HTTPException(400, f"response_format must be one of {list(FORMATTERS)}")
    job = _submit(file, {
        "model": model or None,
        "language": language or None,
        "diarize": diarize,
        "num_speakers": num_speakers,
        "diarization_model": diarization_model or None,
    })
    while job.status in ("queued", "running"):
        time.sleep(0.3)
    if job.status == "error":
        raise HTTPException(500, job.error)
    if job.status == "cancelled":
        raise HTTPException(499, "transcription cancelled")

    result = dict(job.result or {})
    if response_format == "verbose_json":
        result = {
            "task": "transcribe",
            "language": result.get("language", ""),
            "duration": result.get("duration", 0),
            "text": result.get("text", ""),
            "segments": [
                {"id": i, **seg} for i, seg in enumerate(result.get("segments", []))
            ],
            "speakers": result.get("speakers", []),
            "model": result.get("model"),
        }
    formatter, content_type = FORMATTERS[response_format]
    return Response(content=formatter(result), media_type=content_type)


@app.get("/v1/models")
def openai_models():
    downloaded = manager.downloaded_repos()
    data = []
    for entry in CATALOG:
        if entry.repo_id in downloaded:
            data.append({
                "id": entry.repo_id,
                "object": "model",
                "created": 0,
                "owned_by": entry.repo_id.split("/")[0],
                "kind": entry.kind,
                "languages": entry.languages,
                "loaded": entry.repo_id in (manager.stt_repo, manager.diar_repo),
            })
    return {"object": "list", "data": data}


# ------------------------------------------------------------------ UI API

@app.get("/api/status")
def api_status():
    return {
        "version": __version__,
        "running": True,
        "ffmpeg": ffmpeg_available(),
        **manager.status(),
        "config": {
            "stt_model": config.stt_model,
            "diarization_model": config.diarization_model,
            "has_hf_token": bool(config.hf_token),
            "port": config.port,
            "max_jobs": config.clamped_max_jobs(),
            "auth_enabled": config.auth_enabled,
            "has_api_key": bool(config.api_key),
        },
    }


@app.get("/api/models")
def api_models():
    downloaded = manager.downloaded_repos()

    def annotate(d: dict, repo_id: str) -> dict:
        d["downloaded"] = repo_id in downloaded
        dl = manager.downloads.get(repo_id)
        d["download"] = None if dl is None else {
            "status": dl.status, "progress": dl.progress,
            "downloaded_bytes": dl.downloaded_bytes, "total_bytes": dl.total_bytes,
            "eta_seconds": dl.eta_seconds,
            "error": dl.error,
        }
        d["selected"] = repo_id in (config.stt_model, config.diarization_model)
        d["loaded"] = repo_id in (manager.stt_repo, manager.diar_repo)
        return d

    stt = [annotate(asdict(e), e.repo_id) for e in STT_CATALOG]
    # Models added via Hugging Face search appear alongside the curated ones —
    # including ones still downloading, so their progress is visible here too.
    catalog_ids = {e.repo_id for e in STT_CATALOG} | {e.repo_id for e in DIARIZATION_CATALOG}
    custom_repos = {c["repo_id"]: c for c in manager.custom_downloaded()}
    for repo_id, dl in manager.downloads.items():
        if repo_id not in catalog_ids and not repo_id.startswith("builtin/") and repo_id not in custom_repos:
            custom_repos[repo_id] = {
                "repo_id": repo_id,
                "engine": classify_hf_model(repo_id, []),
                "supported": classify_hf_model(repo_id, []) is not None,
            }
    for custom in custom_repos.values():
        stt.append(annotate({
            "repo_id": custom["repo_id"],
            "kind": "stt",
            "engine": custom["engine"] or "unknown",
            "display_name": custom["repo_id"],
            "languages": "see model page",
            "size": "",
            "strengths": "Added from Hugging Face search."
                         + ("" if custom["supported"] else " Unsupported architecture — cannot be loaded."),
            "license": "See the model page on huggingface.co",
            "gated": False,
            "word_timestamps": custom["engine"] == "whisper",
            "requires_extra": "",
            "tags": ["custom"],
        }, custom["repo_id"]))
    return {
        "stt": stt,
        "diarization": [annotate(asdict(e), e.repo_id) for e in DIARIZATION_CATALOG],
    }


@app.post("/api/models/download")
def api_download(body: dict):
    repo_id = body.get("repo_id", "")
    if not repo_id:
        raise HTTPException(400, "repo_id required")
    entry = find_entry(repo_id)
    if entry and entry.gated and not config.hf_token:
        raise HTTPException(400, f"'{repo_id}' is gated — add your Hugging Face token in Settings first.")
    state = manager.start_download(repo_id)
    return {"repo_id": repo_id, "status": state.status}


@app.post("/api/models/download/cancel")
def api_download_cancel(body: dict):
    repo_id = body.get("repo_id", "")
    if not repo_id:
        raise HTTPException(400, "repo_id required")
    state = manager.cancel_download(repo_id)
    if state is None:
        raise HTTPException(404, "no download for this model")
    return {"repo_id": repo_id, "status": state.status}


@app.post("/api/models/remove")
def api_remove(body: dict):
    repo_id = body.get("repo_id", "")
    if not repo_id:
        raise HTTPException(400, "repo_id required")
    try:
        manager.delete_model(repo_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True}


@app.get("/api/models/search")
def api_search(q: str = "", limit: int = 20):
    if not q.strip():
        return {"results": []}
    try:
        return {"results": manager.search_hub(q.strip(), limit=min(limit, 50))}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Hugging Face search failed (offline?): {exc}") from exc


@app.post("/api/config")
def api_config(body: dict):
    changed_models = False
    for key in ("stt_model", "diarization_model", "hf_token", "device_override"):
        if key in body and getattr(config, key) != body[key]:
            setattr(config, key, body[key])
            if key in ("stt_model", "diarization_model"):
                changed_models = True
    if "max_jobs" in body:
        try:
            config.max_jobs = max(3, min(20, int(body["max_jobs"])))
        except (TypeError, ValueError):
            raise HTTPException(400, "max_jobs must be a number between 3 and 20") from None
        jobs.enforce_limit()

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
            raise HTTPException(400, "Port must be 1024-65535") from None
        if p != config.port:
            config.port = p
            restart_needed = True

    config.save()
    # Load when the selection changed OR when what's in memory differs from the
    # selection (e.g. a per-request model override loaded something else).
    out_of_sync = (
        manager.stt_repo != config.stt_model
        or manager.diar_repo != config.diarization_model
    )
    if (changed_models or out_of_sync) and body.get("load_now", True):
        def load():
            try:
                manager.ensure_loaded()
            except Exception as exc:  # noqa: BLE001
                log.warning("model load failed: %s", exc)
        threading.Thread(target=load, daemon=True).start()

    if restart_needed:
        # Delay so this HTTP response is flushed before the socket drops.
        threading.Timer(0.4, _restart_web).start()
        return {"ok": True, "restart": True, "port": config.port}
    return {"ok": True}


@app.post("/api/config/regenerate-key")
def regen_key():
    import secrets
    config.api_key = secrets.token_urlsafe(24)
    config.save()
    return {"api_key": config.api_key}


# ------------------------------------------------------------------ network

_CGNAT = ipaddress.ip_network("100.64.0.0/10")  # RFC 6598 — Tailscale & carrier NAT


def _primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # no packet sent; picks default-route IP
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _all_ipv4() -> set[str]:
    """Every IPv4 the host has, best-effort across macOS/Linux without extra deps."""
    ips: set[str] = set()
    for cmd in (["ip", "-o", "-4", "addr", "show"], ["ifconfig", "-a"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout
        except Exception:
            continue
        if out:
            # matches "inet 192.168.1.5" (mac/ip) and "inet addr:192.168.1.5" (old linux)
            ips.update(re.findall(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", out))
            break
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    ips.add(_primary_ip())
    return ips


def _classify_ip(ip: str) -> str | None:
    """'lan' | 'vpn' | 'public' | None (loopback/link-local, skipped)."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if a.is_loopback or a.is_link_local or a.is_unspecified:
        return None
    if a in _CGNAT:
        return "vpn"          # Tailscale etc — not reachable by ordinary LAN neighbours
    if a.is_private:
        return "lan"
    return "public"


@app.get("/api/network")
def api_network():
    host = socket.gethostname()
    lan, vpn = [], []
    for ip in _all_ipv4():
        kind = _classify_ip(ip)
        if kind == "lan":
            lan.append(ip)
        elif kind == "vpn":
            vpn.append(ip)
    lan.sort()
    vpn.sort()
    # Bonjour/mDNS uses the short hostname; a resolver may hand back an FQDN
    # (e.g. a Tailscale ...ts.net name), so strip to the first label.
    short = host.split(".")[0]
    return {
        "hostname": host,
        "mdns": f"{short}.local" if short else None,
        "port": config.port,
        "addresses": lan,          # WiFi/Ethernet LAN — what neighbours use
        "vpn_addresses": vpn,      # e.g. Tailscale, reachable only on the same tailnet
        "auth_enabled": config.auth_enabled,
    }


# ------------------------------------------------------------------- jobs

@app.post("/api/transcribe")
def api_transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
    diarize: bool = Form(True),
    num_speakers: int | None = Form(None),
):
    job = _submit(file, {
        "language": language or None,
        "diarize": diarize,
        "num_speakers": num_speakers,
    })
    return job.public()


@app.get("/api/jobs")
def api_jobs():
    return {"jobs": [j.public() for j in jobs.all()]}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.public(include_result=True)


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    job = jobs.cancel(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.public()


@app.delete("/api/jobs/{job_id}")
def api_job_delete(job_id: str):
    try:
        if not jobs.delete(job_id):
            raise HTTPException(404, "job not found")
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"ok": True}


# Browser-friendly types where mimetypes guesses poorly (.m4a → audio/mp4a-latm).
_AUDIO_MIME = {".m4a": "audio/mp4", ".aac": "audio/aac", ".opus": "audio/ogg", ".caf": "audio/x-caf"}


@app.get("/api/jobs/{job_id}/audio")
def api_job_audio(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.audio_path or not Path(job.audio_path).exists():
        raise HTTPException(404, "audio not available for this job")
    suffix = Path(job.audio_path).suffix.lower()
    media_type = _AUDIO_MIME.get(suffix) or mimetypes.guess_type(job.audio_path)[0] or "application/octet-stream"
    return FileResponse(job.audio_path, media_type=media_type)


@app.put("/api/jobs/{job_id}/result")
def api_job_update_result(job_id: str, body: dict):
    """Persist edits from the transcript editor (segments and speaker names)."""
    job = jobs.get(job_id)
    if not job or job.status != "done" or not job.result:
        raise HTTPException(404, "no finished result for this job")
    segments = body.get("segments")
    if not isinstance(segments, list):
        raise HTTPException(400, "body must contain a 'segments' list")
    cleaned = []
    for seg in segments:
        try:
            cleaned.append({
                "start": round(float(seg["start"]), 3),
                "end": round(float(seg["end"]), 3),
                "speaker": str(seg["speaker"])[:80] or "SPEAKER_00",
                "text": str(seg["text"]),
            })
        except (KeyError, TypeError, ValueError):
            raise HTTPException(400, "each segment needs start, end, speaker, text") from None
    cleaned.sort(key=lambda s: s["start"])
    result = dict(job.result)
    result["segments"] = cleaned
    result["speakers"] = sorted({s["speaker"] for s in cleaned})
    result["text"] = " ".join(s["text"] for s in cleaned).strip()
    result["edited"] = True
    colors = body.get("speaker_colors")
    if isinstance(colors, dict):
        result["speaker_colors"] = {
            str(name)[:80]: int(idx) % 10
            for name, idx in colors.items()
            if isinstance(idx, (int, float)) and str(name) in result["speakers"]
        }
    jobs.update_result(job, result)
    return {"ok": True, "result": result}


@app.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")

    async def stream():
        while True:
            payload = job.public(include_result=job.finished)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if job.finished:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/download")
def api_job_download(job_id: str, format: str = "json"):
    job = jobs.get(job_id)
    if not job or job.status != "done" or not job.result:
        raise HTTPException(404, "no finished result for this job")
    if format not in FORMATTERS:
        raise HTTPException(400, f"format must be one of {list(FORMATTERS)}")
    formatter, content_type = FORMATTERS[format]
    stem = Path(job.filename).stem or "transcript"
    ext = {"json": "json", "verbose_json": "json", "vtt": "vtt", "srt": "srt",
           "text": "txt", "docx": "docx"}[format]
    return Response(
        content=formatter(job.result),
        media_type=content_type,
        headers=_attachment_headers(f"{stem}.{ext}"),
    )


# ---------------------------------------------------------------- static UI

@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
