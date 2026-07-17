"""Demo: how an LLM application calls the SST service.

Run the server first (./start.command or ./start.sh), then:

    uv run python client_example.py path/to/audio.m4a

Three integration styles are shown:
  1. Plain `requests` (works from any language via HTTP)
  2. The official OpenAI Python SDK pointed at the local server
  3. Async job submission with live progress (what the web UI uses)

If you've enabled login in Settings (Access & security), requests from other
machines need the API key as a Bearer token, e.g.:

    requests.post(url, ..., headers={"Authorization": "Bearer YOUR_KEY"})
    OpenAI(base_url=f"{BASE_URL}/v1", api_key="YOUR_KEY")   # SDK sends the header for you

Calls from localhost (this machine) are never challenged, even with login enabled.
"""

from __future__ import annotations

import json
import sys
import time

import requests

BASE_URL = "http://localhost:8756"


# ------------------------------------------------------------------
# 0. Discover models and validate one before transcribing
# ------------------------------------------------------------------
def list_models(headers: dict | None = None) -> list[dict]:
    """GET /v1/models — every model the server can serve, with capabilities."""
    resp = requests.get(f"{BASE_URL}/v1/models", headers=headers or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]


def validate_model(model_id: str, need: str = "transcription") -> dict:
    """Check a configured model id against the server before sending audio.

    Raises ValueError with the available ids if it can't be served — cheaper
    than discovering it after uploading a long recording.
    """
    models = list_models()
    match = next((m for m in models if m["id"] == model_id), None)
    if match is None:
        usable = [m["id"] for m in models if need in m["capabilities"]]
        raise ValueError(f"Model {model_id!r} not available. Server offers: {usable}")
    if need not in match["capabilities"]:
        raise ValueError(f"Model {model_id!r} cannot do {need}; it does {match['capabilities']}")
    return match


def explain_error(resp: requests.Response) -> str:
    """/v1 errors are OpenAI-shaped: {"error": {message, type, code}}."""
    try:
        err = resp.json()["error"]
        return f"[{resp.status_code} {err['code']}] {err['message']}"
    except Exception:
        return f"[{resp.status_code}] {resp.text[:200]}"


# ------------------------------------------------------------------
# 1. Simple blocking call (OpenAI-compatible endpoint)
# ------------------------------------------------------------------
def transcribe_simple(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            files={"file": f},
            data={
                "response_format": "verbose_json",  # json | verbose_json | text | srt | vtt
                "language": "",                     # "" = auto-detect; or "yue", "zh", "en"
                "diarize": "true",                  # SST extension: speaker diarization
                # "num_speakers": "2",              # optional hint
            },
            timeout=3600,
        )
    if not resp.ok:
        raise RuntimeError(explain_error(resp))
    return resp.json()


# ------------------------------------------------------------------
# 2. Using the official OpenAI SDK (pip install openai)
# ------------------------------------------------------------------
def transcribe_with_openai_sdk(audio_path: str):
    from openai import OpenAI

    # api_key is ignored by the server unless login is enabled in Settings;
    # if it is, pass your real key here (e.g. api_key="YOUR_KEY").
    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="not-needed-local")
    with open(audio_path, "rb") as f:
        return client.audio.transcriptions.create(
            file=f,
            model="",  # empty = server's configured model
            response_format="verbose_json",
        )


# ------------------------------------------------------------------
# 3. Async submission with progress (for long recordings)
# ------------------------------------------------------------------
def transcribe_with_progress(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        job = requests.post(
            f"{BASE_URL}/api/transcribe",
            files={"file": f},
            data={"diarize": "true"},
        ).json()

    while True:
        state = requests.get(f"{BASE_URL}/api/jobs/{job['id']}").json()
        eta = f" · ~{state['eta_seconds']:.0f}s left" if state.get("eta_seconds") else ""
        print(f"\r  {state['stage']:<14} {state['progress'] * 100:5.1f}%{eta}   ", end="", flush=True)
        if state["status"] in ("done", "error"):
            print()
            if state["status"] == "error":
                raise RuntimeError(state["error"])
            return state["result"]
        time.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]

    print("Models the server can serve:")
    for m in list_models():
        extra = f" · {len(m['languages'])} languages" if m.get("languages") else ""
        print(f"  {m['id']}  ({', '.join(m['capabilities'])}{extra})")

    print(f"\nTranscribing {path} …")
    result = transcribe_with_progress(path)

    print(f"\nLanguage: {result['language']}   Duration: {result['duration']}s   "
          f"Speakers: {', '.join(result['speakers'])}\n")
    for seg in result["segments"]:
        print(f"[{seg['start']:7.2f} – {seg['end']:7.2f}] {seg['speaker']}: {seg['text']}")

    out = "transcript.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out}")
