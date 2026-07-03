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
    resp.raise_for_status()
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

    print(f"Transcribing {path} …")
    result = transcribe_with_progress(path)

    print(f"\nLanguage: {result['language']}   Duration: {result['duration']}s   "
          f"Speakers: {', '.join(result['speakers'])}\n")
    for seg in result["segments"]:
        print(f"[{seg['start']:7.2f} – {seg['end']:7.2f}] {seg['speaker']}: {seg['text']}")

    out = "transcript.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out}")
