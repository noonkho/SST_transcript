# SST API reference

Everything the web UI does is also available over HTTP. The `/v1` endpoints follow
the OpenAI audio API, so OpenAI SDKs work unchanged. If login is enabled, send
`Authorization: Bearer <your API key>` with every request (the machine running the
server itself is never asked).

Interactive docs: <http://localhost:8756/docs>.

## `POST /v1/audio/transcriptions` (OpenAI-compatible)

Multipart form fields:

| Field | Default | Description |
|---|---|---|
| `file` | *required* | The audio file |
| `model` | server default | Any id from [`GET /v1/models`](#get-v1models--discovery--validation); unknown ids → 400 `model_not_found` |
| `language` | auto | `yue` (Cantonese), `zh` (Mandarin), `en` (English), … |
| `response_format` | `json` | `json`, `verbose_json`, `text`, `srt`, `vtt` (DOCX is UI-only, via `/api/jobs/{id}/download`) |
| `diarize` | `true` | *(SST extension)* speaker diarization on/off |
| `num_speakers` | auto | *(SST extension)* exact number of speakers, if known |
| `min_speakers` / `max_speakers` | auto | *(SST extension)* bounds for the automatic speaker count (ignored when `num_speakers` is set) |
| `diarization_model` | server default | *(SST extension)* diarization model id |

```bash
curl -s http://localhost:8756/v1/audio/transcriptions \
  -F file=@meeting.m4a \
  -F response_format=verbose_json | jq .
```

Response (`verbose_json`):

```json
{
  "task": "transcribe",
  "language": "yue",
  "duration": 1834.2,
  "text": "…full transcript…",
  "segments": [
    {"id": 0, "start": 0.0, "end": 6.6, "speaker": "SPEAKER_00", "text": "大家好，歡迎…"},
    {"id": 1, "start": 6.6, "end": 13.2, "speaker": "SPEAKER_01", "text": "Thank you, 我哋開始啦"}
  ],
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "model": "openai/whisper-large-v3",
  "diarization_model": "pyannote/speaker-diarization-community-1",
  "warnings": []
}
```

## Word-level timestamps & stable speaker labels

`verbose_json` segments carry a `words` array:

```json
{"id": 0, "start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "Good morning everyone.",
 "words": [{"word": "Good", "start": 0.0, "end": 0.3}, {"word": "morning", "start": 0.3, "end": 0.8}]}
```

- Whisper produces real word timings (they already drive speaker splitting).
- Engines without word timestamps (SenseVoice) degrade to **one entry spanning the
  segment** — the array is never missing.
- **Speaker labels are deterministic**: `SPEAKER_00` is always the first speaker
  heard, so re-running the same file yields identical labels. Labels are *per file* —
  `SPEAKER_00` in file A and file B are not guaranteed to be the same person.

## Progress for long files

Audio is always split on silence into ≤28 s chunks (Whisper's window), so multi-hour
files stream through without timing out — no size threshold to configure.

**Poll:** every response carries an `X-Job-ID` header.

```bash
curl -s http://localhost:8756/v1/audio/transcriptions/<job-id>/progress
# {"status":"processing","chunks_complete":2,"chunks_total":4,"percent":50,"stage":"transcribing","eta_seconds":31.2}
```

**Stream (NDJSON):** add `?stream_progress=true` to get progress on the same request;
the last line carries the finished transcript, identical to the non-streaming response.

```bash
curl -N -s "http://localhost:8756/v1/audio/transcriptions?stream_progress=true" \
  -F file=@meeting.m4a -F response_format=verbose_json
```
```
{"event":"accepted","job_id":"a1b2c3"}
{"event":"chunks_planned","chunks_total":4,"duration_seconds":7200}
{"event":"chunk_complete","chunk_index":0,"chunks_complete":1,"chunks_total":4,"percent":30}
{"event":"complete","job_id":"a1b2c3","total_text_length":10840,"duration_seconds":7200,"response":{…}}
```

## Graceful degradation when diarization is unavailable

`diarize` defaults to `true`, but a missing or unloadable diarization model never
fails the request — **transcription is not lost**. The server returns **200** with the
transcript, `speakers: null`, every `segments[].speaker` set to `null` (the key is
always present), and a `warnings` entry:

```json
{
  "text": "Good morning everyone…",
  "language": "en",
  "duration": 42.5,
  "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": "Good morning…", "speaker": null}],
  "speakers": null,
  "diarization_model": null,
  "warnings": [
    {
      "code": "diarization_unavailable",
      "message": "Diarization model pyannote/speaker-diarization-community-1 not loaded; segments processed without speaker labels. Segments[].speaker is null."
    }
  ]
}
```

- `warnings` is present on **every** response — an empty array when nothing degraded.
- The same `diarization_unavailable` code covers both "the model wouldn't load" and
  "the diarizer failed mid-run"; the message says which.
- `speaker` is `null` whenever no diarization ran (including `diarize=false`) rather
  than a fabricated `SPEAKER_00`. `speakers` is `null` too, and `diarization_model`
  reports what actually ran.
- Only the STT model failing to load is fatal (500) — without it there is no transcript.
- An explicit `diarization_model=` that the server doesn't have is still a
  **400 `model_not_found`**: you asked for something specific, so it's a client error
  rather than a silent downgrade.

## From an LLM application (Python)

```python
# Option A — OpenAI SDK, pointed at the local server
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8756/v1", api_key="not-needed")
with open("meeting.m4a", "rb") as f:
    result = client.audio.transcriptions.create(file=f, model="", response_format="verbose_json")

# Option B — plain HTTP (any language)
import requests
with open("meeting.m4a", "rb") as f:
    result = requests.post(
        "http://localhost:8756/v1/audio/transcriptions",
        files={"file": f}, data={"response_format": "verbose_json"},
    ).json()
```

See **[client_example.py](../client_example.py)** for a runnable demo, including async
submission with a live progress bar:

```bash
uv run python client_example.py meeting.m4a
```

## `GET /health` — readiness probe

Lightweight liveness/readiness check for orchestrators and clients that verify the
backend at startup. **No auth** (works even with login enabled), no inference, no
disk access — it reads in-memory state only and answers in ~2 ms.

```bash
curl -s http://localhost:8756/health
```

| Server state | HTTP | Body |
|---|---|---|
| Both models loaded | 200 | `{"status":"ready","version":"0.1.0","models_loaded":["openai/whisper-large-v3","pyannote/speaker-diarization-community-1"]}` |
| STT up, diarizer not loaded | 200 | adds `"warnings":[{"code":"model_not_loaded","model":"pyannote/speaker-diarization-community-1"}]` — transcription still works, so this is **not** a failure |
| STT not loaded (starting, downloading, or failed) | 503 | `{"status":"degraded","version":"0.1.0","error":"Required model openai/whisper-large-v3 not loaded or accessible"}` |

`ready` means the configured STT model is resident and transcription can serve.
Use it as a Kubernetes liveness/readiness probe, or poll it after boot — a cold
start returns 503 until the model finishes loading.

## `GET /v1/models` — discovery & validation

Lists every model the server can serve *right now* (curated + any added from
Hugging Face search), with what each can do. Validate a configured model id
against this before uploading audio.

```bash
curl -s http://localhost:8756/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/whisper-large-v3",
      "object": "model",
      "owned_by": "openai",
      "kind": "stt",
      "capabilities": ["transcription"],
      "languages": ["af", "am", "…", "yue", "zh"],
      "input_modality": "audio",
      "max_audio_length_seconds": 43200,
      "loaded": true
    },
    {
      "id": "pyannote/speaker-diarization-community-1",
      "object": "model",
      "owned_by": "pyannote",
      "kind": "diarization",
      "capabilities": ["diarization"],
      "input_modality": "audio",
      "loaded": true
    }
  ]
}
```

- `capabilities` — what the model itself does. STT models report only
  `transcription`; **no STT model can identify speakers.** Diarization comes from a
  separate model (the entries with the `diarization` capability), which the server
  pairs with the STT model when a request asks for it. So `diarize=true` works as
  long as *some* diarization model is available — check for one in this list.
- `languages` — read live from the model's own tokenizer (Whisper reports all 99
  codes including `yue`); omitted for diarization models, which are language-independent.
- `max_audio_length_seconds` — 43200 (12 h). **Advisory guidance for clients, not
  enforced.** The pipeline chunks on silence and handles multi-hour files; this is
  what a comfortable single request looks like.

```python
# validate before sending audio (see client_example.py)
models = requests.get("http://localhost:8756/v1/models").json()["data"]
ids = [m["id"] for m in models if "transcription" in m["capabilities"]]
assert MY_MODEL in ids, f"{MY_MODEL} unavailable; server offers {ids}"
```

## `/v1` error format

Every `/v1` error returns the OpenAI envelope, so OpenAI SDKs and other clients can
parse failures uniformly:

```json
{"error": {"message": "Model not found: nonexistent-whisper",
           "type": "invalid_request_error",
           "code": "model_not_found"}}
```

| Status | `type` | Example `code` | When |
|---|---|---|---|
| 400 | `invalid_request_error` | `model_not_found` | `model`/`diarization_model` isn't downloaded |
| 400 | `invalid_request_error` | `missing_required_field` | no `file` in the request |
| 400 | `invalid_request_error` | `invalid_response_format` | `response_format` not one of `json, verbose_json, text, srt, vtt` |
| 401 | `invalid_api_key` | `invalid_api_key` | login enabled and the Bearer key is wrong/missing |
| 422 | `invalid_request_error` | `invalid_value` | a field is present but unusable |
| 429 | `rate_limit_error` | `rate_limit_exceeded` | too many failed logins |
| 500 | `server_error` | `internal_error` | transcription failed |

The web UI's own `/api/*` endpoints are unchanged — they keep FastAPI's
`{"detail": "..."}` shape. Only `/v1` speaks the OpenAI dialect.

## Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness probe — 200 `ready` / 503 `degraded`. No auth |
| `GET /v1/models` | List servable models + capabilities (OpenAI-style) |
| `POST /api/transcribe` | Async job submission (returns a job id immediately) |
| `GET /api/jobs/{id}` | Job status, progress, ETA, and result |
| `GET /api/jobs/{id}/events` | Server-sent events stream of progress |
| `POST /api/jobs/{id}/cancel` | Cancel a queued or running job (a running job stops within ~1–3 s) |
| `POST /api/jobs/{id}/rediarize` | Re-detect speakers of a finished job: body `{"num_speakers": 4}` (or `min_speakers` / `max_speakers`, or `{}` for auto). Creates a **new** job; only the diarizer runs, the text is reused |
| `DELETE /api/jobs/{id}` | Delete a finished job (transcript + audio) |
| `GET /api/jobs/{id}/audio` | Stream the original audio (used by the player) |
| `PUT /api/jobs/{id}/result` | Save transcript edits (`{"segments": [...]}`; a segment may keep its `words` list) |
| `GET /api/jobs/{id}/download?format=srt` | Download result as `json`/`vtt`/`srt`/`text`/`docx` |
| `GET /api/status` | Server, device, and loaded-model status |
| `GET /api/logs?lines=300` | Last lines of the server log (what the Dashboard shows) |
| `POST /api/logs/clear` | Clear the log view and start a new log file |
| `GET /api/logs/download` | Download the current log file |

The server listens on `0.0.0.0:8756`, so other machines on your local network can
use it at `http://<this-machine>.local:8756`.
