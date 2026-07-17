# SST — Local Speech-to-Text with Speaker Diarization

A **100% local, offline** speech-to-text service with **speaker diarization**, built for
**Cantonese (粵語), Mandarin (普通話), English, and code-switching** between them.

Runs as a 24/7 server with an **OpenAI-compatible REST API** and a clean, macOS-style
web UI. No audio ever leaves your machine.

| | |
|---|---|
| **Languages** | Cantonese, Mandarin, English (+ 96 more via Whisper), auto-detected with manual override |
| **Diarization** | pyannote community-1 (best open-source quality, free HF token) or built-in ungated fallback |
| **Input** | mp3, m4a, wav, flac, ogg, mp4, mov … anything ffmpeg reads; drag & drop or API |
| **Output** | JSON `{start, end, speaker, text}`, WebVTT, SRT, plain text, Word (.docx) table |
| **Editing** | Karaoke-style playback with click-to-seek, inline transcript editing, speaker renaming |
| **Hardware** | Apple Silicon (Metal/MPS), NVIDIA CUDA, or plain CPU — auto-detected |
| **Long files** | Hours-long audio handled via silence-aware chunking, with progress bar + ETA; jobs cancellable |
| **Licensing** | Default model stack is commercially usable (see [Model licensing](#model-licensing--commercial-use)) |

---

## Quick start

### macOS (Mac Studio / MacBook)

**Double-click `start.command`.** That's it. It installs everything it needs on
first run (uv, ffmpeg, Python packages), starts the server, and opens the UI at
<http://localhost:8756>.

From a terminal instead:

```bash
./start.sh          # or: uv sync && uv run sst-server
```

### Linux / NVIDIA (e.g. DGX Spark) — Docker

```bash
docker compose up -d --build
# UI at http://<machine>:8756 — models persist in a named volume
```

CPU-only host? Delete the `deploy:` block in `docker-compose.yml`.

### First run

1. The default STT model (**Whisper large-v3**, ~3 GB) downloads automatically in
   the background — watch progress in the **Models** tab.
2. The built-in diarizer works immediately, no account needed.
3. *Recommended:* enable **pyannote community-1** diarization (better accuracy):
   - Create a free token at <https://huggingface.co/settings/tokens>
   - Accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - Paste the token in **Settings**, then download it in **Models**.
   - After the download, everything runs offline; the token is stored locally.

---

## Choosing models

Open the **Models** tab to download and switch models. Guidance:

| Model | Best for | Size | Notes |
|---|---|---|---|
| **Whisper large-v3** (default) | Highest accuracy, word timestamps | 3 GB | Great Mandarin/English; decent Cantonese |
| **Whisper large-v3-turbo** | Long recordings, 4× faster | 1.6 GB | Near-equal accuracy |
| **SenseVoice Small** | Cantonese + heavy **code-switching** | 1 GB | Very fast; segment-level timestamps only. Needs `uv sync --extra sensevoice` |
| **pyannote community-1** (diarization, default) | Best open-source accuracy, overlapping speech | 30 MB | Gated — free HF token |
| **pyannote 3.1** (diarization) | Same job, previous generation | 30 MB | Gated — free HF token; pure MIT license |
| **Built-in** (diarization) | Zero-setup diarization | 90 MB | VAD + ECAPA embeddings + clustering (see below) |

Only one STT model is held in memory at a time (models load once at startup and
stay resident for low latency). Switching models while a transcription is running
is safe — the switch waits until the current job finishes.

### Adding models from Hugging Face / removing models

The catalog above is the curated default list; you can extend it. In
**Models → Search Hugging Face**, type e.g. `whisper cantonese` — real results
include community fine-tunes such as `alvanlii/whisper-small-cantonese`
(350k+ downloads). Results show download counts and a compatibility badge.
Click **Download** and the model appears in the *Speech-to-text models* list
(marked "Added from Hugging Face search") and in the model selector, like any
built-in entry. Check the model's page for its license before commercial use.

While a model downloads, both its search row and its entry in the models list
show **live progress** — percentage, downloaded/total size, and estimated time
remaining (e.g. `12% · 380 MB / 3.0 GB · ~25m left`) — plus a **✕ Cancel**
button. Cancelling stops the transfer and **removes the partially downloaded
files** from disk; the Download button reappears if you change your mind. If
the server restarts mid-download, click **Download** again — it resumes from
where it stopped.

Every downloaded model — curated or custom — has a **Remove** button that deletes
its files from local storage (you can re-download any time). The model currently
selected/loaded can't be removed; switch to another model first.

**Will a searched model run on my machine?** Yes, if it carries the
`compatible` badge. Everything in this service runs through **PyTorch**, which
covers all supported hardware with the same code and settings — Apple Silicon
(MPS), NVIDIA (CUDA), and plain CPU. A compatible model is compatible
*everywhere*; there is no macOS-only or Linux-only model, and no per-OS
configuration. What the badge filters out is repos published in a **different
runtime's format** — `mlx-community/…` (MLX), `…-ct2` / faster-whisper
(CTranslate2), GGUF/GGML (whisper.cpp), ONNX — which this service intentionally
does not load. Those are alternative *packagings* of the same models, not
better ones for your Mac: if you see `mlx-community/whisper-large-v3`, just use
the standard `openai/whisper-large-v3` — same weights, runs on your GPU via
PyTorch/MPS.

### Offline / air-gapped use

Only the **first download** of each model needs the Internet; after that,
everything runs fully offline (models are cached in `~/.cache/huggingface`).
Models aren't bundled in this repository because they're multi-GB and some
(pyannote) are distributed through gated Hugging Face repos.

To prepare a machine with **no Internet at all**:

1. On a connected machine, download the models you need via the Models tab.
2. Copy `~/.cache/huggingface/hub` to the same path on the offline machine
   (or anywhere, and point `HF_HOME` at it).
3. Copy `data/downloaded_models.json` from this project folder too — it's the
   record of which models are complete.
4. Optionally set `HF_HUB_OFFLINE=1` on the offline machine so nothing ever
   attempts a network call.

### Model licensing / commercial use

Every model card in the Models tab shows its license. Verified summary (July 2026):

| Model | License | Commercial use |
|---|---|---|
| Whisper (all sizes) | Apache-2.0 | ✅ Yes |
| pyannote **community-1** | CC-BY-4.0 | ✅ Yes, **with attribution** — credit "pyannote speaker-diarization-community-1" in your product's docs/about page |
| pyannote **3.1** | MIT | ✅ Yes, no conditions |
| Built-in diarizer (Silero VAD / SpeechBrain ECAPA / scikit-learn) | MIT / Apache-2.0 / BSD | ✅ Yes |
| SenseVoice Small | FunASR Model License | ⚠️ **Ambiguous** — confirm with FunAudioLLM before shipping commercially |
| NVIDIA Sortformer (not included) | CC-BY-**NC**-4.0 | ❌ No — this is why it's not in the catalog |

Note: pyannote's *open* models are free for commercial use — the gating on
Hugging Face only collects contact info. (The paid "pyannoteAI Precision" API
is a separate commercial product.)

### Attribution

This project uses the following third-party models and libraries:

> Speaker diarization by
> [“speaker-diarization-community-1”](https://huggingface.co/pyannote/speaker-diarization-community-1)
> © [pyannoteAI](https://www.pyannote.ai/), licensed under
> [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Used without modification.

If you build a product on top of this service with community-1 as the diarizer,
**carry this credit forward** into your product's documentation or about page
(that's the only condition of CC BY 4.0 — attribution "in any reasonable manner";
an in-app display is not required).

Other components (no attribution required, listed for completeness):
[OpenAI Whisper](https://huggingface.co/openai/whisper-large-v3) (Apache-2.0) ·
[pyannote.audio library](https://github.com/pyannote/pyannote-audio) (MIT) ·
[Silero VAD](https://github.com/snakers4/silero-vad) (MIT) ·
[SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (Apache-2.0) ·
scikit-learn (BSD).

### What is the built-in diarizer?

The built-in option is a classic three-stage diarization pipeline assembled from
permissively-licensed components — no single "diarization model", but a chain:

1. **VAD (Voice Activity Detection)** — [Silero VAD](https://github.com/snakers4/silero-vad)
   (MIT), a tiny neural net that scans the audio and answers one question:
   *when is anyone speaking at all?* Output: speech regions like `2.1s–7.8s`,
   with silences and noise removed.
2. **ECAPA speaker embeddings** — each speech region is cut into 1.5 s windows, and
   [SpeechBrain's ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
   (Apache-2.0) converts each window into a 192-number "voiceprint" vector.
   ECAPA-TDNN is a neural architecture trained on VoxCeleb (~7,000 speakers) so that
   *the same voice always lands near itself* in that vector space, regardless of
   what is being said or in which language.
3. **Clustering** — scikit-learn (BSD) groups those voiceprints by cosine
   similarity (agglomerative clustering). Each group = one speaker; the groups
   are mapped back to time ranges to produce "SPEAKER_00 spoke 0:00–0:07".

Because it treats each window independently, it's weaker than pyannote when
people talk over each other (pyannote's neural pipeline detects overlapping
speech explicitly). For meetings where people mostly take turns, it performs well.

---

## Using the web UI

The interface is **responsive** and adapts to the screen: on wide monitors and
TVs the Transcribe tab uses a two-column layout (transcription on the left,
**Recent jobs** always visible on the right) and the container widens to fill
the space; on tablets and phones the sidebar becomes a top bar and everything
stacks into a single, touch-friendly column.

### Karaoke playback

After a transcription finishes (or when you open a job from **Recent jobs**), the
result card shows an **audio player** above the transcript:

- The line currently being spoken is **highlighted** and auto-scrolled into view.
- **Click any line** to jump the audio there and keep playing.
- Original audio is kept on disk, so playback still works after a server restart.

### Editing the transcript

**Double-click a line** to edit it. While editing, the audio **loops over that
line** so you can listen while you fix the text. Controls:

| Action | How |
|---|---|
| Save the line | <kbd>Enter</kbd> or **✓ Done** |
| Split into two lines at the cursor | <kbd>Shift+Enter</kbd> or **✂ Split** (timestamps split proportionally) |
| Merge into the previous line | <kbd>Backspace</kbd> at the very start of the line, or **⇧ Merge up** |
| Delete the line | **✕ Delete** |
| Insert a new empty line | **＋ Line below**, the **＋** button on line hover, or **＋ Add line at end** |
| Change the line's speaker | Dropdown in the edit toolbar (includes "＋ New speaker…") |
| Discard changes | <kbd>Esc</kbd> or **Cancel** |

**Rename / recolour speakers**: click a speaker chip in the bar above the
transcript — a small panel opens where you can type a new name (e.g.
`SPEAKER_00` → `Alice`) and pick one of 10 colours. Both apply to every line of
that speaker and are saved with the transcript. (By default each speaker name
gets a stable colour derived from the name itself, so colours no longer shuffle
around after edits.)

All edits are **saved on the server immediately** — exports (JSON/VTT/SRT/TXT/DOCX)
always reflect your edits, and edits survive restarts.

### Exports

Buttons at the top of the result card: **JSON**, **VTT**, **SRT**, **TXT**, and
**DOCX** — a Word document with a 6-column table (ID · Start · End · Person · ':' · Transcript).

### Job history & retention

Finished jobs (transcript + audio for playback) are kept on disk. The server keeps
the **last 5 jobs** by default (configurable **3–20** in *Settings → Job history*);
older ones are auto-deleted. Each job row has a **🗑 delete** button, and
running/queued jobs have a **■ Cancel** button (also available on the progress
card during transcription).

To save disk space, uncompressed uploads (wav/flac/aiff…) are re-encoded to
mono AAC (~40 MB per hour instead of ~1 GB) after transcription — playback
quality is unaffected. Already-compressed uploads (mp3/m4a/…) are stored as-is.
Failed or cancelled jobs don't keep their audio.

## API

Interactive docs at <http://localhost:8756/docs>.

### `POST /v1/audio/transcriptions` (OpenAI-compatible)

Multipart form fields:

| Field | Default | Description |
|---|---|---|
| `file` | *required* | The audio file |
| `model` | server default | Any id from [`GET /v1/models`](#get-v1models--discovery--validation); unknown ids → 400 `model_not_found` |
| `language` | auto | `yue` (Cantonese), `zh` (Mandarin), `en` (English), … |
| `response_format` | `json` | `json`, `verbose_json`, `text`, `srt`, `vtt` (DOCX is UI-only, via `/api/jobs/{id}/download`) |
| `diarize` | `true` | *(SST extension)* speaker diarization on/off |
| `num_speakers` | auto | *(SST extension)* number of speakers, if known |
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
  "model": "openai/whisper-large-v3"
}
```

### From an LLM application (Python)

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

See **[client_example.py](client_example.py)** for a runnable demo, including async
submission with a live progress bar:

```bash
uv run python client_example.py meeting.m4a
```

### `GET /v1/models` — discovery & validation

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
      "capabilities": ["transcription", "diarization"],
      "languages": ["af", "am", "…", "yue", "zh"],
      "input_modality": "audio",
      "max_audio_length_seconds": 3600,
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

- `capabilities` — STT models report `transcription` **and** `diarization`, because
  the service pairs any STT model with a diarizer on the same request.
- `languages` — read live from the model's own tokenizer (Whisper reports all 99
  codes including `yue`); omitted for diarization models, which are language-independent.
- `max_audio_length_seconds` — **advisory guidance for clients, not enforced.** The
  pipeline chunks on silence and handles multi-hour files; this is what a comfortable
  single request looks like.

```python
# validate before sending audio (see client_example.py)
models = requests.get("http://localhost:8756/v1/models").json()["data"]
ids = [m["id"] for m in models if "transcription" in m["capabilities"]]
assert MY_MODEL in ids, f"{MY_MODEL} unavailable; server offers {ids}"
```

### `/v1` error format

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

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List servable models + capabilities (OpenAI-style) |
| `POST /api/transcribe` | Async job submission (returns a job id immediately) |
| `GET /api/jobs/{id}` | Job status, progress, ETA, and result |
| `GET /api/jobs/{id}/events` | Server-sent events stream of progress |
| `POST /api/jobs/{id}/cancel` | Cancel a queued or running job |
| `DELETE /api/jobs/{id}` | Delete a finished job (transcript + audio) |
| `GET /api/jobs/{id}/audio` | Stream the original audio (used by the player) |
| `PUT /api/jobs/{id}/result` | Save transcript edits (`{"segments": [...]}`) |
| `GET /api/jobs/{id}/download?format=srt` | Download result as `json`/`vtt`/`srt`/`text`/`docx` |
| `GET /api/status` | Server, device, and loaded-model status |

The server listens on `0.0.0.0:8756`, so other machines on your local network can
use it at `http://<this-machine>.local:8756`.

---

## Network access & security

- **Same machine:** `http://localhost:<port>`.
- **Other devices on WiFi/Ethernet:** on the same router/subnet, open
  `http://<LAN-IP>:<port>` (find the IP on the Dashboard's "Share access" card).
  mDNS `http://<hostname>.local:<port>` works on Apple devices and most other OSes.
  Wired vs wireless doesn't matter — both use the LAN IP.
- **macOS firewall:** System Settings → Network → Firewall. Either keep it on and, on
  first launch, click **Allow incoming connections** for the Python/uv process, or add
  it under *Options…*. (The firewall filters by app, not port — allowing the process is
  enough.) Windows: allow the app on Private networks. Linux ufw: `sudo ufw allow <port>/tcp`.
- **Enable login:** Settings → Access & security → set an API key → toggle "Require
  login for other devices" on. This machine (localhost) is never challenged.
- **API with the key:**
  ```bash
  curl -H "Authorization: Bearer YOUR_KEY" http://<LAN-IP>:<port>/v1/models
  ```
  OpenAI SDK: `OpenAI(base_url="http://<LAN-IP>:<port>/v1", api_key="YOUR_KEY")` — the
  SDK already sends the Bearer header, so no code change is needed beyond the key.
  Browsers log in once at `/login`; the session cookie lasts 24h, then it's login again.
- **Change port:** Settings → Server port → Save & restart. The web server rebinds
  in-process — models already in memory stay loaded, no re-download or reload.
- Note: LAN traffic is plain HTTP (fine inside a trusted network). For untrusted
  networks, put Caddy/nginx in front for HTTPS — out of scope here.

---

## How it works

```
audio file ─ ffmpeg → 16 kHz mono
             ├─ Silero VAD → silence-aware ≤28 s chunks (long-file support, progress %)
             ├─ Diarization (pyannote or built-in) → speaker turns
             └─ STT per chunk (Whisper on CUDA/MPS/CPU) → words with timestamps
                        └─ words × turns overlap → segments split at speaker changes
                                     → JSON / VTT / SRT with {start, end, speaker, text}
```

- **Cross-platform by design:** a single PyTorch codebase; the device is picked at
  startup (`cuda` → `mps` → `cpu`). Works today on Apple Silicon, tomorrow on a
  DGX box, unchanged.
- Jobs run one at a time on a worker thread; the API stays responsive and
  progress/ETA are computed from measured per-chunk throughput.

## Configuration

Everything is configurable in the UI (Settings tab). State lives in `data/config.json`:

| Key | Default | Meaning |
|---|---|---|
| `stt_model` | `openai/whisper-large-v3` | Active STT model |
| `diarization_model` | `pyannote/speaker-diarization-community-1` | Active diarizer (falls back to the built-in one until a HF token is saved) |
| `device_override` | auto | Force `cuda` / `mps` / `cpu` |
| `port` | `8756` | Server port |
| `max_jobs` | `5` | Finished jobs kept on disk (3–20); oldest auto-deleted |
| `auth_enabled` | `false` | Require an API key / browser login for non-localhost requests |
| `api_key` | `""` | API key for the Bearer header / login page (never returned by the API — only a `has_api_key` flag) |

Environment variables: `SST_DATA_DIR` (config/state location), `HF_HOME`
(model cache location).

## Troubleshooting

- **"ffmpeg is required"** — `brew install ffmpeg` (macOS) / `apt-get install ffmpeg` (Linux).
- **"Model … is not downloaded yet"** — open the Models tab and download it (the
  default model auto-downloads on first start; wait for it to finish).
- **pyannote fails to load** — you need to (1) save an HF token in Settings and
  (2) accept the model terms on huggingface.co (both links in Settings). Or switch
  to the built-in diarizer in Models. Until pyannote is available, the server
  automatically falls back to the built-in diarizer (the result JSON's
  `diarization_model` field always tells you which one actually ran).
- **Upgrading from an older version** — if you previously saved settings, your
  `data/config.json` may still point at `pyannote/speaker-diarization-3.1`;
  switch to community-1 in the Models tab (or delete `data/config.json`).
- **Cantonese comes out as written Chinese** — that's Whisper's normalization.
  Try `language=yue` explicitly, or use SenseVoice for colloquial Cantonese.
- **Out of memory** — use `whisper-large-v3-turbo` or `whisper-medium` instead of
  large-v3, and keep only one STT model loaded (automatic).
