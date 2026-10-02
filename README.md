# SST — Local Speech-to-Text with Speaker Diarization

Turn recordings into **text with speaker names and timestamps** — on your own
computer. Built for **Cantonese (粵語), Mandarin (普通話), English, and mixes of
them**. No audio ever leaves your machine.

You get a web app (drag in a file, read, play, fix, export to Word) and an
**OpenAI-compatible API** for other programs.

| | |
|---|---|
| **Languages** | Cantonese, Mandarin, English (+ 96 more via Whisper), auto-detected or chosen |
| **Speakers** | Finds who spoke when (pyannote community-1, or a built-in no-account option). Wrong count? Re-detect in seconds |
| **Input** | mp3, m4a, wav, flac, ogg, mp4, mov … anything ffmpeg reads; several files at once |
| **Output** | Word (.docx, Times New Roman + PMingLiU), SRT/VTT subtitles, text, JSON with word timings |
| **Editing** | Karaoke playback, click-to-play, inline editing, rename/merge/recolour speakers |
| **Hardware** | Apple Silicon (MPS), NVIDIA CUDA, or plain CPU — picked automatically |
| **Long files** | Hours-long audio, progress bar with time left, cancel any time |
| **Licensing** | Default models may be used commercially ([details](#model-licensing--commercial-use)) |

**Documentation**

- 📘 [User guide](docs/USER_GUIDE.md) — every feature, step by step
- 🧠 [Models](docs/MODELS.md) — which model to pick, licenses, newer options
- 🔌 [API reference](docs/API.md) — for developers and other apps

---

## Quick start

### What you need

- A Mac with Apple Silicon (M1 or newer), or a Linux PC, ideally with an NVIDIA
  GPU. 16 GB RAM or more is recommended for the default model.
- About **5 GB of free disk** for the default models.
- Internet for the **first** start only (to download the models).

### macOS — easiest

1. Get the project. Either click the green **Code** button on GitHub →
   *Download ZIP* and unzip it, or in Terminal:
   ```bash
   git clone https://github.com/noonkho/SST_transcript.git
   ```
2. **Double-click `start.command`** in the project folder.
   (The first time, macOS may block it: right-click → *Open* → *Open*.)
3. Wait. The first start installs what it needs (uv, ffmpeg, Python packages)
   and opens <http://localhost:8756> in your browser.

### macOS / Linux — from a terminal

```bash
cd SST_transcript
./start.sh
```

Or by hand, if you already have [uv](https://docs.astral.sh/uv/) and ffmpeg:

```bash
uv sync
uv run sst-server
```

Stop the server with <kbd>Ctrl</kbd>+<kbd>C</kbd>.

### Linux with NVIDIA (e.g. DGX Spark) — Docker

```bash
docker compose up -d --build
```

The app is then at `http://<machine>:8756`. Models are kept in a Docker volume.
No GPU? Delete the `deploy:` block in `docker-compose.yml` first.

### First start

1. The speech model (**Whisper large-v3**, ~3 GB) downloads by itself — watch it
   in the **Models** tab. Transcription works once it is loaded (the Dashboard
   then shows it under "STT model loaded").
2. Speaker detection works straight away with the built-in model.
3. *Recommended:* switch on the better **pyannote community-1** speaker model:
   - Make a free token at <https://huggingface.co/settings/tokens>
   - Accept the terms on [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - Paste the token in **Settings → Hugging Face token**, then download the
     model in **Models**. After that, everything runs offline.

### Your first transcript

1. Open the **Transcribe** tab.
2. Drag an audio or video file onto the box (or click it to choose).
3. Pick the language and number of speakers, or leave both on *Auto-detect*.
4. Click **Start transcribing** and watch the progress bar.
5. When it is done: click a line to play it, double-click to fix it, click a
   speaker chip to rename the speaker, and click **DOCX** to export to Word.

Speaker count wrong? Click **👥 Re-detect speakers** — no need to transcribe
again. Everything else is in the [user guide](docs/USER_GUIDE.md).

### Update to the latest version

```bash
cd SST_transcript
git pull
```

Then stop the server (<kbd>Ctrl</kbd>+<kbd>C</kbd>) and start it again with
`./start.sh` or `start.command` (both also install new Python packages). If you
start with `uv run sst-server`, run `uv sync` first. Your settings, jobs and
models are kept.

---

## Use it from other computers (network & login)

- **Same machine:** `http://localhost:<port>`.
- **Other devices on WiFi/Ethernet:** on the same router/subnet, open
  `http://<LAN-IP>:<port>` (find the IP on the Dashboard's "Share access" card).
  mDNS `http://<hostname>.local:<port>` works on Apple devices and most other OSes.
- **Tailscale / VPN:** the Dashboard also lists the `100.x.y.z` address, which
  works from anywhere on your tailnet.
- **macOS firewall:** System Settings → Network → Firewall. Either keep it on and, on
  first launch, click **Allow incoming connections** for the Python/uv process, or add
  it under *Options…*. Linux ufw: `sudo ufw allow <port>/tcp`.
- **Enable login:** Settings → Access & security → set an API key → turn on
  "Require login for other devices". This machine (localhost) is never asked.
- **API with the key:**
  ```bash
  curl -H "Authorization: Bearer YOUR_KEY" http://<LAN-IP>:<port>/v1/models
  ```
  OpenAI SDK: `OpenAI(base_url="http://<LAN-IP>:<port>/v1", api_key="YOUR_KEY")`.
  Browsers log in once at `/login`; the session lasts 24 h.
- **Change port:** Settings → Server port → Save & restart. Models stay loaded.
- Traffic is plain HTTP — fine inside a trusted network or tailnet. For the open
  internet, put Caddy/nginx in front for HTTPS.

---

## How it works

```
audio file ─ ffmpeg → 16 kHz mono
             ├─ Silero VAD → silence-aware ≤28 s chunks (long-file support, progress %)
             ├─ Diarization (pyannote or built-in) → speaker turns
             └─ STT per chunk (Whisper on CUDA/MPS/CPU) → words with timestamps
                        └─ words × turns overlap → segments split at speaker changes
                                     → JSON / VTT / SRT / DOCX with {start, end, speaker, text}
```

- **One code base for every machine:** PyTorch, with the device picked at
  start (`cuda` → `mps` → `cpu`). Works on a Mac today and a DGX box tomorrow.
- Jobs run one at a time on a worker thread; the web app stays responsive.
  Progress and time-left come from measured speed.
- **Re-detect speakers** re-runs only the diarization step and re-assigns the
  stored words, so fixing the speaker count is much faster than a new transcript.

## Configuration

Everything can be changed in the web app. Settings are saved in `data/config.json`:

| Key | Default | Meaning |
|---|---|---|
| `stt_model` | `openai/whisper-large-v3` | Speech-to-text model |
| `diarization_model` | `pyannote/speaker-diarization-community-1` | Speaker model (falls back to the built-in one until a Hugging Face token is saved) |
| `device_override` | auto | Force `cuda` / `mps` / `cpu` |
| `port` | `8756` | Server port |
| `max_jobs` | `5` | Finished jobs kept on disk (3–20); the oldest are deleted |
| `auth_enabled` | `false` | Require the API key / a login for other computers |
| `api_key` | `""` | The key for the Bearer header and the login page |
| `access_log` | `quiet` | Request lines in the log: `quiet` (hide routine page refreshes), `full`, `off` |

Environment variables: `SST_DATA_DIR` (where settings, jobs and logs live),
`HF_HOME` (model cache).

### Logs

The server writes to the terminal **and** to `data/logs/sst.log` (rotated at
5 MB, 3 old files kept, so at most ~20 MB). In the default *quiet* mode it no
longer prints a `GET /api/jobs 200 OK` line every few seconds for every open
browser — only uploads, changes, job start/finish, and errors. View, download
and clear the log on the **Dashboard**, and switch to *full* there when you need
every request.

## Troubleshooting

| Problem | Fix |
|---|---|
| "ffmpeg is required" | `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Linux) |
| "Model … is not downloaded yet" | Open **Models** and download it. The default model downloads by itself on the first start — wait for it |
| Speaker model (pyannote) won't load | Save a Hugging Face token in **Settings** *and* accept the model terms on huggingface.co. Until then the built-in speaker model is used automatically |
| Wrong number of speakers | **👥 Re-detect speakers** on the transcript, or merge two speakers by renaming one to the other's name |
| Cantonese comes out as standard written Chinese | Set Language to Cantonese. See [MODELS.md](docs/MODELS.md) for Cantonese-focused models |
| Very slow / out of memory | Use `whisper-large-v3-turbo` (Models tab) and close other big apps — a 16 GB Mac swaps hard with large-v3 plus a browser and an IDE |
| Cancel says "Cancelling…" for a while | It stops at the next safe point (1–3 s normally). Only model loading can't be interrupted |
| Other computers can't connect | Check the address on **Dashboard → Share access**, the firewall, and that both are on the same network or tailnet |
| Upgraded from an old version and pyannote 3.1 is still selected | Switch to community-1 in **Models** |
| Something else | **Dashboard → Server log** shows what happened. If the server seems frozen, run `kill -USR1 <server pid>` — every thread's stack is printed to the terminal; include it when reporting the problem |

---

## Model licensing / commercial use

Every model card in the Models tab shows its license. Verified summary (July 2026):

| Model | License | Commercial use |
|---|---|---|
| Whisper (all sizes) | Apache-2.0 | ✅ Yes |
| pyannote **community-1** | CC-BY-4.0 | ✅ Yes, **with attribution** — credit "pyannote speaker-diarization-community-1" in your product's docs/about page |
| pyannote **3.1** | MIT | ✅ Yes, no conditions |
| Built-in diarizer (Silero VAD / SpeechBrain ECAPA / scikit-learn) | MIT / Apache-2.0 / BSD | ✅ Yes |
| SenseVoice Small | FunASR Model License | ⚠️ **Ambiguous** — confirm with FunAudioLLM before shipping commercially |
| NVIDIA Sortformer v1 (not included) | CC-BY-**NC**-4.0 | ❌ No — this is why it's not in the catalog |

More, including newer models checked in October 2026: [docs/MODELS.md](docs/MODELS.md).

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
