# Models: choosing, adding, licensing

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

## Adding models from Hugging Face / removing models

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

## Offline / air-gapped use

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

## Model licensing / commercial use

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

## Attribution

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

## What is the built-in diarizer?

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


## Newer models worth knowing about (checked October 2026)

The current defaults (Whisper large-v3 + pyannote community-1) are still a good,
commercially safe pair. This is what else exists and whether it would help.

### Speech-to-text

| Model | License | Good at | Fit for SST |
|---|---|---|---|
| **Qwen3-ASR-1.7B** (+ Qwen3-ForcedAligner-0.6B for word times) | Apache-2.0 ✅ | 30 languages + 22 Chinese dialects. Published Cantonese error rate (Common Voice yue) **7.6 %** vs Whisper large-v3 **16.2 %**; Mandarin (AISHELL-2) 2.7 % vs 5.1 %; English about equal | **Best upgrade candidate for Cantonese.** PyTorch-based, so it fits the cross-platform design. Needs a new engine in `src/sst/stt/`; the aligner handles ≤ 5 min per call, so it must run per chunk (SST already chunks to ≤ 28 s). Not built in yet |
| **Gemma 4** E2B / E4B / 12B (audio input) | Apache-2.0 ✅ | General multimodal LLM that can also transcribe | **Not a good fit.** Max 30 s of audio per request, **no timestamps** (so no karaoke, no speaker alignment), no published Cantonese accuracy. Better used *after* transcription, e.g. for summaries |
| **oMLX 0.7.0** | — | A model *server* for Apple's MLX; this release fixed Gemma 4 audio | Apple-only runtime. SST is PyTorch so the same code runs on Mac, NVIDIA (DGX) and CPU — adopting MLX would split the code base |
| Whisper large-v3-turbo | Apache-2.0 ✅ | ~4× faster, near-same accuracy | Already in the catalog |

### Speaker diarization

| Model | License | Notes |
|---|---|---|
| **pyannote community-1** (current default) | CC-BY-4.0 ✅ (attribution) | Still the best open model you may use commercially; no speaker limit |
| NVIDIA streaming Sortformer 4spk-v2 | CC-BY-4.0 ✅ | Very fast, but **max 4 speakers**; needs NVIDIA NeMo. Newer *Nemotron-3-Diarization* (8 speakers) exists — check its license first |
| DiariZen (BUT) | weights **CC-BY-NC** ❌ | Accurate, but non-commercial only |

Sources: [Qwen3-ASR model card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) ·
[Qwen3-ASR paper](https://arxiv.org/abs/2601.21337) ·
[Gemma audio docs](https://ai.google.dev/gemma/docs/capabilities/audio) ·
[oMLX 0.7.0 release](https://github.com/jundot/omlx/releases/tag/v0.7.0) ·
[pyannote community-1](https://www.pyannote.ai/blog/community-1) ·
[Sortformer v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) ·
[DiariZen](https://github.com/BUTSpeechFIT/DiariZen)
