"""Curated catalog of known-good models plus helpers to classify arbitrary HF models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CatalogEntry:
    repo_id: str
    kind: str                    # "stt" | "diarization"
    engine: str                  # "whisper" | "sensevoice" | "pyannote" | "builtin"
    display_name: str
    languages: str
    size: str
    strengths: str
    license: str = ""            # SPDX-ish id + commercial-use note, shown in UI
    gated: bool = False
    word_timestamps: bool = False
    requires_extra: str = ""     # optional uv extra needed
    tags: list[str] = field(default_factory=list)


STT_CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        repo_id="openai/whisper-large-v3",
        kind="stt", engine="whisper",
        display_name="Whisper large-v3",
        languages="99 languages incl. Cantonese, Mandarin, English",
        size="~3.1 GB",
        strengths="Best overall multilingual accuracy; reliable word-level timestamps. Default choice.",
        license="Apache-2.0 — commercial use allowed",
        word_timestamps=True,
        tags=["default", "multilingual"],
    ),
    CatalogEntry(
        repo_id="openai/whisper-large-v3-turbo",
        kind="stt", engine="whisper",
        display_name="Whisper large-v3 Turbo",
        languages="99 languages incl. Cantonese, Mandarin, English",
        size="~1.6 GB",
        strengths="~4x faster than large-v3 with near-equal accuracy. Best for long recordings.",
        license="Apache-2.0 — commercial use allowed",
        word_timestamps=True,
        tags=["fast", "multilingual"],
    ),
    CatalogEntry(
        repo_id="openai/whisper-medium",
        kind="stt", engine="whisper",
        display_name="Whisper medium",
        languages="99 languages",
        size="~1.5 GB",
        strengths="Lighter option for CPU-only machines; noticeably lower accuracy on Cantonese.",
        license="Apache-2.0 — commercial use allowed",
        word_timestamps=True,
        tags=["light"],
    ),
    CatalogEntry(
        repo_id="openai/whisper-tiny",
        kind="stt", engine="whisper",
        display_name="Whisper tiny",
        languages="99 languages (low accuracy)",
        size="~150 MB",
        strengths="For quick smoke tests only.",
        license="Apache-2.0 — commercial use allowed",
        word_timestamps=True,
        tags=["test"],
    ),
    CatalogEntry(
        repo_id="FunAudioLLM/SenseVoiceSmall",
        kind="stt", engine="sensevoice",
        display_name="SenseVoice Small",
        languages="Mandarin, Cantonese, English, Japanese, Korean",
        size="~1 GB",
        strengths="Industry favourite for Cantonese/Mandarin/English CODE-SWITCHING; very fast even on CPU. "
                  "Segment-level timestamps only (no word timestamps).",
        license="FunASR Model License — commercial use AMBIGUOUS; verify with FunAudioLLM before commercial deployment",
        requires_extra="sensevoice",
        tags=["code-switching", "fast"],
    ),
]

DIARIZATION_CATALOG: list[CatalogEntry] = [
    CatalogEntry(
        repo_id="pyannote/speaker-diarization-community-1",
        kind="diarization", engine="pyannote",
        display_name="pyannote community-1",
        languages="Language-independent",
        size="~30 MB",
        strengths="Best open-source diarization accuracy (2026 benchmarks); handles overlapping "
                  "speech. Gated: requires a free Hugging Face token (one-time). Default choice.",
        license="CC-BY-4.0 — commercial use allowed with attribution (see README)",
        gated=True,
        tags=["default"],
    ),
    CatalogEntry(
        repo_id="pyannote/speaker-diarization-3.1",
        kind="diarization", engine="pyannote",
        display_name="pyannote speaker-diarization 3.1",
        languages="Language-independent",
        size="~30 MB",
        strengths="Previous-generation pyannote; slightly lower accuracy than community-1. "
                  "Gated: requires a free Hugging Face token (one-time).",
        license="MIT — commercial use allowed, no attribution required",
        gated=True,
        tags=[],
    ),
    CatalogEntry(
        repo_id="builtin/vad-ecapa-clustering",
        kind="diarization", engine="builtin",
        display_name="Built-in (VAD + ECAPA + clustering)",
        languages="Language-independent",
        size="~90 MB",
        strengths="No token needed — works out of the box. Silero VAD + SpeechBrain ECAPA speaker "
                  "embeddings + clustering. Lower accuracy on overlapping speech than pyannote.",
        license="MIT (Silero VAD) + Apache-2.0 (ECAPA) + BSD (scikit-learn) — commercial use allowed",
        tags=["no-token"],
    ),
]

CATALOG: list[CatalogEntry] = STT_CATALOG + DIARIZATION_CATALOG

# Repos the builtin diarizer needs downloaded.
BUILTIN_DIARIZATION_DEPS = ["speechbrain/spkrec-ecapa-voxceleb"]


def find_entry(repo_id: str) -> CatalogEntry | None:
    for entry in CATALOG:
        if entry.repo_id == repo_id:
            return entry
    return None


# Weight formats that need a different runtime (MLX, whisper.cpp, CTranslate2,
# ONNX…). This service runs everything through PyTorch — which itself covers
# Apple Silicon (MPS), NVIDIA (CUDA), and plain CPU — so these are rejected
# rather than failing confusingly at load time.
_FOREIGN_FORMAT_MARKERS = (
    "mlx", "gguf", "ggml", "ctranslate2", "-ct2", "ct2-", "faster-whisper",
    "onnx", "openvino", "coreml", "tflite", "tensorrt",
)


def classify_hf_model(repo_id: str, hf_tags: list[str], library: str | None = None) -> str | None:
    """Best-effort engine detection for models found via HF search.

    Returns the engine name for models this service can load (PyTorch /
    transformers format), or None for unsupported architectures and formats.
    """
    lowered = repo_id.lower()
    tagset = {t.lower() for t in hf_tags}
    if any(marker in lowered for marker in _FOREIGN_FORMAT_MARKERS):
        return None
    if tagset & set(_FOREIGN_FORMAT_MARKERS):
        return None
    if library and library.lower() not in ("transformers", "funasr"):
        return None
    if "whisper" in lowered or "whisper" in tagset:
        return "whisper"
    if "sensevoice" in lowered:
        return "sensevoice"
    return None
