"""OpenAI-compatible surface for /v1: error envelope + model metadata.

Only /v1 speaks this dialect. The web UI's own /api endpoints keep FastAPI's
plain {"detail": ...} shape, which app.js reads.
"""

from __future__ import annotations

# response_format values /v1 accepts. Deliberately excludes "docx" (not an
# OpenAI format); DOCX export stays on /api/jobs/{id}/download.
OPENAI_FORMATS = ("json", "verbose_json", "text", "srt", "vtt")

# Advertised guidance for clients, in seconds. NOT enforced: the pipeline
# chunks on silence and handles multi-hour audio. It tells callers what a
# comfortable single request looks like.
MAX_AUDIO_SECONDS = 43200  # 12 h

# OpenAI error `type` values this service emits.
TYPE_INVALID_REQUEST = "invalid_request_error"
TYPE_INVALID_API_KEY = "invalid_api_key"
TYPE_RATE_LIMIT = "rate_limit_error"
TYPE_SERVER = "server_error"


class OpenAIError(Exception):
    """Raised inside /v1 handlers; rendered as the OpenAI error envelope."""

    def __init__(self, status: int, message: str, code: str,
                 type_: str = TYPE_INVALID_REQUEST) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.type = type_


def error_body(message: str, type_: str, code: str | None) -> dict:
    return {"error": {"message": message, "type": type_, "code": code}}


def type_for_status(status: int) -> str:
    if status == 401 or status == 403:
        return TYPE_INVALID_API_KEY
    if status == 429:
        return TYPE_RATE_LIMIT
    if status >= 500:
        return TYPE_SERVER
    return TYPE_INVALID_REQUEST


def code_for_status(status: int) -> str | None:
    return {
        401: "invalid_api_key",
        403: "invalid_api_key",
        404: "not_found",
        429: "rate_limit_exceeded",
        500: "internal_error",
    }.get(status)


def whisper_languages() -> list[str]:
    """Whisper's real language set, read from the tokenizer (99 codes incl. yue)."""
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        return sorted(LANGUAGES.keys())
    except Exception:
        return ["en", "yue", "zh"]


# SenseVoice ships a fixed 5-language model.
SENSEVOICE_LANGUAGES = ["en", "ja", "ko", "yue", "zh"]


def languages_for(engine: str) -> list[str]:
    if engine == "whisper":
        return whisper_languages()
    if engine == "sensevoice":
        return list(SENSEVOICE_LANGUAGES)
    return []


def model_entry(repo_id: str, kind: str, engine: str, *, loaded: bool) -> dict:
    """One /v1/models item, OpenAI-shaped plus this service's capability fields."""
    entry = {
        "id": repo_id,
        "object": "model",
        "created": 0,
        "owned_by": repo_id.split("/")[0],
        "kind": kind,
        "input_modality": "audio",
        "loaded": loaded,
    }
    if kind == "stt":
        # The service pairs any STT model with a diarizer, so transcription
        # requests against it can return speaker labels.
        entry["capabilities"] = ["transcription", "diarization"]
        entry["languages"] = languages_for(engine)
        entry["max_audio_length_seconds"] = MAX_AUDIO_SECONDS
    else:
        entry["capabilities"] = ["diarization"]
    return entry
