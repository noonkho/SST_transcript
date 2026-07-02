"""SenseVoice-Small via FunASR — excellent Cantonese/Mandarin/English code-switching.

Optional: requires `uv sync --extra sensevoice`.
"""

from __future__ import annotations

import re

import numpy as np

from ..audio import SAMPLE_RATE
from .base import SttEngine, SttSegment

_TAG_RE = re.compile(r"<\|[^|]*\|>")

_LANG_MAP = {"yue": "yue", "zh": "zh", "en": "en", "ja": "ja", "ko": "ko"}


class SenseVoiceEngine(SttEngine):
    word_timestamps = False

    def __init__(self, repo_id: str, device: str) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "SenseVoice requires the optional 'funasr' dependency. "
                "Install it with: uv sync --extra sensevoice"
            ) from exc
        # FunASR uses "mps" via torch but is most stable on cpu/cuda; it is fast enough on CPU.
        dev = device if device in ("cuda", "cpu") else "cpu"
        self.model = AutoModel(
            model=repo_id,
            hub="hf",
            device=dev,
            disable_update=True,
            disable_log=True,
            disable_pbar=True,
        )

    def transcribe_chunk(self, audio: np.ndarray, language: str | None) -> list[SttSegment]:
        lang = _LANG_MAP.get(language or "", "auto")
        result = self.model.generate(
            input=audio.astype(np.float32),
            fs=SAMPLE_RATE,
            language=lang,
            use_itn=True,
        )
        if not result:
            return []
        raw = result[0].get("text", "")
        detected = ""
        m = re.match(r"<\|(\w+)\|>", raw)
        if m and m.group(1) in _LANG_MAP:
            detected = m.group(1)
        text = _TAG_RE.sub("", raw).strip()
        if not text:
            return []
        return [SttSegment(start=0.0, end=len(audio) / SAMPLE_RATE, text=text, language=detected)]
