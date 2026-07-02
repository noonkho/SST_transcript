"""STT engine interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class SttSegment:
    start: float
    end: float
    text: str
    language: str = ""
    words: list[Word] = field(default_factory=list)


class SttEngine:
    """Transcribes a single chunk of 16 kHz mono audio (<= ~30 s)."""

    word_timestamps: bool = False

    def transcribe_chunk(self, audio: np.ndarray, language: str | None) -> list[SttSegment]:
        raise NotImplementedError
