"""STT engine interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

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

    def transcribe_chunk(
        self, audio: np.ndarray, language: str | None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> list[SttSegment]:
        """`check_cancelled()` raises when the job was cancelled; engines that
        generate token by token should call it inside their loop."""
        raise NotImplementedError
