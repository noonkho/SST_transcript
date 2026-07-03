"""Diarization engine interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # "SPEAKER_00", "SPEAKER_01", ...


class Diarizer:
    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None = None,
        speech: list[tuple[float, float]] | None = None,
    ) -> list[SpeakerTurn]:
        """`speech` are pre-computed VAD regions [(start_s, end_s), ...]; engines
        that have their own voice-activity detection may ignore it."""
        raise NotImplementedError
