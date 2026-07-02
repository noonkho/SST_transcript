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
    ) -> list[SpeakerTurn]:
        raise NotImplementedError
