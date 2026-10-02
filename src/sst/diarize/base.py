"""Diarization engine interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[SpeakerTurn]:
        """`speech` are pre-computed VAD regions [(start_s, end_s), ...]; engines
        that have their own voice-activity detection may ignore it.

        `num_speakers` pins the count exactly. `min_speakers`/`max_speakers` bound it while
        leaving the engine to estimate within those bounds — the usual case, since a caller
        normally knows "at least two people are talking" but not the exact number. All three
        are None by default (fully automatic).

        `progress(fraction_done)` should be called regularly from long loops; it
        may raise (that is how a running job is cancelled), so let it propagate.
        """
        raise NotImplementedError
