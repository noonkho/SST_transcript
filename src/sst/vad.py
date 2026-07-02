"""Voice activity detection (Silero VAD) and packing speech into chunks for STT.

Long files are split at natural silences into chunks of at most `max_chunk_s`
seconds so the STT model never sees a mid-word cut and progress can be
reported per chunk.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE

_vad_model = None
_vad_lock = threading.Lock()


@dataclass
class SpeechChunk:
    start: float  # seconds, position in the original audio
    end: float
    audio: np.ndarray


def _get_vad():
    global _vad_model
    with _vad_lock:
        if _vad_model is None:
            from silero_vad import load_silero_vad
            _vad_model = load_silero_vad()
    return _vad_model


def detect_speech(audio: np.ndarray) -> list[tuple[float, float]]:
    """Return [(start_s, end_s), ...] speech regions."""
    import torch
    from silero_vad import get_speech_timestamps

    model = _get_vad()
    with _vad_lock:
        timestamps = get_speech_timestamps(
            torch.from_numpy(audio), model,
            sampling_rate=SAMPLE_RATE, return_seconds=True,
            min_silence_duration_ms=300,
            speech_pad_ms=100,
        )
    return [(t["start"], t["end"]) for t in timestamps]


def pack_chunks(
    audio: np.ndarray,
    speech: list[tuple[float, float]],
    max_chunk_s: float = 28.0,
    max_gap_s: float = 1.0,
) -> list[SpeechChunk]:
    """Merge adjacent speech regions into chunks of at most max_chunk_s seconds."""
    if not speech:
        return []

    # First split any single region longer than the chunk limit.
    regions: list[tuple[float, float]] = []
    for start, end in speech:
        while end - start > max_chunk_s:
            regions.append((start, start + max_chunk_s))
            start += max_chunk_s
        regions.append((start, end))

    chunks: list[SpeechChunk] = []
    cur_start, cur_end = regions[0]
    for start, end in regions[1:]:
        gap = start - cur_end
        if end - cur_start <= max_chunk_s and gap <= max_gap_s:
            cur_end = end
        else:
            chunks.append(_slice(audio, cur_start, cur_end))
            cur_start, cur_end = start, end
    chunks.append(_slice(audio, cur_start, cur_end))
    return chunks


def _slice(audio: np.ndarray, start: float, end: float) -> SpeechChunk:
    i0 = max(0, int(start * SAMPLE_RATE))
    i1 = min(len(audio), int(end * SAMPLE_RATE))
    return SpeechChunk(start=start, end=end, audio=audio[i0:i1])
