"""Transcription pipeline: decode → VAD chunking → diarization → STT → speaker merge."""

from __future__ import annotations

import logging
import time

import numpy as np

from .audio import decode_audio, duration_seconds
from .diarize.base import SpeakerTurn
from .jobs import Job
from .manager import manager
from .stt.base import SttSegment
from .vad import detect_speech, pack_chunks

log = logging.getLogger("sst.pipeline")

# Stage weights for the overall progress bar.
W_LOAD, W_DECODE, W_DIAR, W_STT = 0.05, 0.05, 0.15, 0.73


def run_transcription(job: Job, audio_path: str) -> dict:
    params = job.params
    language = params.get("language") or None
    num_speakers = params.get("num_speakers") or None
    diarize = params.get("diarize", True)
    stt_model = params.get("model") or None
    diar_model = params.get("diarization_model") or None

    job.stage = "loading models"
    job.progress = 0.0
    manager.ensure_loaded(stt_repo=stt_model, diar_repo=diar_model, load_diar=diarize)
    stt = manager.stt_engine
    job.progress = W_LOAD
    job.check_cancelled()

    job.stage = "decoding"
    audio = decode_audio(audio_path)
    total = duration_seconds(audio)
    job.audio_duration = total
    job.progress = W_LOAD + W_DECODE
    job.check_cancelled()

    speech = detect_speech(audio)
    chunks = pack_chunks(audio, speech)
    log.info("audio %.1fs → %d speech chunks", total, len(chunks))
    job.check_cancelled()

    turns: list[SpeakerTurn] = []
    if diarize and chunks:
        job.stage = "diarizing"
        turns = manager.diar_engine.diarize(audio, num_speakers=num_speakers)
    job.progress = W_LOAD + W_DECODE + W_DIAR
    job.check_cancelled()

    job.stage = "transcribing"
    segments: list[SttSegment] = []
    detected_language = ""
    chunk_times: list[float] = []
    processed_audio = 0.0
    total_chunk_audio = sum(c.end - c.start for c in chunks) or 1.0

    for i, chunk in enumerate(chunks):
        job.check_cancelled()
        t0 = time.time()
        for seg in stt.transcribe_chunk(chunk.audio, language):
            seg.start += chunk.start
            seg.end = min(seg.end + chunk.start, total)
            for w in seg.words:
                w.start += chunk.start
                w.end += chunk.start
            segments.append(seg)
            detected_language = detected_language or seg.language
        chunk_times.append(time.time() - t0)
        processed_audio += chunk.end - chunk.start

        frac = processed_audio / total_chunk_audio
        job.progress = W_LOAD + W_DECODE + W_DIAR + W_STT * frac
        if chunk_times:
            speed = processed_audio / max(sum(chunk_times), 1e-6)  # audio-seconds per wall-second
            job.eta_seconds = max(0.0, (total_chunk_audio - processed_audio) / max(speed, 1e-6))

    if not detected_language and segments and hasattr(stt, "detect_language") and chunks:
        detected_language = stt.detect_language(chunks[0].audio)

    job.stage = "finalizing"
    out_segments = _merge_speakers(segments, turns) if turns else [
        {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": "SPEAKER_00", "text": s.text.strip()}
        for s in segments if s.text.strip()
    ]

    return {
        "language": detected_language or (language or ""),
        "duration": round(total, 3),
        "text": " ".join(s["text"] for s in out_segments).strip(),
        "segments": out_segments,
        "speakers": sorted({s["speaker"] for s in out_segments}),
        "model": manager.stt_repo,
        "diarization_model": manager.diar_repo if turns else None,
    }


def _speaker_at(turns: list[SpeakerTurn], start: float, end: float) -> str | None:
    """Speaker with the largest temporal overlap with [start, end], else nearest turn."""
    best, best_overlap = None, 0.0
    for t in turns:
        overlap = min(end, t.end) - max(start, t.start)
        if overlap > best_overlap:
            best, best_overlap = t.speaker, overlap
    if best is not None:
        return best
    mid = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(t.start - mid), abs(t.end - mid)), default=None)
    return nearest.speaker if nearest else None


def _merge_speakers(segments: list[SttSegment], turns: list[SpeakerTurn]) -> list[dict]:
    """Assign speakers. With word timestamps, split segments at speaker changes."""
    out: list[dict] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        if seg.words:
            cur_words: list = []
            cur_speaker: str | None = None
            for w in seg.words:
                spk = _speaker_at(turns, w.start, w.end) or cur_speaker or "SPEAKER_00"
                if cur_speaker is None:
                    cur_speaker = spk
                if spk != cur_speaker and cur_words:
                    out.append(_words_to_segment(cur_words, cur_speaker))
                    cur_words = []
                    cur_speaker = spk
                cur_words.append(w)
            if cur_words:
                out.append(_words_to_segment(cur_words, cur_speaker or "SPEAKER_00"))
        else:
            spk = _speaker_at(turns, seg.start, seg.end) or "SPEAKER_00"
            out.append({
                "start": round(seg.start, 3), "end": round(seg.end, 3),
                "speaker": spk, "text": seg.text.strip(),
            })

    # Merge consecutive same-speaker fragments separated by < 1 s.
    merged: list[dict] = []
    for seg in out:
        if merged and seg["speaker"] == merged[-1]["speaker"] and seg["start"] - merged[-1]["end"] < 1.0:
            merged[-1]["end"] = seg["end"]
            joiner = "" if _is_cjk_boundary(merged[-1]["text"], seg["text"]) else " "
            merged[-1]["text"] = (merged[-1]["text"] + joiner + seg["text"]).strip()
        else:
            merged.append(dict(seg))
    return merged


def _words_to_segment(words: list, speaker: str) -> dict:
    return {
        "start": round(words[0].start, 3),
        "end": round(words[-1].end, 3),
        "speaker": speaker,
        "text": "".join(w.text for w in words).strip(),
    }


def _is_cjk_boundary(left: str, right: str) -> bool:
    if not left or not right:
        return True
    return _is_cjk(left[-1]) and _is_cjk(right[0])


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
        or 0xF900 <= code <= 0xFAFF or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
    )
