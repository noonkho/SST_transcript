"""Transcription pipeline: decode → VAD chunking → diarization → STT → speaker merge."""

from __future__ import annotations

import logging
import time

import numpy as np

from .audio import decode_audio, duration_seconds
from .config import config
from .diarize.base import SpeakerTurn
from .jobs import Job, JobCancelled
from .manager import manager
from .stt.base import SttSegment, Word
from .vad import detect_speech, pack_chunks

log = logging.getLogger("sst.pipeline")

# Stage weights for the overall progress bar.
W_LOAD, W_DECODE, W_DIAR, W_STT = 0.05, 0.05, 0.15, 0.73


def run_transcription(job: Job, audio_path: str) -> dict:
    # Hold the engines lock for the whole job: a model switch requested from the
    # UI mid-transcription waits until this job finishes instead of loading a
    # second multi-GB model alongside the one in use.
    with manager.engines_lock:
        return _run_locked(job, audio_path)


def _run_locked(job: Job, audio_path: str) -> dict:
    params = job.params
    language = params.get("language") or None
    diarize = params.get("diarize", True)
    stt_model = params.get("model") or None
    diar_model = params.get("diarization_model") or None

    warnings: list[dict] = []

    job.stage = "loading models"
    job.progress = 0.0
    # STT first and on its own: without it there is no transcript to return, so
    # a failure here is fatal. The diarizer is loaded separately below so that
    # its absence degrades the request instead of killing it.
    manager.ensure_loaded(stt_repo=stt_model, load_diar=False)
    # Capture engines and repo names now — manager state can't change while we
    # hold the engines lock, but the result must reflect what actually ran.
    stt = manager.stt_engine
    stt_repo_used = manager.stt_repo

    diar_engine = None
    diar_repo_used = None
    if diarize:
        wanted = diar_model or config.diarization_model
        try:
            manager.ensure_loaded(stt_repo=stt_model, diar_repo=diar_model, load_diar=True)
            diar_engine = manager.diar_engine
            diar_repo_used = manager.diar_repo
        except Exception as exc:  # noqa: BLE001
            log.warning("diarization unavailable (%s) — transcribing without speaker labels", exc)
            warnings.append({
                "code": "diarization_unavailable",
                "message": f"Diarization model {wanted} not loaded; segments processed "
                           "without speaker labels. Segments[].speaker is null.",
            })
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
    if diar_engine and chunks:
        turns, diar_repo_used = _diarize(job, diar_engine, diar_repo_used, audio, speech,
                                         W_LOAD + W_DECODE, W_DIAR, warnings)
    job.progress = W_LOAD + W_DECODE + W_DIAR
    job.check_cancelled()

    job.stage = "transcribing"
    segments: list[SttSegment] = []
    detected_language = ""
    chunk_times: list[float] = []
    processed_audio = 0.0
    total_chunk_audio = sum(c.end - c.start for c in chunks) or 1.0
    job.chunks_total = len(chunks)
    job.chunks_done = 0

    for i, chunk in enumerate(chunks):
        job.check_cancelled()
        t0 = time.time()
        try:
            chunk_segments = stt.transcribe_chunk(chunk.audio, language,
                                                  should_stop=lambda: job.cancel_requested)
        except Exception:
            job.check_cancelled()  # a chunk cut short by Cancel may fail to decode
            raise
        for seg in chunk_segments:
            seg.start += chunk.start
            seg.end = min(seg.end + chunk.start, total)
            for w in seg.words:
                w.start += chunk.start
                w.end += chunk.start
            segments.append(seg)
            detected_language = detected_language or seg.language
        job.check_cancelled()  # a stopped chunk is partial — never keep it
        chunk_times.append(time.time() - t0)
        processed_audio += chunk.end - chunk.start
        job.chunks_done = i + 1

        frac = processed_audio / total_chunk_audio
        job.progress = W_LOAD + W_DECODE + W_DIAR + W_STT * frac
        if chunk_times:
            speed = processed_audio / max(sum(chunk_times), 1e-6)  # audio-seconds per wall-second
            job.eta_seconds = max(0.0, (total_chunk_audio - processed_audio) / max(speed, 1e-6))

    if not detected_language and segments and hasattr(stt, "detect_language") and chunks:
        detected_language = stt.detect_language(chunks[0].audio)

    job.stage = "finalizing"
    # No turns (diarization off, unavailable, or it found nothing) => speaker is
    # genuinely unknown. Say so with null rather than inventing "SPEAKER_00".
    out_segments = _merge_speakers(segments, turns) if turns else [
        {"start": round(s.start, 3), "end": round(s.end, 3), "speaker": None,
         "text": s.text.strip(),
         "words": _word_list(s.words, s.start, s.end, s.text)}
        for s in segments if s.text.strip()
    ]
    speakers = sorted({s["speaker"] for s in out_segments if s["speaker"]}) or None

    return {
        "language": detected_language or (language or ""),
        "duration": round(total, 3),
        "text": " ".join(s["text"] for s in out_segments).strip(),
        "segments": out_segments,
        "speakers": speakers,
        "model": stt_repo_used,
        "diarization_model": diar_repo_used if turns else None,
        "warnings": warnings,
    }


def _diarize(job: Job, engine, repo: str | None, audio: np.ndarray,
             speech: list[tuple[float, float]], base: float, weight: float,
             warnings: list[dict]) -> tuple[list[SpeakerTurn], str | None]:
    """Run the diarizer with live progress and cancellation.

    `progress(frac)` is called by the engine from inside its loops; it raises
    JobCancelled when the user pressed Cancel, so a long diarization stops
    within a second instead of only after it finishes. Any other failure
    degrades to "no speaker labels" instead of losing the transcript.
    """
    params = job.params
    job.stage = "diarizing"

    def progress(frac: float) -> None:
        job.progress = base + weight * min(max(frac, 0.0), 1.0)
        job.check_cancelled()

    try:
        turns = engine.diarize(
            audio, num_speakers=params.get("num_speakers") or None, speech=speech,
            min_speakers=params.get("min_speakers") or None,
            max_speakers=params.get("max_speakers") or None,
            progress=progress,
        )
        return turns, repo
    except JobCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        # Loaded but blew up mid-inference — still return the transcript.
        log.warning("diarization failed (%s) — returning transcript without labels", exc)
        warnings.append({
            "code": "diarization_unavailable",
            "message": f"Diarization model {repo} failed: {exc}. Segments "
                       "processed without speaker labels. Segments[].speaker is null.",
        })
        return [], None


def run_rediarization(job: Job, audio_path: str, source: dict) -> dict:
    """Re-detect speakers for a finished transcript without re-running speech-to-text.

    Only the diarizer runs (minutes faster than a full transcription); the words
    and timings from `source` (the earlier result) are re-assigned to the new
    speaker turns. Used when the first run guessed the wrong number of speakers.
    """
    with manager.engines_lock:
        params = job.params
        warnings: list[dict] = []
        job.stage = "loading models"
        manager.ensure_loaded(diar_repo=params.get("diarization_model") or None,
                              load_stt=False, load_diar=True)
        engine, repo = manager.diar_engine, manager.diar_repo
        job.progress = W_LOAD
        job.check_cancelled()

        job.stage = "decoding"
        audio = decode_audio(audio_path)
        job.audio_duration = duration_seconds(audio)
        job.progress = 0.15
        job.check_cancelled()
        speech = detect_speech(audio)
        job.check_cancelled()

        turns, repo = _diarize(job, engine, repo, audio, speech, 0.2, 0.75, warnings)
        if not turns:
            raise RuntimeError(warnings[0]["message"] if warnings else
                               "The diarizer found no speech in this audio.")

        job.stage = "finalizing"
        segments = _merge_speakers(_segments_from_result(source.get("segments") or []), turns)
        result = {k: v for k, v in source.items()
                  if k not in ("segments", "speakers", "speaker_colors", "edited", "warnings")}
        result.update({
            "segments": segments,
            "speakers": sorted({s["speaker"] for s in segments if s["speaker"]}) or None,
            "text": " ".join(s["text"] for s in segments).strip(),
            "diarization_model": repo,
            "warnings": warnings,
        })
        return result


def _segments_from_result(segments: list[dict]) -> list[SttSegment]:
    """Rebuild STT segments from a stored result so speakers can be re-assigned.

    Word timings are used when they still match the line's text; a line whose
    text was edited (or that comes from an older result with incomplete word
    lists) is kept whole and given to the speaker who talks most during it.
    """
    out: list[SttSegment] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        stored = seg.get("words") or []
        words: list[Word] = []
        if len(stored) > 1 and _squash("".join(w.get("word", "") for w in stored)) == _squash(text):
            prev = ""
            for w in stored:
                token = str(w["word"])
                # stored words are stripped; restore the space between Latin words
                if prev and not _is_cjk_boundary(prev, token):
                    token = " " + token
                words.append(Word(start=float(w["start"]), end=float(w["end"]), text=token))
                prev = str(w["word"])
        out.append(SttSegment(start=float(seg["start"]), end=float(seg["end"]),
                              text=text, words=words))
    return out


def _squash(text: str) -> str:
    return "".join(text.split())


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
                "words": _word_list(seg.words, seg.start, seg.end, seg.text),
            })

    # Merge consecutive same-speaker fragments separated by < 1 s.
    merged: list[dict] = []
    for seg in out:
        if merged and seg["speaker"] == merged[-1]["speaker"] and seg["start"] - merged[-1]["end"] < 1.0:
            merged[-1]["end"] = seg["end"]
            joiner = "" if _is_cjk_boundary(merged[-1]["text"], seg["text"]) else " "
            merged[-1]["text"] = (merged[-1]["text"] + joiner + seg["text"]).strip()
            merged[-1]["words"] = merged[-1]["words"] + seg["words"]
        else:
            merged.append(dict(seg))
    return merged


def _word_list(words: list, start: float, end: float, text: str) -> list[dict]:
    """Word timings for the API. Engines without word timestamps (SenseVoice)
    degrade to a single entry spanning the segment."""
    if not words:
        return [{"word": text.strip(), "start": round(start, 3), "end": round(end, 3)}]
    return [{"word": w.text.strip(), "start": round(w.start, 3), "end": round(w.end, 3)}
            for w in words if w.text.strip()]


def _words_to_segment(words: list, speaker: str) -> dict:
    text = "".join(w.text for w in words).strip()
    start, end = words[0].start, words[-1].end
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "speaker": speaker,
        "text": text,
        "words": _word_list(words, start, end, text),
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
