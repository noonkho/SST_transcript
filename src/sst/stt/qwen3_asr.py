"""Qwen3-ASR (Apache-2.0) via the `qwen-asr` package — strong Cantonese/Mandarin.

Qwen3-ASR itself returns text only; word timings come from the separate
Qwen3-ForcedAligner model (downloaded together with it, see registry). The
aligner drops punctuation, so its tokens are matched back onto the transcript
to keep commas and full stops in the words that SST joins into lines.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Callable

import numpy as np
import torch

from .. import _compat  # noqa: F401
from ..audio import SAMPLE_RATE
from ..registry import QWEN3_ASR_LANGUAGES
from .base import SttEngine, SttSegment, Word
from .cancel import cancel_criteria

log = logging.getLogger("sst.qwen3asr")

DEFAULT_ALIGNER = "Qwen/Qwen3-ForcedAligner-0.6B"

_TO_QWEN = QWEN3_ASR_LANGUAGES
_FROM_QWEN = {v: k for k, v in _TO_QWEN.items()}


class Qwen3AsrEngine(SttEngine):
    word_timestamps = True

    def __init__(self, repo_id: str, device: str, aligner_repo: str = DEFAULT_ALIGNER) -> None:
        from qwen_asr import Qwen3ASRModel

        # Qwen is trained in bfloat16; float16 can overflow. CPU stays float32.
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
        load = {"dtype": dtype, "device_map": device}
        self.asr = Qwen3ASRModel.from_pretrained(
            repo_id, forced_aligner=aligner_repo, forced_aligner_kwargs=load,
            max_inference_batch_size=1, max_new_tokens=512, **load,
        )
        self._check: Callable[[], None] | None = None
        # qwen-asr calls model.generate() itself with fixed arguments; wrap it
        # once so a cancel check can run between tokens, like for Whisper.
        generate = self.asr.model.generate

        def generate_with_cancel(*args, **kwargs):
            if self._check is not None:
                kwargs["stopping_criteria"] = cancel_criteria(self._check)
            return generate(*args, **kwargs)

        self.asr.model.generate = generate_with_cancel

    def transcribe_chunk(
        self, audio: np.ndarray, language: str | None,
        check_cancelled: Callable[[], None] | None = None,
    ) -> list[SttSegment]:
        self._check = check_cancelled
        try:
            res = self.asr.transcribe(
                audio=(audio, SAMPLE_RATE), language=_TO_QWEN.get(language or ""),
                return_time_stamps=True,
            )[0]
        finally:
            self._check = None
        text = (res.text or "").strip()
        if not text:
            return []
        detected = _FROM_QWEN.get((res.language or "").split(",")[0].strip(), "")
        words = _words_with_punctuation(text, res.time_stamps.items if res.time_stamps else [])
        duration = len(audio) / SAMPLE_RATE
        start = words[0].start if words else 0.0
        end = words[-1].end if words else duration
        return [SttSegment(start=start, end=end, text=text, language=detected, words=words)]


def _words_with_punctuation(text: str, items) -> list[Word]:
    """Map aligner tokens onto `text`.

    The aligner drops punctuation, even inside words ("one-time" -> "onetime"),
    so each token is located character by character, skipping punctuation. Each
    word then takes the text from its own start up to the next word's start, so
    punctuation and spaces stay attached and "".join(words) == text. Returns []
    (segment-level timing only) when the tokens can't be found in order.
    """
    spans: list[tuple[int, float, float]] = []
    cursor = 0
    for item in items:
        token = (item.text or "").strip()
        if not token:
            continue
        found = _locate(text, token, cursor)
        if found is None:
            log.debug("aligner token %r not found in transcript; using line timing", token)
            return []
        pos, cursor = found
        spans.append((pos, float(item.start_time), float(item.end_time)))
    words: list[Word] = []
    for i, (pos, start, end) in enumerate(spans):
        stop = spans[i + 1][0] if i + 1 < len(spans) else len(text)
        begin = 0 if i == 0 else pos  # leading punctuation joins the first word
        words.append(Word(start=start, end=max(end, start), text=text[begin:stop]))
    return words


def _is_word_char(ch: str) -> bool:
    # what the aligner keeps: letters, digits, apostrophes
    return ch == "'" or unicodedata.category(ch)[0] in "LN"


def _locate(text: str, token: str, cursor: int) -> tuple[int, int] | None:
    """(start, end) of `token` in text[cursor:], ignoring punctuation inside it."""
    for start in range(cursor, len(text)):
        if text[start] != token[0]:
            continue
        i, k = start, 0
        while i < len(text) and k < len(token):
            if text[i] == token[k]:
                k += 1
            elif _is_word_char(text[i]):
                break
            i += 1
        if k == len(token):
            return start, i
    return None
