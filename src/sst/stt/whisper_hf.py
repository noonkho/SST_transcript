"""Whisper via Hugging Face transformers. Works on CUDA, MPS (Apple Silicon), and CPU."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

# The ASR pipeline passes return_token_timestamps internally for word
# timestamps; transformers warns it will change in v5 (we pin <5).
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")

# The ASR pipeline imports torchcodec on *every* call just to check whether
# the input is a torchcodec decoder. When torchcodec can't load its native
# libraries (e.g. Homebrew moved to an ffmpeg major version it doesn't
# support yet) that import raises and every transcription fails. We only ever
# pass in-memory arrays (ffmpeg decoding happens in sst.audio), so tell the
# pipeline torchcodec is absent.
import transformers.pipelines.automatic_speech_recognition as _asr_pipeline  # noqa: E402

_asr_pipeline.is_torchcodec_available = lambda: False

from ..audio import SAMPLE_RATE
from ..config import config
from ..device import pick_dtype
from .base import SttEngine, SttSegment, Word


class WhisperEngine(SttEngine):
    word_timestamps = True

    def __init__(self, repo_id: str, device: str) -> None:
        self.device = device
        self.dtype = pick_dtype(device)
        token = config.hf_token or None
        self.processor = AutoProcessor.from_pretrained(repo_id, token=token)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo_id, dtype=self.dtype, low_cpu_mem_usage=True, token=token,
        ).to(device)
        self.model.eval()
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device=device,
        )
        self.is_multilingual = getattr(self.model.config, "vocab_size", 0) >= 51865

    @torch.inference_mode()
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the dominant language of an audio chunk ('' if unavailable)."""
        if not self.is_multilingual:
            return "en"
        try:
            features = self.processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to(self.device, dtype=self.dtype)
            lang_ids = self.model.detect_language(features)
            token = self.processor.tokenizer.decode(lang_ids[0])
            return token.replace("<|", "").replace("|>", "")
        except Exception:
            return ""

    def transcribe_chunk(
        self, audio: np.ndarray, language: str | None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[SttSegment]:
        generate_kwargs: dict = {"task": "transcribe"}
        if language and language != "auto" and self.is_multilingual:
            generate_kwargs["language"] = language
        if should_stop is not None:
            # checked after every generated token, so Cancel takes effect in
            # well under a second even on a slow machine
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopWhen(should_stop)])

        result = self.pipe(
            {"array": audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
        )

        text = (result.get("text") or "").strip()
        if not text:
            return []

        duration = len(audio) / SAMPLE_RATE
        words: list[Word] = []
        for chunk in result.get("chunks") or []:
            w = chunk.get("text", "")
            if not w.strip():
                continue
            ts = chunk.get("timestamp") or (None, None)
            start = float(ts[0]) if ts[0] is not None else (words[-1].end if words else 0.0)
            end = float(ts[1]) if ts[1] is not None else min(start + 1.0, duration)
            words.append(Word(start=start, end=end, text=w))

        seg_start = words[0].start if words else 0.0
        seg_end = words[-1].end if words else duration
        return [SttSegment(start=seg_start, end=seg_end, text=text, words=words)]


class _StopWhen(StoppingCriteria):
    def __init__(self, should_stop: Callable[[], bool]) -> None:
        self.should_stop = should_stop

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> torch.BoolTensor:
        return torch.full((input_ids.shape[0],), bool(self.should_stop()),
                          dtype=torch.bool, device=input_ids.device)
