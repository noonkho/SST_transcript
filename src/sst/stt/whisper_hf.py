"""Whisper via Hugging Face transformers. Works on CUDA, MPS (Apple Silicon), and CPU."""

from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# The ASR pipeline passes return_token_timestamps internally for word
# timestamps; transformers warns it will change in v5 (we pin <5).
warnings.filterwarnings("ignore", message=".*return_token_timestamps.*")

from .. import _compat  # noqa: F401  — before the ASR pipeline runs
from ..audio import SAMPLE_RATE
from ..config import config
from ..device import pick_dtype
from .base import SttEngine, SttSegment, Word
from .cancel import cancel_criteria

log = logging.getLogger("sst.whisper")


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

    def _supported_language(self, language: str | None) -> str | None:
        """The language token to force, or None for auto-detect. Cantonese
        (`yue`) exists only in large-v3 models; older ones get Chinese instead
        of failing the whole job."""
        if not language or language == "auto" or not self.is_multilingual:
            return None
        known = getattr(self.model.generation_config, "lang_to_id", None) or {}
        if not known or f"<|{language}|>" in known:
            return language
        fallback = "zh" if language == "yue" and "<|zh|>" in known else None
        log.warning("this Whisper model has no '%s' language token — using %s instead",
                    language, fallback or "auto-detect")
        return fallback

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
        check_cancelled: Callable[[], None] | None = None,
    ) -> list[SttSegment]:
        generate_kwargs: dict = {"task": "transcribe"}
        language = self._supported_language(language)
        if language:
            generate_kwargs["language"] = language
        if check_cancelled is not None:
            generate_kwargs["stopping_criteria"] = cancel_criteria(check_cancelled)

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

