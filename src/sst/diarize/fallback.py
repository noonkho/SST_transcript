"""Ungated built-in diarizer: Silero VAD + SpeechBrain ECAPA embeddings + clustering.

Works with zero setup (no HF token). Less accurate than pyannote on
overlapping speech, but solid for meetings and interviews.
"""

from __future__ import annotations

import numpy as np
import torch

from ..audio import SAMPLE_RATE
from ..vad import detect_speech
from .base import Diarizer, SpeakerTurn

WINDOW_S = 1.5
STEP_S = 0.75
MIN_WINDOW_S = 0.4
# Cosine-distance threshold for deciding "same speaker" when num_speakers is unknown.
CLUSTER_THRESHOLD = 0.65
MAX_AUTO_SPEAKERS = 10


class BuiltinDiarizer(Diarizer):
    def __init__(self, device: str) -> None:
        from speechbrain.inference.speaker import EncoderClassifier

        run_opts = {"device": device if device != "mps" else "cpu"}
        # ECAPA is tiny; CPU is fast enough and avoids MPS op gaps in speechbrain.
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts=run_opts,
        )

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None = None,
        speech: list[tuple[float, float]] | None = None,
    ) -> list[SpeakerTurn]:
        if speech is None:
            speech = detect_speech(audio)
        if not speech:
            return []

        windows: list[tuple[float, float]] = []
        for seg_start, seg_end in speech:
            t = seg_start
            while t < seg_end:
                end = min(t + WINDOW_S, seg_end)
                if end - t >= MIN_WINDOW_S:
                    windows.append((t, end))
                t += STEP_S
        if not windows:
            windows = [(s, e) for s, e in speech]

        embeddings = self._embed(audio, windows)
        labels = self._cluster(embeddings, num_speakers)

        # Order speaker ids by first appearance and merge consecutive windows.
        rename: dict[int, str] = {}
        turns: list[SpeakerTurn] = []
        for (start, end), label in zip(windows, labels):
            if label not in rename:
                rename[label] = f"SPEAKER_{len(rename):02d}"
            name = rename[label]
            if turns and turns[-1].speaker == name and start <= turns[-1].end + STEP_S:
                turns[-1].end = max(turns[-1].end, end)
            else:
                turns.append(SpeakerTurn(start=start, end=end, speaker=name))
        return turns

    @torch.inference_mode()
    def _embed(self, audio: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
        embs = []
        spans = []
        for start, end in windows:
            i0, i1 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
            spans.append(audio[i0:i1])
        # Batch in groups of 64, padded to the longest clip in the group.
        for i in range(0, len(spans), 64):
            group = spans[i:i + 64]
            max_len = max(len(g) for g in group)
            padded = np.zeros((len(group), max_len), dtype=np.float32)
            lens = torch.ones(len(group))
            for j, g in enumerate(group):
                padded[j, :len(g)] = g
                lens[j] = len(g) / max_len
            wavs = torch.from_numpy(padded)
            out = self.encoder.encode_batch(wavs, wav_lens=lens).squeeze(1).cpu().numpy()
            embs.append(out)
        result = np.concatenate(embs, axis=0)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-8)

    @staticmethod
    def _cluster(embeddings: np.ndarray, num_speakers: int | None) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        n = len(embeddings)
        if n == 1:
            return np.zeros(1, dtype=int)
        if num_speakers == 1:
            return np.zeros(n, dtype=int)
        if num_speakers:
            model = AgglomerativeClustering(
                n_clusters=min(num_speakers, n), metric="cosine", linkage="average",
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=None, distance_threshold=CLUSTER_THRESHOLD,
                metric="cosine", linkage="average",
            )
        labels = model.fit_predict(embeddings)
        # Guard against over-segmentation in auto mode.
        if num_speakers is None and labels.max() + 1 > MAX_AUTO_SPEAKERS:
            model = AgglomerativeClustering(
                n_clusters=MAX_AUTO_SPEAKERS, metric="cosine", linkage="average",
            )
            labels = model.fit_predict(embeddings)
        return labels
