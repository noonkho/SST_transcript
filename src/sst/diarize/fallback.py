"""Ungated built-in diarizer: Silero VAD + SpeechBrain ECAPA embeddings + clustering.

Works with zero setup (no HF token). Less accurate than pyannote on
overlapping speech, but solid for meetings and interviews.
"""

from __future__ import annotations

from typing import Callable

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
# A "speaker" holding less than this much total audio is a clustering artifact, not a person:
# 1.5s windows slide by 0.75s, so a single mis-clustered window at a speaker change invents a
# whole new speaker. Such clusters are folded into their nearest surviving cluster instead of
# being reported. Without this the auto path returned 10 speakers for a 2-speaker clip, because
# the MAX_AUTO_SPEAKERS guard below re-clustered to exactly the cap.
MIN_SPEAKER_S = 2.0


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
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: Callable[[float], None] | None = None,
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

        embeddings = self._embed(audio, windows, progress)
        labels = self._cluster(embeddings, num_speakers, min_speakers, max_speakers)
        if num_speakers is None:
            labels = _drop_tiny_clusters(labels, embeddings, windows, min_speakers)

        # Order speaker ids by first appearance and merge consecutive windows.
        rename: dict[int, str] = {}
        turns: list[SpeakerTurn] = []
        for (start, end), label in zip(windows, labels):
            if label not in rename:
                rename[label] = f"SPEAKER_{len(rename):02d}"
            name = rename[label]
            if turns and turns[-1].speaker == name and start <= turns[-1].end + STEP_S:
                turns[-1].end = max(turns[-1].end, end)
                continue
            if turns and start < turns[-1].end:
                # Analysis windows overlap by WINDOW_S - STEP_S, so a speaker change between two
                # adjacent windows used to emit turns that overlap in time — two people shown as
                # talking at once when only the window grid overlapped. Split the contested span
                # at its midpoint and give each side to one speaker. A turn squeezed to nothing
                # by this is dropped rather than emitted zero- or negative-length.
                boundary = (start + turns[-1].end) / 2
                turns[-1].end = boundary
                start = boundary
                if turns[-1].end <= turns[-1].start:
                    turns.pop()
            if end > start:
                turns.append(SpeakerTurn(start=start, end=end, speaker=name))
        return turns

    @torch.inference_mode()
    def _embed(self, audio: np.ndarray, windows: list[tuple[float, float]],
               progress: Callable[[float], None] | None = None) -> np.ndarray:
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
            if progress:
                progress(0.95 * min(i + 64, len(spans)) / len(spans))
        result = np.concatenate(embs, axis=0)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-8)

    @staticmethod
    def _cluster(
        embeddings: np.ndarray,
        num_speakers: int | None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> np.ndarray:
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
            return model.fit_predict(embeddings)

        labels = AgglomerativeClustering(
            n_clusters=None, distance_threshold=CLUSTER_THRESHOLD,
            metric="cosine", linkage="average",
        ).fit_predict(embeddings)

        # Bound the automatic estimate. The old code re-clustered to exactly MAX_AUTO_SPEAKERS
        # whenever the threshold over-segmented, so *any* over-segmented audio reported exactly
        # 10 speakers — a cap masquerading as an answer. Now the cap is a ceiling like any other,
        # and callers can narrow it (or raise the floor) with max_speakers / min_speakers.
        ceiling = min(max_speakers or MAX_AUTO_SPEAKERS, MAX_AUTO_SPEAKERS, n)
        floor = min(max(min_speakers or 1, 1), ceiling)
        found = int(labels.max()) + 1
        target = min(max(found, floor), ceiling)
        if target != found:
            labels = AgglomerativeClustering(
                n_clusters=target, metric="cosine", linkage="average",
            ).fit_predict(embeddings)
        return labels


def _drop_tiny_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    windows: list[tuple[float, float]],
    min_speakers: int | None,
) -> np.ndarray:
    """Fold clusters holding under MIN_SPEAKER_S of audio into their nearest surviving cluster.

    Sliding windows overlap, so a single mis-clustered window at a speaker change invents a
    speaker who "spoke" for half a second. Reporting those as people is worse than merging them:
    a transcript claiming ten speakers in a two-person interview is not usable, and the caller
    cannot tell which labels are real. Never drops below `min_speakers` (or below 1).
    """
    n_clusters = int(labels.max()) + 1
    if n_clusters <= 1:
        return labels

    duration = np.zeros(n_clusters)
    for (start, end), label in zip(windows, labels):
        duration[label] += end - start

    keep = [c for c in range(n_clusters) if duration[c] >= MIN_SPEAKER_S]
    floor = max(min_speakers or 1, 1)
    if len(keep) < floor:
        # Not enough substantial clusters — keep the longest ones instead of forcing a merge.
        keep = list(np.argsort(duration)[::-1][:min(floor, n_clusters)])
    if len(keep) == n_clusters:
        return labels
    if not keep:
        return np.zeros_like(labels)

    centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in keep])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-8)

    out = labels.copy()
    for c in range(n_clusters):
        if c in keep:
            continue
        members = labels == c
        # Cosine similarity of each orphaned window against the surviving centroids.
        nearest = np.argmax(embeddings[members] @ centroids.T, axis=1)
        out[members] = np.array(keep)[nearest]

    # Re-label to a dense 0..k-1 range so the caller's SPEAKER_NN numbering has no gaps.
    remap = {old: new for new, old in enumerate(sorted(set(out.tolist())))}
    return np.array([remap[v] for v in out.tolist()])
