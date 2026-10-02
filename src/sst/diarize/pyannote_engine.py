"""Speaker diarization via pyannote.audio (gated model, needs a free HF token once)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..audio import SAMPLE_RATE
from .base import Diarizer, SpeakerTurn


class PyannoteDiarizer(Diarizer):
    def __init__(self, repo_id: str, device: str, token: str | None) -> None:
        from pyannote.audio import Pipeline

        try:
            # pyannote.audio 4.x renamed use_auth_token= to token=
            try:
                self.pipeline = Pipeline.from_pretrained(repo_id, token=token)
            except TypeError:
                self.pipeline = Pipeline.from_pretrained(repo_id, use_auth_token=token)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load '{repo_id}'. This model is gated on Hugging Face: "
                "accept its terms at https://huggingface.co/" + repo_id +
                " and set your HF token in Settings. Original error: " + str(exc)
            ) from exc
        if self.pipeline is None:
            raise RuntimeError(
                f"'{repo_id}' returned no pipeline — usually a missing/invalid HF token. "
                "Set your token in Settings and make sure you accepted the model terms."
            )
        self.pipeline.to(torch.device(device))

    def diarize(
        self,
        audio: np.ndarray,
        num_speakers: int | None = None,
        speech: list[tuple[float, float]] | None = None,  # unused; pyannote has its own VAD
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> list[SpeakerTurn]:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        else:
            # Only meaningful without num_speakers — pyannote rejects the combination.
            if min_speakers:
                kwargs["min_speakers"] = min_speakers
            if max_speakers:
                kwargs["max_speakers"] = max_speakers
        if progress:
            kwargs["hook"] = _progress_hook(progress)
        output = self.pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)

        # pyannote.audio 3.x returns an Annotation directly; 4.x (community-1)
        # wraps it in a DiarizeOutput with a .speaker_diarization attribute.
        annotation = getattr(output, "speaker_diarization", output)
        if not hasattr(annotation, "itertracks"):
            raise RuntimeError(
                f"Unexpected diarization output type {type(output).__name__} — "
                "this pyannote.audio version is not supported."
            )

        # Normalize speaker names to SPEAKER_00.. ordered by first appearance.
        turns: list[SpeakerTurn] = []
        rename: dict[str, str] = {}
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            if speaker not in rename:
                rename[speaker] = f"SPEAKER_{len(rename):02d}"
            turns.append(SpeakerTurn(
                start=float(segment.start), end=float(segment.end), speaker=rename[speaker],
            ))
        turns.sort(key=lambda t: t.start)
        return turns


# Share of the pipeline's run time each step takes (embeddings dominate).
_STEP_SPAN = {"segmentation": (0.0, 0.3), "embeddings": (0.3, 0.95)}


def _progress_hook(progress: Callable[[float], None]):
    """Adapt pyannote's hook(step, artifact, file=, total=, completed=) to a
    0..1 progress callback. pyannote calls it once per batch, so raising from
    `progress` (job cancelled) stops the pipeline within one batch."""
    last = [0.0]

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        span = _STEP_SPAN.get(step_name)
        if span and total:
            lo, hi = span
            last[0] = max(last[0], lo + (hi - lo) * (completed or 0) / total)
        progress(last[0])  # also the cancellation check for un-weighted steps
    return hook
