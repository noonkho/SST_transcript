"""Speaker diarization via pyannote.audio (gated model, needs a free HF token once)."""

from __future__ import annotations

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
    ) -> list[SpeakerTurn]:
        waveform = torch.from_numpy(audio).unsqueeze(0)
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
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
