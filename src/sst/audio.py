"""Audio decoding. Uses ffmpeg to convert any input format to 16 kHz mono float32."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".mov", ".webm", ".mkv", ".aiff", ".aif", ".caf", ".amr", ".3gp",
}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_audio(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any audio/video file to mono float32 PCM at the given sample rate."""
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to decode audio. Install it with: "
            "brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)."
        )
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg failed to decode audio: {err[-1] if err else 'unknown error'}")
    # copy() makes the buffer writable, which torch.from_numpy requires
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty — is this a valid audio file?")
    return audio


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return float(len(audio)) / sample_rate
