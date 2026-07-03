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
    """Decode any audio/video file to mono float32 PCM at the given sample rate.

    Streams ffmpeg's output into a single growing buffer instead of buffering
    the whole thing twice — for hours-long recordings this halves peak memory.
    """
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required to decode audio. Install it with: "
            "brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)."
        )
    # -nostats/-loglevel error keep stderr tiny so its pipe can't fill up and
    # deadlock ffmpeg while we're draining stdout.
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "error",
        "-threads", "0",
        "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(sample_rate),
        "-vn", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buf = bytearray()
    assert proc.stdout is not None
    while chunk := proc.stdout.read(1 << 20):
        buf += chunk
    stderr = proc.stderr.read() if proc.stderr else b""
    if proc.wait() != 0:
        err = stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"ffmpeg failed to decode audio: {err[-1] if err else 'unknown error'}")
    buf = buf[: len(buf) - len(buf) % 4]  # guard against a truncated final sample
    audio = np.frombuffer(buf, dtype=np.float32)  # bytearray-backed → already writable
    if audio.size == 0:
        raise RuntimeError("Decoded audio is empty — is this a valid audio file?")
    return audio


# Formats that are already compressed enough to store as-is for playback.
COMPRESSED_EXTENSIONS = {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".wma", ".amr", ".3gp", ".mp4", ".mov", ".mkv"}


def transcode_for_storage(src: str, dest: str) -> bool:
    """Re-encode uncompressed audio (wav/flac/aiff…) to mono 96 kbps AAC for
    the playback archive. Returns True on success."""
    if not ffmpeg_available():
        return False
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", src, "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "96k", dest,
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return float(len(audio)) / sample_rate
