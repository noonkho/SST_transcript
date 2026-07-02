"""Device selection: CUDA > MPS (Apple Silicon) > CPU."""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def pick_device(override: str | None = None) -> str:
    if override in ("cuda", "mps", "cpu"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    return torch.float16 if device in ("cuda", "mps") else torch.float32


def device_description(device: str) -> str:
    if device == "cuda":
        return f"NVIDIA GPU ({torch.cuda.get_device_name(0)})"
    if device == "mps":
        return "Apple Silicon GPU (Metal / MPS)"
    return "CPU"
