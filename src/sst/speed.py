"""Learned processing speed, for "how long will this take?" before a job runs.

Every finished job records how many seconds each stage needed per second of
audio, per model and device (an exponential moving average, so the numbers
follow hardware or model changes within a few jobs). Estimates use those
ratios; until a combination has been seen once, conservative defaults apply
and the estimate is flagged as rough.
"""

from __future__ import annotations

import json
import logging
import threading

from .config import DATA_DIR, config
from .device import pick_device

log = logging.getLogger("sst.speed")

SPEED_PATH = DATA_DIR / "speed.json"
_ALPHA = 0.4            # weight of the newest job in the moving average
_MIN_AUDIO_S = 20.0     # shorter clips are dominated by fixed overhead — don't learn from them
OVERHEAD_S = 4.0        # decode + VAD + finalize, roughly constant
# seconds of work per second of audio before anything was measured
_DEFAULTS = {
    "gpu": {"stt": 0.25, "diar": 0.05},
    "cpu": {"stt": 1.5, "diar": 0.3},
}

_lock = threading.Lock()


def _load() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in json.loads(SPEED_PATH.read_text()).items()}
    except (OSError, ValueError, AttributeError):
        return {}


_ratios = _load()


def _key(stage: str, repo: str) -> str:
    return f"{stage}|{repo}|{pick_device(config.device_override or None)}"


def _ratio(stage: str, repo: str) -> tuple[float, bool]:
    learned = _ratios.get(_key(stage, repo))
    if learned is not None:
        return learned, True
    hw = "cpu" if pick_device(config.device_override or None) == "cpu" else "gpu"
    return _DEFAULTS[hw][stage], False


def record(stage: str, repo: str | None, seconds: float, audio_seconds: float) -> None:
    if not repo or audio_seconds < _MIN_AUDIO_S or seconds <= 0:
        return
    key, ratio = _key(stage, repo), seconds / audio_seconds
    with _lock:
        old = _ratios.get(key)
        _ratios[key] = ratio if old is None else (1 - _ALPHA) * old + _ALPHA * ratio
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            SPEED_PATH.write_text(json.dumps(_ratios, indent=1))
        except OSError as exc:
            log.warning("could not save speed stats: %s", exc)


def stage_seconds(stage: str, repo: str, audio_seconds: float) -> float:
    return _ratio(stage, repo)[0] * audio_seconds


def estimate(audio_seconds: float, diarize: bool = True,
             stt_repo: str | None = None, diar_repo: str | None = None) -> dict:
    """Expected wall time for one file, and whether it comes from measurements."""
    stt_repo = stt_repo or config.stt_model
    diar_repo = diar_repo or config.diarization_model
    stt, learned = _ratio("stt", stt_repo)
    total = OVERHEAD_S + stt * audio_seconds
    if diarize:
        diar, diar_learned = _ratio("diar", diar_repo)
        total += diar * audio_seconds
        learned = learned and diar_learned
    return {"seconds": round(total, 1), "learned": learned}
