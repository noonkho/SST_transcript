"""Learned processing speed, for "how long will this take?" before a job runs.

Each stage's time is modelled as  seconds = fixed + per_second × audio_seconds:
a short clip is dominated by the fixed setup cost, a long recording by the
per-second cost. Every finished job adds one (audio_seconds, seconds) point
per stage, model and device; a least-squares line through the recent points
(older ones fade out) gives both numbers. Until a combination has been seen,
conservative defaults apply and the estimate is flagged as rough.
"""

from __future__ import annotations

import json
import logging
import threading

from .config import DATA_DIR, config
from .device import pick_device

log = logging.getLogger("sst.speed")

SPEED_PATH = DATA_DIR / "speed.json"
_FORMAT = 3          # bump to discard data stored under different rules
_DECAY = 0.75        # weight left on older jobs each time a new one is added
_MIN_AUDIO_S = 20.0  # tiny clips say little about speed
OVERHEAD_S = 4.0     # decode + VAD + finalize, roughly constant
# per second of audio, before anything was measured
_DEFAULTS = {
    "gpu": {"stt": 0.25, "diar": 0.05},
    "cpu": {"stt": 1.5, "diar": 0.3},
}

_lock = threading.Lock()


def _load() -> dict[str, dict[str, float]]:
    try:
        raw = json.loads(SPEED_PATH.read_text())
        if raw.get("format") != _FORMAT:
            return {}  # stored by an older version
        return {k: {s: float(v) for s, v in sums.items()} for k, sums in raw["stats"].items()}
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return {}


# per key: decayed sums n, x, y, xx, xy of (audio_seconds, seconds) points
_stats = _load()


def _key(stage: str, repo: str) -> str:
    return f"{stage}|{repo}|{pick_device(config.device_override or None)}"


def _fit(stats: dict[str, float]) -> tuple[float, float]:
    """(fixed, per_second) from the decayed sums; through the origin while
    the recordings so far are all about the same length."""
    n, x, y, xx, xy = (stats[k] for k in ("n", "x", "y", "xx", "xy"))
    var = n * xx - x * x
    if var > 1e-6 * max(xx * n, 1.0):
        slope = (n * xy - x * y) / var
        fixed = (y - slope * x) / n
        if slope > 0 and fixed >= 0:
            return fixed, slope
    return 0.0, y / x if x else 0.0


def stage_seconds(stage: str, repo: str, audio_seconds: float) -> tuple[float, bool]:
    """Expected seconds for one stage, and whether it was learned."""
    stats = _stats.get(_key(stage, repo))
    if stats:
        fixed, per_second = _fit(stats)
        return fixed + per_second * audio_seconds, True
    hw = "cpu" if pick_device(config.device_override or None) == "cpu" else "gpu"
    return _DEFAULTS[hw][stage] * audio_seconds, False


def record(stage: str, repo: str | None, seconds: float, audio_seconds: float) -> None:
    if not repo or audio_seconds < _MIN_AUDIO_S or seconds <= 0:
        return
    with _lock:
        s = _stats.setdefault(_key(stage, repo), dict.fromkeys(("n", "x", "y", "xx", "xy"), 0.0))
        for k in s:
            s[k] *= _DECAY
        s["n"] += 1
        s["x"] += audio_seconds
        s["y"] += seconds
        s["xx"] += audio_seconds * audio_seconds
        s["xy"] += audio_seconds * seconds
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            SPEED_PATH.write_text(json.dumps({"format": _FORMAT, "stats": _stats}, indent=1))
        except OSError as exc:
            log.warning("could not save speed stats: %s", exc)


def estimate(audio_seconds: float, diarize: bool = True,
             stt_repo: str | None = None, diar_repo: str | None = None) -> dict:
    """Expected wall time for one file, and whether it comes from measurements."""
    total, learned = stage_seconds("stt", stt_repo or config.stt_model, audio_seconds)
    if diarize:
        diar, diar_learned = stage_seconds("diar", diar_repo or config.diarization_model,
                                           audio_seconds)
        total += diar
        learned = learned and diar_learned
    return {"seconds": round(OVERHEAD_S + total, 1), "learned": learned}
