"""Work around a broken torchcodec install (one place, imported by the engines).

torchcodec comes in through pyannote-audio. It ships native libraries built
for specific ffmpeg major versions, so when Homebrew/apt moves to a newer
ffmpeg it can't load. Two things then go wrong, although SST never needs
torchcodec (audio is decoded with the ffmpeg CLI in sst.audio):

- pyannote prints a 60-line warning on import;
- transformers' ASR pipeline imports torchcodec on every call (it only
  checks that the *package* is installed) and every transcription fails.

Import this module before pyannote or the transformers ASR pipeline.
"""

from __future__ import annotations

import logging
import warnings

log = logging.getLogger("sst.compat")


def _torchcodec_loads() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import torchcodec.decoders  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — RuntimeError when its dylibs can't load
        return False


TORCHCODEC_OK = _torchcodec_loads()

if not TORCHCODEC_OK:
    warnings.filterwarnings("ignore", message=r"(?s).*torchcodec is not installed correctly.*")
    import transformers.pipelines.automatic_speech_recognition as _asr_pipeline

    if not hasattr(_asr_pipeline, "is_torchcodec_available"):
        raise RuntimeError("transformers changed: update the torchcodec workaround in sst/_compat.py")
    _asr_pipeline.is_torchcodec_available = lambda: False
    log.info("torchcodec can't load (ffmpeg version mismatch?) — not needed, "
             "SST decodes audio with the ffmpeg command instead")
