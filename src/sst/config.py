"""Persistent app configuration, stored as JSON under ./data/config.json."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA_DIR = Path(os.environ.get("SST_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
CONFIG_PATH = DATA_DIR / "config.json"

_lock = threading.Lock()


@dataclass
class AppConfig:
    stt_model: str = "openai/whisper-large-v3"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    hf_token: str = ""
    device_override: str = ""  # "", "cuda", "mps", "cpu"
    host: str = "0.0.0.0"
    port: int = 8756
    default_output_format: str = "json"
    max_jobs: int = 5  # finished jobs (and their audio) kept on disk; 3..20
    extra: dict = field(default_factory=dict)

    def clamped_max_jobs(self) -> int:
        return max(3, min(20, int(self.max_jobs or 5)))

    def save(self) -> None:
        with _lock:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls) -> "AppConfig":
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text())
                known = {f for f in cls.__dataclass_fields__}
                return cls(**{k: v for k, v in raw.items() if k in known})
            except Exception:
                pass
        return cls()


config = AppConfig.load()
