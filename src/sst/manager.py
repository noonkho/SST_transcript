"""Model management: download (with progress), cache inspection, load/unload engines."""

from __future__ import annotations

import gc
import json
import logging
import threading
import time
from dataclasses import dataclass, field

import huggingface_hub
from huggingface_hub import HfApi, snapshot_download

from .config import DATA_DIR, config
from .device import pick_device
from .registry import BUILTIN_DIARIZATION_DEPS, CatalogEntry, classify_hf_model, find_entry

log = logging.getLogger("sst.manager")

COMPLETED_PATH = DATA_DIR / "downloaded_models.json"

# Redundant weight formats we never need (PyTorch backend, fp16/fp32-auto).
IGNORE_ALWAYS = ["*.msgpack", "*.h5", "*.tflite", "*flax*", "*.onnx", "*.mlmodelc*", "*fp32*"]


# ---------------------------------------------------------------- downloads

@dataclass
class DownloadState:
    repo_id: str
    status: str = "downloading"      # downloading | done | error
    progress: float = 0.0            # 0..1
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: str = ""
    started_at: float = field(default_factory=time.time)


class _ProgressTqdm(huggingface_hub.utils.tqdm):
    """tqdm subclass that reports snapshot_download progress to a DownloadState."""

    _state: DownloadState | None = None  # set per-download via closure class

    def update(self, n=1):
        super().update(n)
        state = type(self)._state
        # Only track the outer bytes-level bar (unit=B); ignore the file-count bar.
        if state is not None and self.unit == "B" and self.total:
            state.downloaded_bytes = int(self.n)
            state.total_bytes = int(self.total)
            state.progress = min(1.0, self.n / self.total)


class ModelManager:
    def __init__(self) -> None:
        self.downloads: dict[str, DownloadState] = {}
        self._dl_lock = threading.Lock()

        self.stt_engine = None
        self.stt_repo: str | None = None
        self.diar_engine = None
        self.diar_repo: str | None = None
        self._load_lock = threading.Lock()

        self._completed: set[str] = set()
        if COMPLETED_PATH.exists():
            try:
                self._completed = set(json.loads(COMPLETED_PATH.read_text()))
            except Exception:
                pass

    # ---------------- cache inspection

    def _mark_complete(self, repo_id: str) -> None:
        self._completed.add(repo_id)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        COMPLETED_PATH.write_text(json.dumps(sorted(self._completed)))

    def downloaded_repos(self) -> set[str]:
        """Repos we know are fully downloaded (verified by our own download path).

        A partially-cached repo (interrupted download) is NOT considered
        downloaded; re-running its download resumes and verifies it.
        """
        try:
            cached = {r.repo_id for r in huggingface_hub.scan_cache_dir().repos if r.size_on_disk > 0}
        except Exception:
            cached = set()
        result = self._completed & cached
        if all(dep in result for dep in BUILTIN_DIARIZATION_DEPS):
            result.add("builtin/vad-ecapa-clustering")
        else:
            result.discard("builtin/vad-ecapa-clustering")
        return result

    # ---------------- downloads

    def start_download(self, repo_id: str) -> DownloadState:
        with self._dl_lock:
            existing = self.downloads.get(repo_id)
            if existing and existing.status == "downloading":
                return existing
            state = DownloadState(repo_id=repo_id)
            self.downloads[repo_id] = state
        threading.Thread(target=self._download, args=(repo_id, state), daemon=True).start()
        return state

    def _download(self, repo_id: str, state: DownloadState) -> None:
        targets = BUILTIN_DIARIZATION_DEPS if repo_id == "builtin/vad-ecapa-clustering" else [repo_id]
        try:
            api = HfApi(token=config.hf_token or None)
            for target in targets:
                ignore = list(IGNORE_ALWAYS)
                try:
                    files = api.list_repo_files(target)
                    if any(f.endswith(".safetensors") for f in files):
                        # safetensors present — skip the duplicate PyTorch .bin weights
                        ignore += ["*.bin", "pytorch_model*"]
                except Exception:
                    pass  # offline or transient: download everything except IGNORE_ALWAYS
                tqdm_cls = type("Bar", (_ProgressTqdm,), {"_state": state})
                snapshot_download(
                    target,
                    token=config.hf_token or None,
                    ignore_patterns=ignore,
                    tqdm_class=tqdm_cls,
                )
                self._mark_complete(target)
            self._mark_complete(repo_id)
            state.progress = 1.0
            state.status = "done"
            # If this is one of the configured models, load it right away.
            if repo_id in (config.stt_model, config.diarization_model):
                try:
                    self.ensure_loaded()
                except Exception as exc:  # noqa: BLE001
                    log.warning("auto-load after download failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            state.status = "error"
            msg = str(exc)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                state.error = (
                    f"'{repo_id}' is gated. Accept its terms on huggingface.co and add your "
                    "HF token in Settings, then retry."
                )
            else:
                state.error = msg.splitlines()[0][:300] if msg else "download failed"
            log.exception("download failed for %s", repo_id)

    # ---------------- HF search

    def search_hub(self, query: str, limit: int = 20) -> list[dict]:
        api = HfApi(token=config.hf_token or None)
        downloaded = self.downloaded_repos()
        results = []
        models = api.list_models(
            search=query,
            pipeline_tag="automatic-speech-recognition",
            sort="downloads", direction=-1, limit=limit,
        )
        for m in models:
            engine = classify_hf_model(m.id, m.tags or [])
            results.append({
                "repo_id": m.id,
                "downloads": m.downloads,
                "likes": m.likes,
                "gated": bool(getattr(m, "gated", False)),
                "engine": engine,
                "compatible": engine is not None,
                "downloaded": m.id in downloaded,
            })
        return results

    # ---------------- engine loading

    def ensure_loaded(
        self,
        stt_repo: str | None = None,
        diar_repo: str | None = None,
        load_diar: bool = True,
    ) -> None:
        """Load (or switch) the STT and diarization engines. Blocking; serialized."""
        stt_repo = stt_repo or config.stt_model
        diar_repo = diar_repo or config.diarization_model
        with self._load_lock:
            if self.stt_repo != stt_repo or self.stt_engine is None:
                self._unload_stt()
                self.stt_engine = self._build_stt(stt_repo)
                self.stt_repo = stt_repo
            if load_diar and (self.diar_repo != diar_repo or self.diar_engine is None):
                self._unload_diar()
                try:
                    self.diar_engine = self._build_diar(diar_repo)
                    self.diar_repo = diar_repo
                except Exception as exc:
                    # Keep transcription usable out of the box: if the configured
                    # diarizer (e.g. gated pyannote before a token is saved) can't
                    # load but the built-in one can, use it. The result JSON
                    # reports which diarization model actually ran.
                    builtin = "builtin/vad-ecapa-clustering"
                    if diar_repo != builtin and builtin in self.downloaded_repos():
                        log.warning("diarizer %s unavailable (%s) — falling back to %s",
                                    diar_repo, exc, builtin)
                        self.diar_engine = self._build_diar(builtin)
                        self.diar_repo = builtin
                    else:
                        raise

    def _require_downloaded(self, repo_id: str) -> None:
        if repo_id not in self.downloaded_repos():
            raise RuntimeError(
                f"Model '{repo_id}' is not downloaded yet. "
                "Open the Models tab and download it first."
            )

    def _build_stt(self, repo_id: str):
        self._require_downloaded(repo_id)
        entry = find_entry(repo_id)
        engine_name = entry.engine if entry else classify_hf_model(repo_id, []) or "whisper"
        device = pick_device(config.device_override or None)
        log.info("loading STT model %s (engine=%s, device=%s)", repo_id, engine_name, device)
        if engine_name == "sensevoice":
            from .stt.sensevoice import SenseVoiceEngine
            return SenseVoiceEngine(repo_id, device)
        from .stt.whisper_hf import WhisperEngine
        return WhisperEngine(repo_id, device)

    def _build_diar(self, repo_id: str):
        self._require_downloaded(repo_id)
        device = pick_device(config.device_override or None)
        log.info("loading diarization model %s (device=%s)", repo_id, device)
        if repo_id == "builtin/vad-ecapa-clustering":
            from .diarize.fallback import BuiltinDiarizer
            return BuiltinDiarizer(device)
        from .diarize.pyannote_engine import PyannoteDiarizer
        return PyannoteDiarizer(repo_id, device, token=config.hf_token or None)

    def _unload_stt(self) -> None:
        if self.stt_engine is not None:
            self.stt_engine = None
            self.stt_repo = None
            self._free_memory()

    def _unload_diar(self) -> None:
        if self.diar_engine is not None:
            self.diar_engine = None
            self.diar_repo = None
            self._free_memory()

    @staticmethod
    def _free_memory() -> None:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def status(self) -> dict:
        device = pick_device(config.device_override or None)
        from .device import device_description
        return {
            "device": device,
            "device_description": device_description(device),
            "stt_loaded": self.stt_repo,
            "diarization_loaded": self.diar_repo,
        }


manager = ModelManager()
