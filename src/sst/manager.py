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
    eta_seconds: float | None = None
    error: str = ""
    started_at: float = field(default_factory=time.time)


def _repo_cache_dir(repo_id: str):
    from pathlib import Path
    return Path(huggingface_hub.constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"


def _bytes_on_disk(repo_ids: list[str]) -> int:
    """Bytes downloaded so far (blobs + partial .incomplete files)."""
    total = 0
    for repo_id in repo_ids:
        blobs = _repo_cache_dir(repo_id) / "blobs"
        if blobs.exists():
            total += sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())
    return total


def _watch_download(state: DownloadState, targets: list[str]) -> None:
    """Filesystem-based progress: independent of the hub's download backend."""
    prev_bytes = _bytes_on_disk(targets)
    prev_t = time.time()
    rate = 0.0
    while state.status == "downloading":
        time.sleep(1.0)
        now_bytes = _bytes_on_disk(targets)
        now_t = time.time()
        delta = (now_bytes - prev_bytes) / max(now_t - prev_t, 1e-3)
        rate = 0.7 * rate + 0.3 * max(delta, 0.0)  # smoothed bytes/sec
        prev_bytes, prev_t = now_bytes, now_t
        state.downloaded_bytes = now_bytes
        if state.total_bytes:
            state.progress = min(0.999, now_bytes / state.total_bytes)
            if rate > 1024:
                state.eta_seconds = max(0.0, (state.total_bytes - now_bytes) / rate)


class ModelManager:
    def __init__(self) -> None:
        self.downloads: dict[str, DownloadState] = {}
        self._dl_lock = threading.Lock()

        self.stt_engine = None
        self.stt_repo: str | None = None
        self.diar_engine = None
        self.diar_repo: str | None = None
        # Serializes model loading/unloading WITH job execution: a model switch
        # requested mid-transcription waits until the running job finishes, so
        # two large models are never resident at the same time. RLock because
        # the pipeline holds it for the whole job and calls ensure_loaded inside.
        self.engines_lock = threading.RLock()

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
        import fnmatch

        targets = BUILTIN_DIARIZATION_DEPS if repo_id == "builtin/vad-ecapa-clustering" else [repo_id]
        try:
            api = HfApi(token=config.hf_token or None)
            plans: list[tuple[str, list[str]]] = []  # (target, ignore_patterns)
            for target in targets:
                ignore = list(IGNORE_ALWAYS)
                try:
                    info = api.model_info(target, files_metadata=True)
                    files = [s.rfilename for s in info.siblings]
                    if any(f.endswith(".safetensors") for f in files):
                        # safetensors present — skip the duplicate PyTorch .bin weights
                        ignore += ["*.bin", "pytorch_model*"]
                    state.total_bytes += sum(
                        s.size or 0 for s in info.siblings
                        if not any(fnmatch.fnmatch(s.rfilename, p) for p in ignore)
                    )
                except Exception:
                    pass  # offline or transient: download everything except IGNORE_ALWAYS
                plans.append((target, ignore))

            watcher = threading.Thread(target=_watch_download, args=(state, targets), daemon=True)
            watcher.start()
            for target, ignore in plans:
                snapshot_download(
                    target,
                    token=config.hf_token or None,
                    ignore_patterns=ignore,
                )
                self._mark_complete(target)
            self._mark_complete(repo_id)
            state.downloaded_bytes = max(state.downloaded_bytes, state.total_bytes)
            state.progress = 1.0
            state.eta_seconds = 0.0
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

    def custom_downloaded(self) -> list[dict]:
        """Downloaded repos that are not part of the curated catalog
        (added via Hugging Face search)."""
        from .registry import CATALOG
        known = {e.repo_id for e in CATALOG} | set(BUILTIN_DIARIZATION_DEPS)
        out = []
        for repo in sorted(self.downloaded_repos()):
            if repo in known or repo.startswith("builtin/"):
                continue
            engine = classify_hf_model(repo, [])
            out.append({"repo_id": repo, "engine": engine, "supported": engine is not None})
        return out

    def delete_model(self, repo_id: str) -> None:
        """Remove a downloaded model from the local cache to free storage."""
        if repo_id in (self.stt_repo, self.diar_repo):
            raise RuntimeError("This model is currently loaded — switch to another model first.")
        if repo_id in (config.stt_model, config.diarization_model):
            raise RuntimeError("This model is currently selected — select another model first.")
        targets = BUILTIN_DIARIZATION_DEPS if repo_id == "builtin/vad-ecapa-clustering" else [repo_id]
        info = huggingface_hub.scan_cache_dir()
        hashes = [
            rev.commit_hash
            for repo in info.repos if repo.repo_id in targets
            for rev in repo.revisions
        ]
        if hashes:
            info.delete_revisions(*hashes).execute()
        for target in [*targets, repo_id]:
            self._completed.discard(target)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        COMPLETED_PATH.write_text(json.dumps(sorted(self._completed)))
        self.downloads.pop(repo_id, None)  # so the UI offers "Download" again
        log.info("removed model %s from cache", repo_id)

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
            engine = classify_hf_model(m.id, m.tags or [], getattr(m, "library_name", None))
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
        with self.engines_lock:
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
