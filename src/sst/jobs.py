"""Job store with disk persistence, cancellation, and a single worker queue.

Finished jobs (transcript + original audio) persist across server restarts in
DATA_DIR/jobs and DATA_DIR/audio, so the player and editor keep working after
a restart. Only the newest `config.max_jobs` finished jobs are kept.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audio import COMPRESSED_EXTENSIONS, transcode_for_storage
from .config import DATA_DIR, config

log = logging.getLogger("sst.jobs")

JOBS_DIR = DATA_DIR / "jobs"
AUDIO_DIR = DATA_DIR / "audio"


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"          # queued | running | done | error | cancelled
    stage: str = "queued"
    progress: float = 0.0           # 0..1 overall
    eta_seconds: float | None = None
    elapsed_seconds: float = 0.0
    audio_duration: float = 0.0
    error: str = ""
    result: dict | None = None
    params: dict = field(default_factory=dict)
    audio_path: str = ""            # persisted original upload (for playback)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False

    def check_cancelled(self) -> None:
        """Called from the pipeline between stages/chunks."""
        if self.cancel_requested:
            raise JobCancelled()

    @property
    def finished(self) -> bool:
        return self.status in ("done", "error", "cancelled")

    def public(self, include_result: bool = False) -> dict:
        d = {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "eta_seconds": round(self.eta_seconds, 1) if self.eta_seconds is not None else None,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "audio_duration": round(self.audio_duration, 1),
            "error": self.error,
            "params": self.params,
            "has_audio": bool(self.audio_path and Path(self.audio_path).exists()),
            "created_at": self.created_at,
        }
        if include_result:
            d["result"] = self.result
        return d


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[tuple[Job, Callable[[Job], Any]]]" = queue.Queue()
        self._lock = threading.Lock()
        self._load_from_disk()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ---------------- persistence

    def _job_file(self, job_id: str) -> Path:
        return JOBS_DIR / f"{job_id}.json"

    def save(self, job: Job) -> None:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        data = job.public(include_result=True)
        data["audio_path"] = job.audio_path
        data["finished_at"] = job.finished_at
        self._job_file(job.id).write_text(json.dumps(data, ensure_ascii=False))

    def _load_from_disk(self) -> None:
        if not JOBS_DIR.exists():
            return
        for path in JOBS_DIR.glob("*.json"):
            try:
                raw = json.loads(path.read_text())
                job = Job(
                    id=raw["id"], filename=raw.get("filename", "audio"),
                    status=raw.get("status", "error"), stage=raw.get("stage", ""),
                    progress=raw.get("progress", 0.0),
                    elapsed_seconds=raw.get("elapsed_seconds", 0.0),
                    audio_duration=raw.get("audio_duration", 0.0),
                    error=raw.get("error", ""), result=raw.get("result"),
                    params=raw.get("params", {}), audio_path=raw.get("audio_path", ""),
                    created_at=raw.get("created_at", path.stat().st_mtime),
                    finished_at=raw.get("finished_at"),
                )
                if not job.finished:  # was interrupted by a restart
                    job.status = "error"
                    job.stage = "error"
                    job.error = "Interrupted by a server restart."
                    job.finished_at = job.finished_at or time.time()
                    self.save(job)
                self.jobs[job.id] = job
            except Exception:  # noqa: BLE001
                log.warning("could not load job file %s", path)

    # ---------------- lifecycle

    def submit(self, filename: str, params: dict, fn: Callable[[Job], Any],
               audio_src: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, params=params)
        if audio_src:
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            suffix = Path(filename).suffix.lower() or Path(audio_src).suffix or ".bin"
            dest = AUDIO_DIR / f"{job.id}{suffix}"
            shutil.move(audio_src, dest)  # rename() would fail across filesystems
            job.audio_path = str(dest)
        with self._lock:
            self.jobs[job.id] = job
        self.save(job)
        self._queue.put((job, fn))
        return job

    def cancel(self, job_id: str) -> Job | None:
        job = self.jobs.get(job_id)
        if not job or job.finished:
            return job
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = time.time()
            self.save(job)
        return job

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        if job.status == "running":
            raise RuntimeError("Job is running — cancel it first.")
        if job.status == "queued":
            job.cancel_requested = True
            job.status = "cancelled"
        with self._lock:
            self.jobs.pop(job_id, None)
        self._job_file(job_id).unlink(missing_ok=True)
        if job.audio_path:
            Path(job.audio_path).unlink(missing_ok=True)
        return True

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update_result(self, job: Job, result: dict) -> None:
        job.result = result
        self.save(job)

    def enforce_limit(self) -> None:
        keep = config.clamped_max_jobs()
        finished = sorted(
            (j for j in self.jobs.values() if j.finished),
            key=lambda j: j.finished_at or j.created_at, reverse=True,
        )
        for job in finished[keep:]:
            try:
                self.delete(job.id)
            except Exception:  # noqa: BLE001
                pass

    # ---------------- worker

    def _run(self) -> None:
        while True:
            job, fn = self._queue.get()
            if job.status != "queued":  # cancelled/deleted while waiting
                continue
            job.status = "running"
            job.started_at = time.time()
            ticker = threading.Event()

            def tick(j: Job = job, stop: threading.Event = ticker) -> None:
                while not stop.is_set():
                    if j.started_at:
                        j.elapsed_seconds = time.time() - j.started_at
                    stop.wait(0.5)

            threading.Thread(target=tick, daemon=True).start()
            try:
                job.result = fn(job)
                job.status = "done"
                job.stage = "done"
                job.progress = 1.0
                job.eta_seconds = 0.0
            except JobCancelled:
                job.status = "cancelled"
                job.stage = "cancelled"
            except Exception as exc:  # noqa: BLE001
                job.status = "error"
                job.stage = "error"
                job.error = str(exc)[:500]
            finally:
                ticker.set()
                job.finished_at = time.time()
                job.elapsed_seconds = job.finished_at - (job.started_at or job.finished_at)
                if job.status == "done":
                    self._compress_audio(job)
                else:
                    # No transcript to play along to — don't keep the audio.
                    if job.audio_path:
                        Path(job.audio_path).unlink(missing_ok=True)
                        job.audio_path = ""
                self.save(job)
                self.enforce_limit()
                self._release_memory()

    @staticmethod
    def _compress_audio(job: Job) -> None:
        """Shrink the stored copy: uncompressed uploads (wav/flac/aiff…) are
        re-encoded to mono AAC for playback (a 1 GB wav becomes ~40 MB)."""
        src = Path(job.audio_path) if job.audio_path else None
        if not src or not src.exists() or src.suffix.lower() in COMPRESSED_EXTENSIONS:
            return
        dest = src.with_suffix(".m4a")
        if transcode_for_storage(str(src), str(dest)):
            src.unlink(missing_ok=True)
            job.audio_path = str(dest)
        else:
            dest.unlink(missing_ok=True)  # keep the original if transcoding failed

    @staticmethod
    def _release_memory() -> None:
        try:
            from .manager import manager
            manager._free_memory()  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass


jobs = JobStore()
