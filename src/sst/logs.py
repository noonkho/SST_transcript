"""Logging setup: quiet access log, rotating log file, in-memory tail for the UI.

The web UI polls /api/status and /api/jobs every few seconds from every open
browser tab, which used to flood the terminal with identical
`GET /api/jobs 200 OK` lines. The access log now has three modes
(config.access_log, changeable live from the Dashboard):

  quiet  (default) — hide successful GET requests; keep uploads, edits,
                     deletes, and every error (status >= 400)
  full             — every request, like plain uvicorn
  off              — no access lines at all (app messages still logged)

Everything that is shown also goes to DATA_DIR/logs/sst.log, which rotates at
5 MB and keeps 3 old files (≤ 20 MB on disk in total), and to a small
in-memory buffer that the Dashboard's "Server log" card reads.
"""

from __future__ import annotations

import collections
import logging
import logging.config
import logging.handlers
import threading

from .config import DATA_DIR, config

LOG_DIR = DATA_DIR / "logs"
LOG_FILE = LOG_DIR / "sst.log"
ACCESS_MODES = ("quiet", "full", "off")

_tail: collections.deque[str] = collections.deque(maxlen=500)
_tail_lock = threading.Lock()
_emitted = 0  # lines ever written to the tail — lets the UI skip unchanged refreshes


class QuietAccessFilter(logging.Filter):
    """Drops uvicorn access records according to config.access_log."""

    def filter(self, record: logging.LogRecord) -> bool:
        mode = config.access_log if config.access_log in ACCESS_MODES else "quiet"
        if mode == "full":
            return True
        if mode == "off":
            return False
        # uvicorn access args: (client_addr, method, full_path, http_version, status_code)
        args = record.args if isinstance(record.args, tuple) else ()
        if len(args) >= 5:
            method, status = args[1], args[4]
            try:
                return not (method == "GET" and int(status) < 400)
            except (TypeError, ValueError):
                return True
        return True


class TailHandler(logging.Handler):
    """Keeps the last few hundred formatted lines for GET /api/logs."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:  # noqa: BLE001
            return
        global _emitted
        with _tail_lock:
            _tail.append(line)
            _emitted += 1


def tail(lines: int = 300) -> tuple[list[str], int]:
    """The last `lines` lines, plus a counter that changes whenever a line is added."""
    with _tail_lock:
        return list(_tail)[-lines:], _emitted


def clear() -> None:
    """Empty the in-memory tail and start a fresh log file (old file kept as sst.log.1)."""
    global _emitted
    with _tail_lock:
        _tail.clear()
        _emitted += 1
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.doRollover()


def logging_config() -> dict:
    """dictConfig for the app *and* uvicorn (passed as uvicorn's log_config, so
    every in-process restart re-applies the same handlers and filters)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"quiet_access": {"()": QuietAccessFilter}},
        "formatters": {
            "app": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"},
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "formatter": "app",
                        "stream": "ext://sys.stderr"},
            "file": {"class": "logging.handlers.RotatingFileHandler", "formatter": "app",
                     "filename": str(LOG_FILE), "maxBytes": 5 * 1024 * 1024,
                     "backupCount": 3, "encoding": "utf-8"},
            "tail": {"()": TailHandler, "formatter": "app"},
        },
        "root": {"level": "INFO", "handlers": ["console", "file", "tail"]},
        "loggers": {
            # uvicorn's own handlers are dropped; everything flows to root.
            "uvicorn": {"level": "INFO", "handlers": [], "propagate": True},
            "uvicorn.error": {"level": "INFO", "handlers": [], "propagate": True},
            "uvicorn.access": {"level": "INFO", "handlers": [], "propagate": True,
                               "filters": ["quiet_access"]},
            # chatty third-party libraries
            "httpx": {"level": "WARNING"},
            "urllib3": {"level": "WARNING"},
            "speechbrain": {"level": "WARNING"},
        },
    }


def setup() -> None:
    logging.config.dictConfig(logging_config())
