"""Entry point: `uv run sst-server` or `python -m sst`."""

from __future__ import annotations

import uvicorn

from .config import config
from .server import app


class WebServer:
    """Supervises the uvicorn HTTP listener so it can be rebound in-process
    (e.g. after a port change) without restarting the Python process — models
    already loaded in RAM (via `manager`) stay loaded."""

    def __init__(self):
        self._server: uvicorn.Server | None = None
        self._stop = False

    def request_restart(self):
        if self._server:
            self._server.should_exit = True   # unblocks .run()

    def run_forever(self):
        last_good_port = config.port
        while not self._stop:
            attempted_port = config.port   # snapshot: the port THIS iteration binds to
            cfg = uvicorn.Config(app, host=config.host, port=attempted_port, log_level="info")
            self._server = uvicorn.Server(cfg)
            try:
                self._server.run()             # blocks until should_exit
                # Only trust this as "good" if nothing changed config.port out
                # from under us while we were serving (avoids a TOCTOU race
                # with a concurrent /api/config port change).
                if config.port == attempted_port:
                    last_good_port = attempted_port
            except (OSError, SystemExit) as e:
                # Port busy / invalid: uvicorn's Server.run() reports bind
                # failures either as an OSError or (depending on version) by
                # calling sys.exit(1) internally, which raises SystemExit here
                # instead of propagating an OSError. Catch both and revert
                # rather than letting the whole process die.
                print(f"[web] cannot bind port {attempted_port}: {e}; reverting to {last_good_port}")
                config.port = last_good_port
                config.save()
                continue


web = WebServer()


def main() -> None:
    web.run_forever()


if __name__ == "__main__":
    main()
