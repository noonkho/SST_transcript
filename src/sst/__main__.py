"""Entry point: `uv run sst-server` or `python -m sst`."""

from __future__ import annotations

import uvicorn

from .config import config


def main() -> None:
    uvicorn.run("sst.server:app", host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
