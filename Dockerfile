# SST — Linux deployment (CPU or NVIDIA GPU).
# On NVIDIA hosts (e.g. DGX Spark) run with:  docker compose up   (GPU wired in compose)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    SST_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv manages Python itself (no system python needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml .python-version ./
COPY src ./src

RUN uv sync --no-dev

EXPOSE 8756
VOLUME ["/cache", "/data"]

CMD ["uv", "run", "sst-server"]
