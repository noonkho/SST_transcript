#!/bin/bash
# SST launcher for Linux (and macOS terminals): ./start.sh
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "→ Installing uv (one-time setup)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚠️  ffmpeg not found — install it first:"
  echo "   Ubuntu/Debian: sudo apt-get install -y ffmpeg"
  echo "   macOS:         brew install ffmpeg"
  exit 1
fi

echo "→ Preparing Python environment…"
uv sync --quiet

echo "→ Starting server (Ctrl+C to stop)…"
exec uv run sst-server
