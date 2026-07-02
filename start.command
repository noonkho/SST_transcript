#!/bin/bash
# SST launcher for macOS — double-click this file to start the server.
set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════"
echo "  SST — Local Speech-to-Text + Diarization"
echo "═══════════════════════════════════════════"

# 1. Ensure uv (Python package manager) is installed.
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "→ Installing uv (one-time setup)…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

# 2. Ensure ffmpeg is installed (needed to decode mp3/m4a/etc).
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "→ Installing ffmpeg (one-time setup)…"
    brew install ffmpeg
  else
    echo "⚠️  ffmpeg not found. Please install Homebrew (https://brew.sh)"
    echo "   then run:  brew install ffmpeg"
    read -r -p "Press Enter to continue anyway…"
  fi
fi

# 3. Install Python dependencies (cached after the first run).
echo "→ Preparing Python environment…"
uv sync --quiet

# 4. Start the server and open the UI.
PORT=$(uv run python -c "from sst.config import config; print(config.port)" 2>/dev/null || echo 8756)
echo "→ Starting server on http://localhost:$PORT  (Ctrl+C to stop)"
( sleep 3 && open "http://localhost:$PORT" ) &
exec uv run sst-server
