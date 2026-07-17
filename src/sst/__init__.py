"""SST — local offline speech-to-text with speaker diarization."""

import os

# Use the plain HTTP downloader instead of the Xet backend: it streams into
# .incomplete files on disk, which gives byte-accurate download progress and
# resumable downloads. Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "1.0.0"
