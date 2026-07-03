"""Child process that runs one snapshot_download.

Downloads run in a separate process (not a thread) so that cancelling a
download can actually stop it: the parent terminates this process and cleans
up the partial files. Progress is tracked by the parent watching the disk.

Usage: python -m sst._dl_worker <repo_id> <ignore_patterns_json>
The HF token, if any, is taken from the HF_TOKEN environment variable.
"""

from __future__ import annotations

import json
import os
import sys

from huggingface_hub import snapshot_download


def main() -> int:
    repo_id, ignore_json = sys.argv[1], sys.argv[2]
    snapshot_download(
        repo_id,
        ignore_patterns=json.loads(ignore_json),
        token=os.environ.get("HF_TOKEN") or None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
