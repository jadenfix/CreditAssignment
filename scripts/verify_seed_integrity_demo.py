#!/usr/bin/env python3
"""Verify media hashes, decoding, and captured source blobs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "media" / "demo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def main() -> None:
    receipt = json.loads((DEMO_DIR / "capture.json").read_text(encoding="utf-8"))
    source_commit = receipt["source_commit"]
    git("cat-file", "-e", f"{source_commit}^{{commit}}")
    assert receipt["source_tree"] == git("rev-parse", f"{source_commit}^{{tree}}")
    for path, blob in receipt["source_blobs"].items():
        assert blob == git("rev-parse", f"{source_commit}:{path}"), path
    assert receipt["same_seed_replay_matches"] is True
    assert receipt["cross_seed_paths_distinct"] is True
    for name, metadata in receipt["artifacts"].items():
        path = DEMO_DIR / name
        assert path.stat().st_size == metadata["bytes"], name
        assert sha256(path) == metadata["sha256"], name
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    print("seed integrity demo: verified")


if __name__ == "__main__":
    main()
