"""ffmpeg concat wrapper. Stitches per-shot mp4s into a single per-scene mp4."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)


def concat_shots(shot_paths: list[str], out_path: str) -> None:
    """Concat shot_paths (in order) into out_path using ffmpeg concat demuxer.

    Assumes all shots share the same codec/resolution/fps (Kling's output is
    consistent per account). If that assumption breaks in production, the
    implementation should switch to re-encoding via `-c:v libx264 -c:a aac`.
    """
    for p in shot_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"shot mp4 missing: {p}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for p in shot_paths:
            escaped = os.path.abspath(p).replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")
        list_file = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", out_path,
        ]
        log.info(f"  concat {len(shot_paths)} shots → {out_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed: {result.stderr[-500:]}"
            )
    finally:
        try:
            os.unlink(list_file)
        except OSError:
            pass
