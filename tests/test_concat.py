import os
import subprocess
from unittest.mock import patch

import pytest

from scripts.video_gen.concat import concat_shots


def test_concat_builds_correct_list_file_and_invokes_ffmpeg(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    for name in ("shot1.mp4", "shot2.mp4"):
        (shots_dir / name).write_bytes(b"x")
    out = tmp_path / "scene.mp4"

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        list_file_idx = cmd.index("-i") + 1
        list_path = cmd[list_file_idx]
        recorded["list"] = open(list_path).read()
        open(cmd[-1], "wb").write(b"stitched")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("subprocess.run", side_effect=fake_run):
        concat_shots(
            shot_paths=[str(shots_dir / "shot1.mp4"), str(shots_dir / "shot2.mp4")],
            out_path=str(out),
        )

    assert os.path.exists(out)
    assert "shot1.mp4" in recorded["list"]
    assert "shot2.mp4" in recorded["list"]
    assert recorded["cmd"][0] == "ffmpeg"


def test_concat_raises_when_any_shot_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        concat_shots(shot_paths=[str(tmp_path / "missing.mp4")],
                     out_path=str(tmp_path / "out.mp4"))
