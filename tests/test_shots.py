import os
from unittest.mock import MagicMock

import pytest
import PIL.Image

from scripts.video_gen.shots import generate_shots
from scripts.video_gen.storyboard import Keyframe, Shot, Storyboard


def _sb(n=2):
    kfs = tuple(Keyframe(id=f"kf{i}", role="r", prompt="p")
                for i in range(n + 1))
    shots = tuple(Shot(id=f"shot{i+1}", from_kf=f"kf{i}", to_kf=f"kf{i+1}",
                        duration="5", narrative_role="r", motion_prompt=f"m{i}")
                   for i in range(n))
    return Storyboard(scene_id="x", scene_summary="s",
                      keyframes=kfs, shots=shots)


def test_generate_shots_passes_both_images_to_i2v(tmp_path):
    sb = _sb(n=2)
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    for i in range(3):
        PIL.Image.new("RGB", (16, 9), (i, 0, 0)).save(kf_dir / f"kf{i}.png")
    shots_dir = tmp_path / "shots"

    kling = MagicMock()
    kling.image_to_video.return_value = "http://v/shot.mp4"
    kling.download.side_effect = lambda url, path: open(path, "wb").write(b"vid")

    generate_shots(kling=kling, storyboard=sb,
                   keyframes_dir=str(kf_dir), shots_dir=str(shots_dir))

    assert kling.image_to_video.call_count == 2
    first_call_kwargs = kling.image_to_video.call_args_list[0].kwargs
    assert first_call_kwargs["image_b64"]
    assert first_call_kwargs["image_tail_b64"]
    assert first_call_kwargs["prompt"] == "m0"


def test_generate_shots_skips_existing_mp4(tmp_path):
    sb = _sb(n=1)
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    for i in range(2):
        PIL.Image.new("RGB", (16, 9), (i, 0, 0)).save(kf_dir / f"kf{i}.png")
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    (shots_dir / "shot1.mp4").write_bytes(b"done")

    kling = MagicMock()
    generate_shots(kling=kling, storyboard=sb,
                   keyframes_dir=str(kf_dir), shots_dir=str(shots_dir))
    kling.image_to_video.assert_not_called()


def test_generate_shots_fails_fast_when_keyframe_missing(tmp_path):
    sb = _sb(n=1)
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    PIL.Image.new("RGB", (16, 9), (0, 0, 0)).save(kf_dir / "kf0.png")

    with pytest.raises(FileNotFoundError, match="kf1.png"):
        generate_shots(kling=MagicMock(), storyboard=sb,
                       keyframes_dir=str(kf_dir),
                       shots_dir=str(tmp_path / "shots"))
