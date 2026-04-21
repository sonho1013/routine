import os
from unittest.mock import MagicMock

from scripts.video_gen.keyframes import (
    build_keyframe_prompt,
    generate_keyframes,
)
from scripts.video_gen.storyboard import Keyframe, Shot, Storyboard


def test_build_keyframe_prompt_composes_preamble_body_and_suffix():
    kf = Keyframe(id="kf0", role="establishing_exterior",
                  prompt="parked Renault on a sunny Paris street, 8am light")
    out = build_keyframe_prompt(kf)
    assert out.startswith("Mary, a 30yo European woman")
    assert "parked Renault" in out
    assert out.endswith(
        "French animation style, watercolor textures, "
        "soft pastel palette, ink linework"
    )
    assert len(out) <= 500


def _sb(n_shots=2):
    kfs = tuple(Keyframe(id=f"kf{i}", role="r", prompt=f"p{i}")
                for i in range(n_shots + 1))
    shots = tuple(Shot(id=f"s{i}", from_kf=f"kf{i}", to_kf=f"kf{i+1}",
                        duration="5", narrative_role="r", motion_prompt="m")
                   for i in range(n_shots))
    return Storyboard(scene_id="x", scene_summary="s",
                      keyframes=kfs, shots=shots)


def _make_ref(tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n")
    return str(ref)


def test_generate_keyframes_calls_kling_once_per_keyframe(tmp_path):
    kling = MagicMock()
    kling.text_to_image.return_value = "http://example/img.png"
    kling.download.side_effect = lambda url, path: open(path, "wb").write(b"\x89PNG")
    sb = _sb(n_shots=2)

    paths = generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=str(tmp_path),
        ref_image_path=_make_ref(tmp_path),
    )

    assert kling.text_to_image.call_count == 3
    assert [os.path.basename(p) for p in paths] == ["kf0.png", "kf1.png", "kf2.png"]


def test_generate_keyframes_skips_existing_when_not_forced(tmp_path):
    kling = MagicMock()
    sb = _sb(n_shots=1)
    for i in range(2):
        (tmp_path / f"kf{i}.png").write_bytes(b"existing")

    generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=str(tmp_path),
        ref_image_path=_make_ref(tmp_path),
        force=False,
    )
    kling.text_to_image.assert_not_called()


def test_generate_keyframes_single_target_regenerates_one(tmp_path):
    kling = MagicMock()
    kling.text_to_image.return_value = "http://example/img.png"
    kling.download.side_effect = lambda url, path: open(path, "wb").write(b"\x89PNG")
    sb = _sb(n_shots=2)
    for i in range(3):
        (tmp_path / f"kf{i}.png").write_bytes(b"stale")

    generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=str(tmp_path),
        ref_image_path=_make_ref(tmp_path),
        only_kf_id="kf1",
    )
    assert kling.text_to_image.call_count == 1
