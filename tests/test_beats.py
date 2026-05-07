"""Unit tests for beats.generate_beats."""
import os
from unittest.mock import MagicMock

from scripts.video_gen.beats import REF_SELECTION, generate_beats
from scripts.video_gen.storyboard import Beat, Storyboard


def _ref_paths(tmp_path) -> dict:
    paths = {}
    for name in ("mary_ref", "car_exterior_ref", "car_interior_ref"):
        p = tmp_path / f"{name}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        paths[name] = str(p)
    return paths


def _three_beat_storyboard() -> Storyboard:
    return Storyboard(
        scene_id="day1_morning_commute",
        scene_summary="test",
        beats=(
            Beat(id="beat1", beat_type="exterior_boarding", actions=(),
                 duration="5",
                 motion_prompt="Mary <<<image_1>>> near <<<image_2>>>."),
            Beat(id="beat2", beat_type="cabin_pov",
                 actions=("engine_status", "gear_position"),
                 duration="5",
                 motion_prompt="POV hands on wheel <<<image_2>>>.",
                 keyframe_prompt="Dashboard with engine on, gear in D."),
            Beat(id="beat3", beat_type="exterior_driveaway", actions=(),
                 duration="5",
                 motion_prompt="Car drives off <<<image_2>>>."),
        ),
    )


def _fake_kling():
    k = MagicMock()
    k.omni_video.return_value = "https://cdn/v.mp4"
    k.text_to_image.return_value = "https://cdn/kf.png"
    k.download.side_effect = lambda url, dest: open(dest, "wb").write(b"v")
    return k


def test_ref_selection_covers_all_three_beat_types():
    assert set(REF_SELECTION.keys()) == {
        "exterior_boarding", "cabin_pov", "exterior_driveaway",
    }
    assert REF_SELECTION["cabin_pov"] == ("mary_ref", "car_interior_ref")
    assert REF_SELECTION["exterior_boarding"] == ("mary_ref", "car_exterior_ref")
    assert REF_SELECTION["exterior_driveaway"] == ("mary_ref", "car_exterior_ref")


def test_generate_beats_calls_omni_once_per_beat(tmp_path):
    kling = _fake_kling()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    refs = _ref_paths(tmp_path)

    paths = generate_beats(
        kling=kling, storyboard=sb,
        out_dir=str(out_dir), ref_paths=refs,
    )

    assert kling.omni_video.call_count == 3
    assert paths[0].endswith("beat1_exterior_boarding.mp4")
    assert paths[1].endswith("beat2_cabin_pov.mp4")
    assert paths[2].endswith("beat3_exterior_driveaway.mp4")
    for p in paths:
        assert os.path.getsize(p) > 0


def test_cabin_pov_generates_keyframe_t2i(tmp_path):
    kling = _fake_kling()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    refs = _ref_paths(tmp_path)

    generate_beats(kling=kling, storyboard=sb,
                   out_dir=str(out_dir), ref_paths=refs)

    assert kling.text_to_image.call_count == 1
    t2i_call = kling.text_to_image.call_args
    assert "Dashboard" in t2i_call.kwargs["prompt"]
    kf_path = str(out_dir / "beats" / "beat2_keyframe.png")
    assert os.path.exists(kf_path)


def test_generate_beats_skips_existing_files(tmp_path):
    kling = _fake_kling()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    beats_dir = out_dir / "beats"
    beats_dir.mkdir(parents=True)
    for name in ("beat1_exterior_boarding.mp4",
                 "beat2_cabin_pov.mp4",
                 "beat3_exterior_driveaway.mp4"):
        (beats_dir / name).write_bytes(b"cached")

    refs = _ref_paths(tmp_path)
    generate_beats(kling=kling, storyboard=sb,
                   out_dir=str(out_dir), ref_paths=refs)
    assert kling.omni_video.call_count == 0
    assert kling.text_to_image.call_count == 0


def test_generate_beats_force_regenerates(tmp_path):
    kling = _fake_kling()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    beats_dir = out_dir / "beats"
    beats_dir.mkdir(parents=True)
    for name in ("beat1_exterior_boarding.mp4",
                 "beat2_cabin_pov.mp4",
                 "beat3_exterior_driveaway.mp4"):
        (beats_dir / name).write_bytes(b"cached")

    refs = _ref_paths(tmp_path)
    generate_beats(kling=kling, storyboard=sb,
                   out_dir=str(out_dir), ref_paths=refs, force=True)
    assert kling.omni_video.call_count == 3
    assert kling.text_to_image.call_count == 1


def test_generate_beats_raises_if_ref_missing(tmp_path):
    kling = _fake_kling()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    refs = {"mary_ref": str(tmp_path / "does_not_exist.png")}
    try:
        generate_beats(kling=kling, storyboard=sb,
                       out_dir=str(out_dir), ref_paths=refs)
    except KeyError as e:
        assert "car_exterior_ref" in str(e)
    else:
        raise AssertionError("expected KeyError for missing ref")
