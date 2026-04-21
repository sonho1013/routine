"""Unit tests for references.build_references — the 3-image shared ref builder."""
import os
from unittest.mock import MagicMock

from scripts.video_gen.references import build_references


def _fake_kling():
    k = MagicMock()
    k.text_to_image.return_value = "https://cdn/img.png"
    k.download.side_effect = lambda url, dest: open(dest, "wb").write(b"x")
    return k


def test_build_references_calls_t2i_three_times_on_empty_dir(tmp_path):
    kling = _fake_kling()
    paths = build_references(kling, out_dir=str(tmp_path), force=False)
    assert set(paths.keys()) == {
        "mary_ref", "car_exterior_ref", "car_interior_ref",
    }
    assert kling.text_to_image.call_count == 3
    for p in paths.values():
        assert p.startswith(str(tmp_path))
        assert p.endswith(".png")
        assert os.path.getsize(p) > 0


def test_build_references_is_noop_when_files_exist(tmp_path):
    for name in ("mary_ref", "car_exterior_ref", "car_interior_ref"):
        (tmp_path / f"{name}.png").write_bytes(b"stale")
    kling = _fake_kling()
    paths = build_references(kling, out_dir=str(tmp_path), force=False)
    assert kling.text_to_image.call_count == 0
    assert len(paths) == 3
    assert (tmp_path / "mary_ref.png").read_bytes() == b"stale"


def test_build_references_force_regenerates_all(tmp_path):
    for name in ("mary_ref", "car_exterior_ref", "car_interior_ref"):
        (tmp_path / f"{name}.png").write_bytes(b"stale")
    kling = _fake_kling()
    build_references(kling, out_dir=str(tmp_path), force=True)
    assert kling.text_to_image.call_count == 3


def test_each_ref_uses_correct_aspect_ratio(tmp_path):
    kling = _fake_kling()
    build_references(kling, out_dir=str(tmp_path), force=False)
    ratios = [c.kwargs["aspect_ratio"]
              for c in kling.text_to_image.call_args_list]
    assert ratios.count("1:1") == 1
    assert ratios.count("16:9") == 2
