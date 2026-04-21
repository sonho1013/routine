from pathlib import Path

from PIL import Image

from scripts.video_gen.contact_sheet import build_contact_sheet
from scripts.video_gen.storyboard import Keyframe, Shot, Storyboard


def _write_png(path: Path, color: tuple[int, int, int]):
    Image.new("RGB", (320, 180), color).save(path)


def _sb(n=3):
    kfs = tuple(Keyframe(id=f"kf{i}", role=f"role{i}", prompt="p")
                for i in range(n))
    shots = tuple(Shot(id=f"s{i}", from_kf=f"kf{i}", to_kf=f"kf{i+1}",
                        duration="5", narrative_role="r", motion_prompt="m")
                   for i in range(n - 1))
    return Storyboard(scene_id="x", scene_summary="s",
                      keyframes=kfs, shots=shots)


def test_contact_sheet_width_is_sum_of_thumbnails(tmp_path):
    sb = _sb(n=3)
    for i in range(3):
        _write_png(tmp_path / f"kf{i}.png", (i * 80, 0, 0))

    out = build_contact_sheet(sb, str(tmp_path))
    img = Image.open(out)
    assert img.size[0] >= 320 * 3
    assert out.endswith("contact_sheet.png")
