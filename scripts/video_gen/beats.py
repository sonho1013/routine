"""Per-beat OmniVideo rendering. Reads a validated Storyboard + 3 shared
reference images, produces one mp4 per beat, skips existing files on restart.
"""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    KLING_OMNI_MODEL,
    KLING_VIDEO_ASPECT,
    KLING_VIDEO_MODE,
)
from scripts.video_gen.kling_client import encode_image_b64
from scripts.video_gen.storyboard import Beat, Storyboard

log = logging.getLogger(__name__)

REF_SELECTION: dict[str, tuple[str, str]] = {
    "exterior_boarding":  ("mary_ref", "car_exterior_ref"),
    "cabin_pov":          ("mary_ref", "car_interior_ref"),
    "exterior_driveaway": ("mary_ref", "car_exterior_ref"),
}


def _select_refs(beat: Beat, ref_paths: dict[str, str]) -> list[str]:
    """Return the 2 base64 strings for the beat's type, in <<<image_N>>> order."""
    names = REF_SELECTION[beat.beat_type]
    for name in names:
        if name not in ref_paths:
            raise KeyError(
                f"reference {name!r} required for beat_type "
                f"{beat.beat_type!r} not found in ref_paths"
            )
    return [encode_image_b64(ref_paths[name]) for name in names]


def generate_beats(
    kling,
    storyboard: Storyboard,
    out_dir: str,
    ref_paths: dict[str, str],
    force: bool = False,
) -> list[str]:
    """Render one OmniVideo mp4 per beat. Returns paths in beat order.

    Skips a beat whose mp4 already exists unless force=True.
    """
    beats_dir = os.path.join(out_dir, "beats")
    os.makedirs(beats_dir, exist_ok=True)

    out_paths: list[str] = []
    for beat in storyboard.beats:
        filename = f"{beat.id}_{beat.beat_type}.mp4"
        out_path = os.path.join(beats_dir, filename)
        out_paths.append(out_path)

        if os.path.exists(out_path) and not force:
            log.info(f"  {beat.id} ({beat.beat_type}) — SKIP (exists)")
            continue

        image_list = _select_refs(beat, ref_paths)
        log.info(f"  {beat.id} ({beat.beat_type}) — OmniVideo")
        url = kling.omni_video(
            prompt=beat.motion_prompt,
            image_list=image_list,
            duration=beat.duration,
            mode=KLING_VIDEO_MODE,
            aspect_ratio=KLING_VIDEO_ASPECT,
            model_name=KLING_OMNI_MODEL,
        )
        kling.download(url, out_path)

    return out_paths
