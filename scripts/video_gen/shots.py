"""Per-shot I2V rendering. Reads approved storyboard + keyframes, produces
one mp4 per shot, skips existing files on restart."""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    KLING_VIDEO_ASPECT,
    KLING_VIDEO_CFG,
    KLING_VIDEO_MODEL,
    KLING_VIDEO_MODE,
    NEGATIVE_PROMPT,
)
from scripts.video_gen.kling_client import encode_image_b64
from scripts.video_gen.storyboard import Storyboard

log = logging.getLogger(__name__)


def generate_shots(
    kling,
    storyboard: Storyboard,
    keyframes_dir: str,
    shots_dir: str,
    force: bool = False,
) -> list[str]:
    """Run I2V for each shot. Returns the list of mp4 paths in scene order.

    Fails fast with FileNotFoundError if any required keyframe PNG is missing.
    Skips a shot whose mp4 already exists unless force=True.
    """
    os.makedirs(shots_dir, exist_ok=True)

    kf_paths: dict[str, str] = {}
    for kf in storyboard.keyframes:
        p = os.path.join(keyframes_dir, f"{kf.id}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"keyframe image missing: {p} — run --stage storyboard first"
            )
        kf_paths[kf.id] = p

    out_paths: list[str] = []
    for shot in storyboard.shots:
        out_path = os.path.join(shots_dir, f"{shot.id}.mp4")
        out_paths.append(out_path)

        if os.path.exists(out_path) and not force:
            log.info(f"  {shot.id} — SKIP (exists)")
            continue

        from_b64 = encode_image_b64(kf_paths[shot.from_kf])
        tail_b64 = encode_image_b64(kf_paths[shot.to_kf])

        log.info(f"  {shot.id} ({shot.narrative_role}) — I2V "
                 f"{shot.from_kf}→{shot.to_kf}")
        url = kling.image_to_video(
            prompt=shot.motion_prompt,
            image_b64=from_b64,
            image_tail_b64=tail_b64,
            negative_prompt=NEGATIVE_PROMPT,
            model_name=KLING_VIDEO_MODEL,
            cfg_scale=KLING_VIDEO_CFG,
            mode=KLING_VIDEO_MODE,
            aspect_ratio=KLING_VIDEO_ASPECT,
            duration=shot.duration,
        )
        kling.download(url, out_path)

    return out_paths
