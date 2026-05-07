"""Per-beat OmniVideo rendering. Reads a validated Storyboard + 3 shared
reference images, produces one mp4 per beat, skips existing files on restart.

For cabin_pov beats that carry a keyframe_prompt, a per-beat T2I keyframe
is generated and used as <<<image_2>>> instead of the generic car_interior_ref.
"""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    KLING_IMAGE_MODEL,
    KLING_OMNI_MODEL,
    KLING_VIDEO_ASPECT,
    KLING_VIDEO_MODE,
    MARY_CHARACTER_PREAMBLE,
    STYLE_SUFFIX_REF,
)
from scripts.video_gen.kling_client import encode_image_b64
from scripts.video_gen.storyboard import Beat, Storyboard

log = logging.getLogger(__name__)

REF_SELECTION: dict[str, tuple[str, str]] = {
    "exterior_boarding":  ("mary_ref", "car_exterior_ref"),
    "cabin_pov":          ("mary_ref", "car_interior_ref"),
    "exterior_driveaway": ("mary_ref", "car_exterior_ref"),
}

REF_SELECTION_APPROACH: dict[str, tuple[str, str]] = {
    "exterior_boarding":  ("mary_ref", "parking_gate_ref"),
    "cabin_pov":          ("mary_ref", "car_interior_ref"),
    "exterior_driveaway": ("mary_ref", "parking_gate_ref"),
}

_KEYFRAME_NEGATIVE = (
    "photorealistic, 3D render, CGI, anime, cartoon, low quality, blurry, "
    "text, watermark, numbers, readable words"
)


def _select_refs(beat: Beat, ref_paths: dict[str, str],
                 direction: str = "departure") -> list[str]:
    """Return the 2 base64 strings for the beat's type, in <<<image_N>>> order."""
    table = REF_SELECTION_APPROACH if direction == "approach" else REF_SELECTION
    names = table[beat.beat_type]
    for name in names:
        if name not in ref_paths:
            raise KeyError(
                f"reference {name!r} required for beat_type "
                f"{beat.beat_type!r} not found in ref_paths"
            )
    return [encode_image_b64(ref_paths[name]) for name in names]


def _build_keyframe(
    kling,
    beat: Beat,
    out_dir: str,
    force: bool = False,
) -> str | None:
    """Generate a per-beat T2I keyframe for cabin_pov beats. Returns path or None."""
    if beat.beat_type != "cabin_pov" or not beat.keyframe_prompt:
        return None
    kf_path = os.path.join(out_dir, f"{beat.id}_keyframe.png")
    if os.path.exists(kf_path) and not force:
        log.info(f"  {beat.id} keyframe — SKIP (cached)")
        return kf_path
    prompt = f"{MARY_CHARACTER_PREAMBLE} {beat.keyframe_prompt} {STYLE_SUFFIX_REF}"
    log.info(f"  {beat.id} keyframe — T2I ({len(prompt)} chars)")
    url = kling.text_to_image(
        prompt=prompt,
        negative_prompt=_KEYFRAME_NEGATIVE,
        model_name=KLING_IMAGE_MODEL,
        aspect_ratio="16:9",
        n=1,
    )
    kling.download(url, kf_path)
    return kf_path


def generate_beats(
    kling,
    storyboard: Storyboard,
    out_dir: str,
    ref_paths: dict[str, str],
    force: bool = False,
) -> list[str]:
    """Render one OmniVideo mp4 per beat. Returns paths in beat order.

    For cabin_pov beats with keyframe_prompt, generates a T2I keyframe first
    and uses it as <<<image_2>>> instead of the generic car_interior_ref.
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

        kf_path = _build_keyframe(kling, beat, beats_dir, force=force)

        image_list = _select_refs(beat, ref_paths)
        if kf_path:
            image_list[1] = encode_image_b64(kf_path)

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
