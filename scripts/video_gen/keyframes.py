"""T2I keyframe generation with Mary-face consistency.

Each keyframe's final T2I prompt is:
    MARY_CHARACTER_PREAMBLE + " " + keyframe.prompt + " " + STYLE_SUFFIX_REF

Every T2I call passes the cached Mary reference portrait as image_reference
(subject type), with human_fidelity raised to 0.7 to bias toward likeness.
"""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    KLING_IMAGE_MODEL,
    MARY_CHARACTER_PREAMBLE,
    STYLE_SUFFIX_REF,
)
from scripts.video_gen.kling_client import encode_image_b64
from scripts.video_gen.storyboard import Keyframe, Storyboard

log = logging.getLogger(__name__)

T2I_NEGATIVE = (
    "photorealistic, 3D render, CGI, anime, cartoon, low quality, blurry, "
    "distorted face, deformed hands, text, watermark, uncanny valley"
)
assert len(T2I_NEGATIVE) < 200


def build_keyframe_prompt(kf: Keyframe) -> str:
    """Compose the final T2I prompt from character preamble + body + style."""
    prompt = f"{MARY_CHARACTER_PREAMBLE} {kf.prompt} {STYLE_SUFFIX_REF}"
    if len(prompt) > 500:
        raise ValueError(
            f"keyframe {kf.id} final prompt is {len(prompt)} chars; "
            f"Kling limit is 500"
        )
    return prompt


def generate_keyframes(
    kling,
    storyboard: Storyboard,
    out_dir: str,
    ref_image_path: str,
    force: bool = False,
    only_kf_id: str | None = None,
) -> list[str]:
    """Generate one PNG per keyframe. Returns the list of resulting paths in order.

    - Skips a keyframe whose PNG already exists unless force=True.
    - If only_kf_id is set, only that one keyframe is regenerated (force implied).
    """
    os.makedirs(out_dir, exist_ok=True)
    ref_b64 = encode_image_b64(ref_image_path)

    paths: list[str] = []
    for kf in storyboard.keyframes:
        path = os.path.join(out_dir, f"{kf.id}.png")
        paths.append(path)

        if only_kf_id is not None and kf.id != only_kf_id:
            continue
        if os.path.exists(path) and not force and only_kf_id is None:
            log.info(f"  keyframe {kf.id} — SKIP (exists)")
            continue

        prompt = build_keyframe_prompt(kf)
        log.info(f"  keyframe {kf.id} ({kf.role}) — T2I")
        url = kling.text_to_image(
            prompt=prompt,
            negative_prompt=T2I_NEGATIVE,
            model_name=KLING_IMAGE_MODEL,
            aspect_ratio="16:9",
            n=1,
            image_fidelity=0.5,
            human_fidelity=0.7,
            image_reference_b64=ref_b64,
            reference_type="subject",
        )
        kling.download(url, path)

    return paths
