"""Build the 3 shared T2I reference images used across all beats/scenes.

  mary_ref.png          — character lock (Mary portrait)
  car_exterior_ref.png  — Renault 3/4 exterior, losange grille, Parisian street
  car_interior_ref.png  — Renault cabin POV, dashboard + losange on wheel

Generated once per project run; subsequent runs reuse the cached files
unless force=True is passed. Used as `image_list` inputs to OmniVideo —
each beat selects 2 of these 3 refs (see REF_SELECTION in beats.py).
"""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    CAR_EXTERIOR_REF_PROMPT,
    CAR_INTERIOR_REF_PROMPT,
    KLING_IMAGE_MODEL,
    MARY_REF_PROMPT,
)

log = logging.getLogger(__name__)

_REF_NEGATIVE = (
    "photorealistic, 3D render, CGI, anime, cartoon, low quality, blurry, "
    "distorted face, deformed hands, text, watermark, uncanny valley"
)
assert len(_REF_NEGATIVE) < 200

_REFS = (
    ("mary_ref",          MARY_REF_PROMPT,          "1:1"),
    ("car_exterior_ref",  CAR_EXTERIOR_REF_PROMPT,  "16:9"),
    ("car_interior_ref",  CAR_INTERIOR_REF_PROMPT,  "16:9"),
)


def build_references(
    kling,
    out_dir: str,
    force: bool = False,
) -> dict[str, str]:
    """Generate the 3 shared reference PNGs. Returns {name: absolute_path}.

    Skips any ref whose file already exists, unless force=True.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for name, prompt, aspect in _REFS:
        path = os.path.join(out_dir, f"{name}.png")
        paths[name] = path
        if os.path.exists(path) and not force:
            log.info(f"  reference {name} — SKIP (cached at {path})")
            continue
        log.info(f"  reference {name} — T2I ({aspect})")
        url = kling.text_to_image(
            prompt=prompt,
            negative_prompt=_REF_NEGATIVE,
            model_name=KLING_IMAGE_MODEL,
            aspect_ratio=aspect,
            n=1,
            image_fidelity=0.5,
            human_fidelity=0.45,
        )
        kling.download(url, path)
    return paths
