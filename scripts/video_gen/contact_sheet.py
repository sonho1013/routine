"""Assemble per-scene keyframe thumbnails into one reviewable filmstrip PNG."""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

from scripts.video_gen.storyboard import Storyboard

THUMB_W = 480
THUMB_H = 270
LABEL_H = 40
GAP = 8


def build_contact_sheet(storyboard: Storyboard, keyframes_dir: str) -> str:
    """Build <keyframes_dir>/contact_sheet.png and return its path."""
    n = len(storyboard.keyframes)
    sheet_w = n * THUMB_W + (n + 1) * GAP
    sheet_h = THUMB_H + LABEL_H + GAP * 2
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, kf in enumerate(storyboard.keyframes):
        kf_path = os.path.join(keyframes_dir, f"{kf.id}.png")
        thumb = Image.open(kf_path).convert("RGB")
        thumb = thumb.resize((THUMB_W, THUMB_H), Image.LANCZOS)
        x = GAP + i * (THUMB_W + GAP)
        sheet.paste(thumb, (x, GAP))
        label = f"{kf.id}  {kf.role}"
        draw.text((x + 6, THUMB_H + GAP + 6), label, fill=(230, 230, 230), font=font)

    out = os.path.join(keyframes_dir, "contact_sheet.png")
    sheet.save(out)
    return out
