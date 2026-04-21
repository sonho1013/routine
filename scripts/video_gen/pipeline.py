"""Multi-shot video generation pipeline.

Two-stage workflow:
  --stage storyboard   # LLM storyboard + T2I keyframes + contact sheet
  --stage video        # I2V per shot + ffmpeg concat → per-scene mp4
  --stage all          # both stages, no review gate (smoke test only)

Per-scene outputs live under:
  output/videos/mary/dayN/<scene_type>/
    storyboard.json
    keyframes/{kf0..kfN}.png
    keyframes/contact_sheet.png
    shots/{shot1..shotN}.mp4
    scene.mp4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.video_gen.concat import concat_shots
from scripts.video_gen.config import (
    KLING_ACCESS_KEY,
    KLING_API_BASE,
    KLING_IMAGE_MODEL,
    KLING_SECRET_KEY,
    MANIFEST_PATH,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUTPUT_DIR,
)
from scripts.video_gen.contact_sheet import build_contact_sheet
from scripts.video_gen.keyframes import generate_keyframes
from scripts.video_gen.kling_client import KlingClient
from scripts.video_gen.scene_extractor import filter_cinematic_actions
from scripts.video_gen.shots import generate_shots
from scripts.video_gen.storyboard import (
    dump_storyboard,
    generate_storyboard,
    load_storyboard_file,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

REF_IMAGE_PATH = os.path.join(OUTPUT_DIR, "mary", "reference.png")


# ── OpenAI (lazy) ──

def build_openai_client():
    for var in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ── Scene manifest helpers ──

def load_scene(scene_id: str) -> dict:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for entry in manifest:
        if entry["id"] == scene_id:
            return entry
    raise SystemExit(
        f"scene '{scene_id}' not found. Available: {[e['id'] for e in manifest]}"
    )


def scene_output_dir(scene: dict) -> str:
    """Per-scene directory: output/videos/mary/dayN/<scene_type>/"""
    return os.path.join(OUTPUT_DIR, scene["output_path"])


def load_raw_signals_for_scene(scene_id: str) -> list[dict]:
    """Re-derive the raw signal list from the mock dataset.

    Manifest entries only store pretty strings ('Sets AC to 22°C'), but
    filter_cinematic_actions needs signal names. We re-run the generator
    and match by scene id.
    """
    from scenarios.mock_data_generator import generate_full_dataset
    data = generate_full_dataset()
    for scene_type, events in data["scenes"].items():
        if scene_type == "noise":
            continue
        for event in events:
            eid = f"day{event['day']}_{scene_type}"
            if eid == scene_id:
                return event["signals"]
    raise SystemExit(f"signals for scene '{scene_id}' not found in dataset")


# ── Stage 1: storyboard + keyframes ──

def run_storyboard_stage(
    kling: KlingClient,
    openai_client,
    scene: dict,
    regen_keyframe: str | None = None,
    force: bool = False,
) -> None:
    out_dir = scene_output_dir(scene)
    os.makedirs(out_dir, exist_ok=True)
    kf_dir = os.path.join(out_dir, "keyframes")
    sb_path = os.path.join(out_dir, "storyboard.json")

    if regen_keyframe is None and (force or not os.path.exists(sb_path)):
        log.info(f"=== {scene['id']} — generate storyboard ===")
        raw_signals = load_raw_signals_for_scene(scene["id"])
        cinematic = filter_cinematic_actions(raw_signals)
        sb = generate_storyboard(
            openai_client=openai_client,
            scene=scene,
            cinematic_actions=cinematic,
            model=OPENAI_MODEL,
        )
        with open(sb_path, "w", encoding="utf-8") as f:
            json.dump(dump_storyboard(sb), f, indent=2, ensure_ascii=False)
        log.info(f"  storyboard → {sb_path}")
    else:
        log.info(f"=== {scene['id']} — storyboard cached: {sb_path} ===")
        sb = load_storyboard_file(sb_path)

    log.info(f"=== {scene['id']} — generate keyframes ({len(sb.keyframes)}) ===")
    generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=kf_dir,
        ref_image_path=REF_IMAGE_PATH,
        force=force,
        only_kf_id=regen_keyframe,
    )

    sheet_path = build_contact_sheet(sb, kf_dir)
    log.info(f"=== {scene['id']} — contact sheet: {sheet_path} ===")
    log.info("Review the contact sheet. Re-roll any bad keyframe with:")
    log.info(f"  --scene {scene['id']} --stage storyboard --regen-keyframe <kf_id>")


# ── Stage 2: shots + concat ──

def run_video_stage(
    kling: KlingClient,
    scene: dict,
    force: bool = False,
) -> None:
    out_dir = scene_output_dir(scene)
    kf_dir = os.path.join(out_dir, "keyframes")
    shots_dir = os.path.join(out_dir, "shots")
    sb_path = os.path.join(out_dir, "storyboard.json")
    scene_mp4 = os.path.join(out_dir, "scene.mp4")

    if not os.path.exists(sb_path):
        raise SystemExit(
            f"no storyboard at {sb_path} — run --stage storyboard first"
        )

    sb = load_storyboard_file(sb_path)

    log.info(f"=== {scene['id']} — render shots ({len(sb.shots)}) ===")
    shot_paths = generate_shots(
        kling=kling,
        storyboard=sb,
        keyframes_dir=kf_dir,
        shots_dir=shots_dir,
        force=force,
    )

    log.info(f"=== {scene['id']} — concat → {scene_mp4} ===")
    concat_shots(shot_paths=shot_paths, out_path=scene_mp4)
    size_mb = os.path.getsize(scene_mp4) / (1024 * 1024)
    log.info(f"=== {scene['id']} — done: {scene_mp4} ({size_mb:.2f} MB) ===")


# ── Reference image ──

def ensure_reference_image(kling: KlingClient, regen: bool = False) -> str:
    from scripts.video_gen.config import CHARACTER_REF_PROMPT
    if os.path.exists(REF_IMAGE_PATH) and not regen:
        log.info(f"=== reference image cached: {REF_IMAGE_PATH} ===")
        return REF_IMAGE_PATH
    log.info("=== generating Mary reference image ===")
    ref_negative = (
        "photorealistic, 3D render, CGI, anime, cartoon, low quality, "
        "blurry, distorted face, deformed hands, text, watermark, "
        "uncanny valley"
    )
    url = kling.text_to_image(
        prompt=CHARACTER_REF_PROMPT,
        negative_prompt=ref_negative,
        model_name=KLING_IMAGE_MODEL,
        aspect_ratio="1:1",
        n=1,
        image_fidelity=0.5,
        human_fidelity=0.45,
    )
    kling.download(url, REF_IMAGE_PATH)
    return REF_IMAGE_PATH


# ── CLI ──

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-shot Kling video generation pipeline"
    )
    p.add_argument("--scene", type=str,
                   help="Scene ID from the manifest, e.g. day1_morning_commute")
    p.add_argument("--stage", type=str,
                   choices=["storyboard", "video", "all"],
                   help="Pipeline stage to run")
    p.add_argument("--regen-keyframe", type=str, default=None,
                   help="During --stage storyboard, re-roll only this keyframe ID")
    p.add_argument("--force", action="store_true",
                   help="Re-generate outputs even if they already exist")
    p.add_argument("--ref-only", action="store_true",
                   help="Only generate Mary reference portrait and exit")
    p.add_argument("--regen-ref", action="store_true",
                   help="Force regenerate cached Mary reference portrait")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        sys.exit("KLING_ACCESS_KEY / KLING_SECRET_KEY not set")
    kling = KlingClient(KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_API_BASE)

    ensure_reference_image(kling, regen=args.regen_ref)
    if args.ref_only:
        return

    if not args.scene or not args.stage:
        sys.exit("--scene and --stage are required (unless --ref-only)")

    if args.stage in ("storyboard", "all") and not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY not set (required for --stage storyboard/all)")

    scene = load_scene(args.scene)

    if args.stage in ("storyboard", "all"):
        openai_client = build_openai_client()
        run_storyboard_stage(kling, openai_client, scene,
                             regen_keyframe=args.regen_keyframe,
                             force=args.force)

    if args.stage in ("video", "all"):
        run_video_stage(kling, scene, force=args.force)


if __name__ == "__main__":
    main()
