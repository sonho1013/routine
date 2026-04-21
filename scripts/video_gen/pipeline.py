"""Multi-beat video generation pipeline (OmniVideo).

Two-stage workflow:
  --stage storyboard   # LLM produces beat-based storyboard.json
  --stage video        # build 3 shared refs (once), then OmniVideo per beat,
                       # then ffmpeg concat → per-scene scene.mp4
  --stage all          # both stages, no review gate (smoke test only)

Per-scene outputs live under:
  output/videos/mary/dayN/<scene_type>/
    storyboard.json
    beats/{beatN_<type>.mp4}
    scene.mp4

Project-wide shared references live under:
  output/videos/refs/
    mary_ref.png
    car_exterior_ref.png
    car_interior_ref.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.video_gen.beats import generate_beats
from scripts.video_gen.concat import concat_shots
from scripts.video_gen.config import (
    KLING_ACCESS_KEY,
    KLING_API_BASE,
    KLING_SECRET_KEY,
    MANIFEST_PATH,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUTPUT_DIR,
)
from scripts.video_gen.kling_client import KlingClient
from scripts.video_gen.references import build_references
from scripts.video_gen.scene_extractor import filter_cinematic_actions
from scripts.video_gen.storyboard import (
    dump_storyboard,
    generate_storyboard,
    load_storyboard_file,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

REFS_DIR = os.path.join(OUTPUT_DIR, "refs")


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
        f"scene '{scene_id}' not found. "
        f"Available: {[e['id'] for e in manifest]}"
    )


def scene_output_dir(scene: dict) -> str:
    return os.path.join(OUTPUT_DIR, scene["output_path"])


def load_raw_signals_for_scene(scene_id: str) -> list[dict]:
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


# ── Stage 1: storyboard ──

def run_storyboard_stage(
    openai_client,
    scene: dict,
    force: bool = False,
) -> None:
    out_dir = scene_output_dir(scene)
    os.makedirs(out_dir, exist_ok=True)
    sb_path = os.path.join(out_dir, "storyboard.json")

    if not force and os.path.exists(sb_path):
        log.info(f"=== {scene['id']} — storyboard cached: {sb_path} ===")
        return

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
    log.info(f"  storyboard → {sb_path} ({len(sb.beats)} beats)")


# ── Stage 2: references + beats + concat ──

def run_video_stage(
    kling: KlingClient,
    scene: dict,
    force: bool = False,
    regen_refs: bool = False,
) -> None:
    out_dir = scene_output_dir(scene)
    sb_path = os.path.join(out_dir, "storyboard.json")
    scene_mp4 = os.path.join(out_dir, "scene.mp4")

    if not os.path.exists(sb_path):
        raise SystemExit(
            f"no storyboard at {sb_path} — run --stage storyboard first"
        )

    log.info(f"=== {scene['id']} — ensure shared references ===")
    ref_paths = build_references(kling, out_dir=REFS_DIR, force=regen_refs)

    sb = load_storyboard_file(sb_path)

    log.info(f"=== {scene['id']} — render beats ({len(sb.beats)}) ===")
    beat_paths = generate_beats(
        kling=kling, storyboard=sb, out_dir=out_dir,
        ref_paths=ref_paths, force=force,
    )

    log.info(f"=== {scene['id']} — concat → {scene_mp4} ===")
    concat_shots(shot_paths=beat_paths, out_path=scene_mp4)
    size_mb = os.path.getsize(scene_mp4) / (1024 * 1024)
    log.info(f"=== {scene['id']} — done: {scene_mp4} ({size_mb:.2f} MB) ===")


# ── CLI ──

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-beat Kling OmniVideo pipeline"
    )
    p.add_argument("--scene", type=str, required=False,
                   help="Scene ID from the manifest (e.g. day1_morning_commute)")
    p.add_argument("--stage", type=str,
                   choices=["storyboard", "video", "all"],
                   required=False,
                   help="Pipeline stage to run")
    p.add_argument("--force", action="store_true",
                   help="Re-generate storyboard/beats even if cached")
    p.add_argument("--regen-refs", action="store_true",
                   help="Force regenerate the 3 shared reference PNGs")
    p.add_argument("--refs-only", action="store_true",
                   help="Only build the 3 shared reference PNGs, then exit")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        sys.exit("KLING_ACCESS_KEY / KLING_SECRET_KEY not set")
    kling = KlingClient(KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_API_BASE)

    if args.refs_only:
        build_references(kling, out_dir=REFS_DIR, force=args.regen_refs)
        return

    if not args.scene or not args.stage:
        sys.exit("--scene and --stage are required (unless --refs-only)")

    if args.stage in ("storyboard", "all") and not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY not set (required for --stage storyboard/all)")

    scene = load_scene(args.scene)

    if args.stage in ("storyboard", "all"):
        openai_client = build_openai_client()
        run_storyboard_stage(openai_client, scene, force=args.force)

    if args.stage in ("video", "all"):
        run_video_stage(kling, scene, force=args.force,
                        regen_refs=args.regen_refs)


if __name__ == "__main__":
    main()
