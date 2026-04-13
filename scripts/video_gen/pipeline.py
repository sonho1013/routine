"""
Direct Kling Video Generation Pipeline — bypasses ComfyUI entirely.

For each scene in scenes_manifest.json:
  1. Ensure Mary reference image exists (cached, regenerate with --regen-ref)
  2. Call OpenAI to expand structured scene data → cinematic video prompt
  3. Submit to Kling I2V (image-to-video) using the cached reference
  4. Poll until done, download the .mp4

Usage:
    python -m scripts.video_gen.pipeline                              # all 15 scenes
    python -m scripts.video_gen.pipeline --ref-only                   # ref image only
    python -m scripts.video_gen.pipeline --scene day1_morning_commute # one scene
    python -m scripts.video_gen.pipeline --regen-ref                  # force re-gen ref
"""
import argparse
import json
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.video_gen.config import (
    KLING_ACCESS_KEY,
    KLING_SECRET_KEY,
    KLING_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUTPUT_DIR,
    MANIFEST_PATH,
    LLM_SYSTEM_PROMPT,
    CHARACTER_REF_PROMPT,
    NEGATIVE_PROMPT,
    KLING_IMAGE_MODEL,
    KLING_VIDEO_MODEL,
    KLING_VIDEO_MODE,
    KLING_VIDEO_DURATION,
    KLING_VIDEO_ASPECT,
    KLING_VIDEO_CFG,
)
from scripts.video_gen.kling_client import KlingClient, KlingError, encode_image_b64

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

REF_IMAGE_PATH = os.path.join(OUTPUT_DIR, "mary", "reference.png")

# Compact negative prompt for T2I (Kling limit: 200 chars)
# Note: "logo" intentionally NOT in this list — we WANT the Renault losange logo.
NEGATIVE_PROMPT_REF = (
    "photorealistic, 3D render, CGI, anime, cartoon, low quality, "
    "blurry, distorted face, deformed hands, text, watermark, "
    "uncanny valley"
)
assert len(NEGATIVE_PROMPT_REF) < 200, (
    f"NEGATIVE_PROMPT_REF is {len(NEGATIVE_PROMPT_REF)} chars; limit is 200"
)


# ═══════════════════════════════════════════════════
# OpenAI client (lazy — only built when scenes are run, not for --ref-only)
# ═══════════════════════════════════════════════════

def build_openai_client():
    """Construct an OpenAI client, working around an env where ALL_PROXY=socks://...
    is set (httpx ships without SOCKS support unless `httpx[socks]` is installed).
    HTTP_PROXY/HTTPS_PROXY remain set so traffic still goes through the local proxy."""
    # Strip socks proxy vars from env *for this process only* before httpx reads them
    for var in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# ═══════════════════════════════════════════════════
# Scene prompt generation (OpenAI)
# ═══════════════════════════════════════════════════

def generate_scene_prompt(openai_client, user_prompt: str) -> str:
    """Call OpenAI to expand structured scene data into a cinematic video prompt."""
    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if model added them despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
        raw = raw.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"OpenAI returned non-JSON: {raw[:200]}") from e
    return data["prompt"]


# ═══════════════════════════════════════════════════
# Reference image (cached)
# ═══════════════════════════════════════════════════

def ensure_reference_image(kling: KlingClient, regen: bool = False) -> str:
    """Generate Mary reference image if not cached. Returns local path."""
    if os.path.exists(REF_IMAGE_PATH) and not regen:
        log.info(f"=== Reference image cached: {REF_IMAGE_PATH} ===")
        return REF_IMAGE_PATH

    log.info("=== Generating Mary reference image ===")
    log.info(f"  prompt length: {len(CHARACTER_REF_PROMPT)} chars")
    url = kling.text_to_image(
        prompt=CHARACTER_REF_PROMPT,
        negative_prompt=NEGATIVE_PROMPT_REF,
        model_name=KLING_IMAGE_MODEL,
        aspect_ratio="1:1",
        n=1,
        image_fidelity=0.5,
        human_fidelity=0.45,
    )
    kling.download(url, REF_IMAGE_PATH)
    return REF_IMAGE_PATH


# ═══════════════════════════════════════════════════
# Single scene execution
# ═══════════════════════════════════════════════════

def run_scene(
    kling: KlingClient,
    openai_client,
    ref_b64: str,
    scene: dict,
    force: bool = False,
) -> bool:
    """Generate one scene's video. Returns True if generated, False if skipped.

    Skips (returns False) when the output .mp4 already exists, unless force=True.
    Skipping happens BEFORE the OpenAI call, so resume-after-interrupt costs $0.
    """
    out_path = os.path.join(OUTPUT_DIR, scene["output_path"] + ".mp4")
    if os.path.exists(out_path) and not force:
        log.info(f"=== Scene: {scene['id']} — SKIP (already exists: {out_path}) ===")
        return False

    log.info(f"=== Scene: {scene['id']} ===")

    # 1. Cinematic prompt via OpenAI
    log.info("  → OpenAI scene prompt generation")
    scene_prompt = generate_scene_prompt(openai_client, scene["llm_user_prompt"])
    log.info(f"  prompt: {scene_prompt[:100]}...")

    # Kling I2V prompt limit is 2500; the LLM is constrained to 2-3 sentences
    # so this is mostly a safety net.
    if len(scene_prompt) > 2500:
        log.warning(f"  prompt too long ({len(scene_prompt)}), truncating to 2500")
        scene_prompt = scene_prompt[:2500]

    # 2. Submit I2V
    video_url = kling.image_to_video(
        prompt=scene_prompt,
        image_b64=ref_b64,
        negative_prompt=NEGATIVE_PROMPT,
        model_name=KLING_VIDEO_MODEL,
        cfg_scale=KLING_VIDEO_CFG,
        mode=KLING_VIDEO_MODE,
        aspect_ratio=KLING_VIDEO_ASPECT,
        duration=KLING_VIDEO_DURATION,
    )

    # 3. Download
    kling.download(video_url, out_path)
    return True


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Direct Kling Video Generation Pipeline")
    parser.add_argument("--ref-only", action="store_true",
                        help="Generate Mary reference image only and exit")
    parser.add_argument("--scene", type=str, default=None,
                        help="Run a single scene by ID (e.g., day1_morning_commute)")
    parser.add_argument("--regen-ref", action="store_true",
                        help="Force regenerate the cached Mary reference image")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate scenes even if their .mp4 already exists "
                             "(default: skip existing — safe to resume after interrupt)")
    args = parser.parse_args()

    # Validate credentials early — fail-fast
    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        log.error("KLING_ACCESS_KEY / KLING_SECRET_KEY not set in env or config.py")
        sys.exit(1)
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set")
        sys.exit(1)

    log.info(f"Kling API base: {KLING_API_BASE}")
    log.info(f"Output dir:     {OUTPUT_DIR}")

    kling = KlingClient(KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_API_BASE)

    # Step A: ensure reference image (always needed for I2V)
    ref_path = ensure_reference_image(kling, regen=args.regen_ref)

    if args.ref_only:
        log.info("--ref-only set, exiting after reference image.")
        return

    # OpenAI client only needed for scene runs (lazy init avoids socks-proxy issue
    # blocking --ref-only path)
    openai_client = build_openai_client()

    # Encode reference image once, reuse across all scenes
    log.info("Encoding reference image to base64...")
    ref_b64 = encode_image_b64(ref_path)
    log.info(f"  reference image base64 size: {len(ref_b64) // 1024} KB")

    # Step B: scenes
    manifest = load_manifest()
    log.info(f"Loaded {len(manifest)} scenes from manifest")

    if args.scene:
        scene = next((s for s in manifest if s["id"] == args.scene), None)
        if not scene:
            log.error(f"Scene '{args.scene}' not found.")
            log.error(f"Available: {[s['id'] for s in manifest]}")
            sys.exit(1)
        try:
            run_scene(kling, openai_client, ref_b64, scene, force=args.force)
        except (KlingError, RuntimeError) as e:
            log.error(f"FAILED: {scene['id']} — {e}")
            sys.exit(1)
        return

    # Batch run (sequential)
    ok, skipped, fail = 0, 0, 0
    failed_ids = []
    for i, scene in enumerate(manifest):
        log.info(f"\n[{i + 1}/{len(manifest)}]")
        try:
            generated = run_scene(kling, openai_client, ref_b64, scene, force=args.force)
            if generated:
                ok += 1
            else:
                skipped += 1
        except (KlingError, RuntimeError) as e:
            log.error(f"  FAILED: {scene['id']} — {e}")
            failed_ids.append(scene["id"])
            fail += 1
            continue

    log.info(f"\n=== Done: {ok} generated, {skipped} skipped (already existed), {fail} failed ===")
    if failed_ids:
        log.info(f"Failed scenes: {failed_ids}")
        log.info("Re-run individual failures with: --scene <id>")


if __name__ == "__main__":
    main()
