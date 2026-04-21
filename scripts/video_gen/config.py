"""
Video generation pipeline configuration.

API keys read from environment variables.
Kling credentials can also be set here as fallback for direct API usage.
"""
import os

# ── API Keys ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Kling — direct API (no ComfyUI gateway)
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY", "132030a84e784f2ab3dce922711ce8af")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY", "f844d49f7564474083f1f34dc6450ecc")
# Region base URL: 国内 = api.klingai.com, 国际 = api-singapore.klingai.com
KLING_API_BASE = os.getenv("KLING_API_BASE", "https://api.klingai.com")

# ── ComfyUI (legacy — no longer used by pipeline.py, kept for reference) ──
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_DIR = os.getenv("COMFYUI_DIR", os.path.expanduser("~/comfy/ComfyUI"))

# ── Paths ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, "comfyui_workflows")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "videos")
MANIFEST_PATH = os.path.join(WORKFLOW_DIR, "scenes_manifest.json")

# ── Style Constants ──
STYLE_SUFFIX = (
    "French animation style, hand-painted watercolor textures, "
    "soft pastel color palette with warm undertones, "
    "delicate ink linework, subtle film grain, "
    "cinematic composition, European illustration aesthetic, "
    "inspired by Sylvain Chomet and Gobelins animation, "
    "gentle ambient lighting, atmospheric depth"
)

NEGATIVE_PROMPT = (
    "photorealistic, 3D render, CGI, anime style, cartoon style, "
    "low quality, blurry, distorted face, extra limbs, deformed hands, "
    "text, watermark, signature, frame border, uncanny valley"
)

# Compact style suffix for Kling Image Gen (which has a 500-char prompt limit).
# The full STYLE_SUFFIX above is reserved for video prompts (Kling I2V allows longer).
STYLE_SUFFIX_REF = (
    "French animation style, watercolor textures, "
    "soft pastel palette, ink linework"
)

CHARACTER_REF_PROMPT = (
    "Character reference of Mary, a 30-year-old European woman with "
    "light brown hair in a casual updo, warm brown eyes, natural look. "
    "Wearing a smart-casual light blue jacket over a cream top with "
    "a thin wool scarf. Sitting in a modern Renault electric car, "
    "with the silver Renault diamond losange logo clearly visible on "
    "the steering wheel hub. Three-quarter portrait, upper body "
    "visible. Soft natural cabin light. "
    + STYLE_SUFFIX_REF
)
assert len(CHARACTER_REF_PROMPT) < 500, (
    f"CHARACTER_REF_PROMPT is {len(CHARACTER_REF_PROMPT)} chars; "
    f"Kling Image Gen limit is 500"
)

# Alias: MARY_REF_PROMPT is the preferred name going forward; the old
# CHARACTER_REF_PROMPT name is kept so obsolete modules still import cleanly.
MARY_REF_PROMPT = CHARACTER_REF_PROMPT

CAR_EXTERIOR_REF_PROMPT = (
    "Parisian street in soft morning light, a Renault electric car parked "
    "by the curb. Three-quarter exterior view, silver Renault diamond "
    "losange logo clearly visible on the front grille. Cream Haussmannian "
    "facade in the background, empty sidewalk, no people. "
    + STYLE_SUFFIX_REF
)
assert len(CAR_EXTERIOR_REF_PROMPT) < 500, (
    f"CAR_EXTERIOR_REF_PROMPT is {len(CAR_EXTERIOR_REF_PROMPT)} chars; "
    f"Kling Image Gen limit is 500"
)

CAR_INTERIOR_REF_PROMPT = (
    "First-person POV from a Renault electric car's driver seat. Clean "
    "modern dashboard with central touchscreen IVI, climate control "
    "panel, gear shifter in P. Steering wheel in frame with the silver "
    "Renault diamond losange logo on the hub. Empty seat, no person, "
    "soft natural cabin light. "
    + STYLE_SUFFIX_REF
)
assert len(CAR_INTERIOR_REF_PROMPT) < 500, (
    f"CAR_INTERIOR_REF_PROMPT is {len(CAR_INTERIOR_REF_PROMPT)} chars; "
    f"Kling Image Gen limit is 500"
)

# ── LLM Prompt Generation ──
LLM_SYSTEM_PROMPT = """\
You are a cinematic scene writer for French-style animated short films \
about daily life in Paris. Given structured driving behavior data, \
write a vivid scene description suitable for AI video generation.

Rules:
- Write 2-3 sentences of visual storytelling
- Include: weather atmosphere, time-of-day lighting, character actions, mood
- Naturally reference the specific driving actions from the data
- Match the color palette to the time of day
- The character is Mary, a ~30-year-old European woman
- DO NOT describe Mary's appearance (reference image handles this)
- DO describe environment, lighting, weather, actions, dashboard, vehicle
- The vehicle is a Renault electric car — when describing the dashboard \
or steering wheel, naturally mention the silver Renault diamond losange \
logo on the steering wheel hub
- Always end with style keywords: "French animation style, hand-painted \
watercolor textures, soft pastel palette, delicate ink linework"

Output strictly as JSON (no markdown fences):
{
  "prompt": "cinematic scene description for video generation",
  "negative_prompt": "elements to avoid",
  "camera_type": "static | pan_left | pan_right | zoom_in | zoom_out | tracking"
}\
"""

# ── Kling Model Settings ──
KLING_IMAGE_MODEL = "kling-v3"  # KlingImageGenModelName
KLING_VIDEO_MODEL = "kling-v1-6"  # I2V: must support image_tail
KLING_OMNI_MODEL = "kling-video-o1"   # OmniVideo multi-image storytelling
KLING_VIDEO_MODE = "pro"
KLING_VIDEO_DURATION = "5"  # seconds
KLING_VIDEO_ASPECT = "16:9"
KLING_VIDEO_CFG = 0.5

# OpenAI model for prompt generation
OPENAI_MODEL = "gpt-4o"

# ── Cinematic Signal Whitelist ──
# Only signals listed here get their own storyboard shot. Priority is an
# LLM ordering hint, not a hard constraint. See
# docs/superpowers/specs/2026-04-20-video-shot-sequencing-design.md.
CINEMATIC_SIGNALS: dict[str, dict] = {
    "engine_status":    {"role_prefix": "action_engine",     "priority": 10},
    "gear_position":    {"role_prefix": "action_gear",       "priority": 20},
    "nav_destination":  {"role_prefix": "action_nav",        "priority": 30},
    "nav_route_pref":   {"role_prefix": "action_route",      "priority": 31},
    "hvac_temp_target": {"role_prefix": "action_hvac",       "priority": 40},
    "media_content_id": {"role_prefix": "action_media",      "priority": 50},
    "drive_mode":       {"role_prefix": "action_drive_mode", "priority": 60},
    "wiper_state":      {"role_prefix": "action_wiper",      "priority": 70},
    "window_position":  {"role_prefix": "action_window",     "priority": 71},
    "door_status":      {"role_prefix": "action_door",       "priority": 80},
    "keyless_entry":    {"role_prefix": "action_keyless",    "priority": 81},
}

# Character preamble reused on every T2I keyframe prompt to keep Mary consistent.
# Kept ≤120 chars so the final T2I prompt (preamble + keyframe.prompt + STYLE_SUFFIX_REF)
# fits under Kling's 500-char limit.
MARY_CHARACTER_PREAMBLE = (
    "Mary, a 30yo European woman, light brown hair casual updo, "
    "light blue jacket, cream top, thin wool scarf."
)
assert len(MARY_CHARACTER_PREAMBLE) <= 120, (
    f"MARY_CHARACTER_PREAMBLE is {len(MARY_CHARACTER_PREAMBLE)} chars; must be ≤120"
)

# ── Storyboard LLM Prompt ──
STORYBOARD_SYSTEM_PROMPT = """\
You are a cinematic storyboard writer for French-style animated short films \
about daily life in Paris, featuring Mary driving a Renault electric car.

Given a scene's metadata (time, weather, location, palette) plus a filtered \
list of "cinematic" driving actions, produce a storyboard that renders as:
- 2 opening shots: establishing exterior, then transition from exterior into \
  Mary's POV inside the cabin.
- N action shots, one per provided cinematic action, in a narrative order you \
  choose (state changes first — engine on / gear shift — then en-route actions \
  like navigation and HVAC, closing with media or environmental beats).

Total keyframes = total shots + 1. Consecutive shots share a keyframe: \
shot[i].to_kf must equal shot[i+1].from_kf.

Each shot is exactly 5 seconds.

Rules:
- DO NOT describe Mary's physical appearance — a subject-reference image \
  handles that. Do describe her actions, her posture, what her hands are doing.
- Match lighting/palette to the time of day given in the scene metadata.
- Mention the Renault silver diamond losange logo when the steering wheel \
  or dashboard is visible.
- Each keyframe.prompt ≤ 300 chars (hard limit; shorter is better).
- Each shot.motion_prompt ≤ 2400 chars but target 2-4 sentences.
- End every keyframe.prompt with: French animation style, watercolour textures, \
  soft pastel palette, ink linework.

Output strictly JSON matching this schema (no markdown fences):
{
  "scene_id": "<echo the scene id>",
  "scene_summary": "<one sentence>",
  "keyframes": [
    {"id": "kfN", "role": "<slug>", "prompt": "<T2I description>"},
    ...
  ],
  "shots": [
    {"id": "shotN", "from_kf": "kfN", "to_kf": "kfN+1", "duration": "5",
     "narrative_role": "<slug>", "motion_prompt": "<I2V description>"},
    ...
  ]
}
"""
