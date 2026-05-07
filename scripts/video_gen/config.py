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

PARKING_GATE_REF_PROMPT = (
    "Underground parking entrance in a Parisian office building. "
    "A red-and-white striped boom barrier across the lane, a card "
    "reader on a short post at driver-window height on the left. "
    "Concrete ceiling, fluorescent lights, no people, no cars. "
    + STYLE_SUFFIX_REF
)
assert len(PARKING_GATE_REF_PROMPT) < 500, (
    f"PARKING_GATE_REF_PROMPT is {len(PARKING_GATE_REF_PROMPT)} chars; "
    f"Kling Image Gen limit is 500"
)

CAR_INTERIOR_REF_PROMPT = (
    "First-person POV from a Renault driver seat. Central touchscreen "
    "IVI glowing softly, showing a stylised nav map with a pastel route "
    "line and rounded icon tiles for climate, media, drive-mode — no "
    "text, no numbers. Steering wheel with the silver Renault losange "
    "on the hub. Gear shifter in P. Empty seat, soft cabin light. "
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
about Mary's daily commute in a Renault electric car in Paris. You write \
OmniVideo motion prompts that render through Kling's multi-image storytelling model.

Given a scene's metadata plus a filtered list of "cinematic" driving actions, \
produce a *beat-based* storyboard with a strict bookend arc. The arc depends \
on the scene DIRECTION (provided in the user prompt):

  departure (e.g. morning commute):
    1. exterior_boarding   (Mary walks up to the car, opens door, gets in)
    2. cabin_pov x N       (start engine, gear D, set nav/media/climate...)
    3. exterior_driveaway  (car pulls away from the curb)

  arrival (e.g. arriving home):
    1. exterior_boarding   (Renault approaches & parks at the curb — 3rd-person)
    2. cabin_pov x N       (gear P, engine off, close windows, open door...)
    3. exterior_driveaway  (Mary steps out of the car and walks away)

  approach (e.g. parking gate entry):
    1. exterior_boarding   (Renault drives up to a boom barrier and stops — 3rd-person; \
       the car MUST be fully stopped at the barrier by the end of this beat)
    2. cabin_pov x N       (window opens, Mary extends her arm out the open window \
       to tap her badge on the card reader post; if wiper_state is listed, handle it \
       here too. The badge-tap gesture and barrier-lift reaction are the visual climax.)
    3. exterior_driveaway  (the boom barrier rises and the car rolls forward through)

IMPORTANT: For "arrival" scenes, the CAUSAL ORDER is REVERSED compared to \
departure — the car is already running, so the actions describe shutting \
things down: gear_position (to P) comes first, then engine_status (off), \
then door_status (open). The exterior_boarding beat shows the car ARRIVING \
(not a person walking up), and the exterior_driveaway beat shows Mary \
EXITING the car (not the car driving off).

Requirements:
- Total beats = N + 2, where N = ceil(action_count / 2).
- Distribute ALL provided actions across the POV beats, at most 2 per beat, \
  with no duplicates and in physically plausible causal order appropriate \
  for the scene direction.
- Each beat.duration MUST be "5".
- POV means over-the-LEFT-shoulder or first-person from the DRIVER seat \
  (left-hand-drive, European). The camera is always behind or beside Mary's \
  LEFT shoulder, looking forward-right toward the centre console and IVI. \
  Mary may be partially visible (left hand on wheel, right hand reaching \
  for controls, sleeve, scarf) but never in a frontal portrait shot and \
  NEVER from the passenger side looking left.
- Never describe Mary's physical appearance — <<<image_1>>> locks it.
- In motion_prompt, reference the character via <<<image_1>>> and the setting \
  via <<<image_2>>>. For exterior beats, <<<image_2>>> is the Renault 3/4 \
  exterior; for POV beats it is the Renault dashboard interior (IVI, gear \
  shifter, steering wheel losange).
- Mention the silver Renault diamond losange logo when a POV beat shows the \
  steering wheel.
- When a POV beat shows the IVI touchscreen or dashboard display, describe \
  the on-screen content as colored watercolor shapes, brushstroke route \
  lines, and simple rounded icon tiles (e.g. a leaf for eco drive-mode, a \
  musical note for media, a snowflake/sun for climate). NEVER write \
  readable text, numbers, percentages, temperature values, speed values, \
  track titles, or brand names onto the screen — watercolor animation \
  cannot render small text legibly and attempting it produces garbled \
  output. Suggest mood/meaning through color and icon only.
- For EACH action listed in beat.actions, the motion_prompt MUST render it \
  as a visible two-part micro-beat: a specific Mary gesture followed by an \
  on-screen IVI response. Use the gesture reference below when writing the \
  gesture half. Both halves must be present for every action — a silent hand \
  movement with no screen feedback, or a screen change with no hand, will \
  not read as the action. Do NOT bundle multiple actions into one shared gesture.
- Gesture reference (use these physical actions for each signal):
    engine_status  → right hand presses the round start/stop button on the dash
    gear_position  → right hand moves the gear shifter lever
    nav_destination → index finger taps the map tile on the IVI touchscreen
    nav_route_pref  → finger swipes or taps a route option on the IVI
    hvac_temp_target → fingers twist the circular climate dial on the console
    media_content_id → finger taps the media/podcast tile on the IVI
    drive_mode     → thumb nudges the drive-mode toggle switch
    wiper_state    → right hand flicks the wiper stalk behind the wheel
    window_position → finger presses the window rocker switch on the door armrest
    door_status    → hand pulls the interior door handle lever
    keyless_entry  → finger presses the small rectangular button on the door \
      handle exterior trim (visible from cabin POV through the window or on \
      the B-pillar touch pad); IVI shows a lock/unlock icon toggling
- Each motion_prompt ≤ 2400 chars; target 2-5 sentences.
- End every motion_prompt with the style tail: \
  "French animation style, watercolor textures, soft pastel palette, ink linework."
- For each cabin_pov beat, add a "keyframe_prompt" field (≤ 340 chars). \
  This will be used to generate a T2I starting keyframe for the beat, so it \
  must describe a STATIC dashboard snapshot showing the END STATE of the \
  actions in that beat: which controls are highlighted, what the IVI screen \
  shows (icon tiles, route lines, colors). Use watercolor icon language, \
  not text or numbers. Do NOT include the style tail — it is appended \
  automatically. Do NOT include character description — it is prepended \
  automatically. Exterior beats must NOT have keyframe_prompt.

Output STRICT JSON (no markdown fences) matching:
{
  "scene_id": "<echo the scene id>",
  "scene_summary": "<one sentence>",
  "beats": [
    {"id": "beat1", "beat_type": "exterior_boarding",
     "actions": [], "duration": "5", "motion_prompt": "..."},
    {"id": "beat2", "beat_type": "cabin_pov",
     "actions": ["signal_a", ...], "duration": "5",
     "motion_prompt": "...", "keyframe_prompt": "..."},
    ...,
    {"id": "beatN", "beat_type": "exterior_driveaway",
     "actions": [], "duration": "5", "motion_prompt": "..."}
  ]
}
"""
