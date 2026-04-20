# Design: Multi-Shot Scene Sequencing for Video Generation

**Date**: 2026-04-20
**Owner**: Ed Chi
**Status**: Spec — awaiting implementation plan

## Problem

The current video workflow (`scripts/video_gen/pipeline.py`) generates **one 5-second Kling I2V clip per scene**, all seeded from the same static portrait `output/videos/mary/reference.png`. Each clip tries to compress an entire scene (e.g. morning commute with 13 logged actions: AC 22°C, navigate to work, fastest route, media "Tech Daily", …) into a single shot.

Limitations:

1. **No action visibility.** Individual driving behaviours (turning the HVAC dial, tapping "Fastest route", selecting a nav destination) are never shown as discrete actions — they are at best hinted at within a single wide shot.
2. **No narrative arc.** A scene has no "establishing exterior → driver POV → specific action" progression. It is one take from one angle.
3. **Monotony across days.** Day 1 through Day 5 of the same scene look nearly identical because the reference frame and the compressed prompt vary only slightly.
4. **No continuity mechanics.** There is no concept of linking clips; even if we rendered multiple clips per scene today, each would start from the static reference with no visual bridge between them.

## Goals

- Per scene, render **multiple linked short clips** that together form a coherent narrative: establishing exterior → transition into the cabin → a sequence of action-focused shots (nav, AC, route choice, media, …).
- Between clips, **first/last frames match exactly**, so the final stitched video feels like one continuous take with camera moves rather than a slideshow of separate clips.
- **Visual style stays consistent** across all clips of a scene and across the 15 scenes (same French watercolour aesthetic, same Mary face, same cabin).
- **Cost is not a primary constraint** — quality and narrative clarity come first.

## Non-Goals

- Real-time or interactive generation. Batch/offline only.
- Voice-over, music, sound design. Visual-only output for now.
- Re-doing the scene selection / manifest generation logic. We reuse `scenes_manifest.json` as-is (15 entries, morning_commute / arriving_home / toll_parking_entry × 5 days).
- Replacing Kling as the video backend. Kling I2V + T2I remain the only model dependency.

## Design Decisions

The following four decisions were locked in during brainstorming; this spec is built on top of them.

### Decision 1 — Shot decomposition strategy: **per-action, adaptive**
Instead of a fixed N-shot template per scene type, each visible action in the scene's action list gets its own dedicated 5-second shot. An opening establishing shot and a POV-transition shot precede the action shots.

### Decision 2 — Action selection: **whitelist filter + LLM sequencing**
A hard-coded whitelist in config defines which signal types are "cinematic" (HVAC dial turn, nav destination select, route preference, media play, gear shift, engine start, wiper activation, window position, …) vs. non-visual (seat heating level, ACC distance, fan speed when on auto, volume percentage). The LLM receives only the filtered, cinematic-capable actions for the scene and is responsible for:
1. Ordering them into a narrative sequence (opening state → en-route actions → settling).
2. Writing the per-shot motion prompts and the keyframe descriptions.

### Decision 3 — Frame continuity: **pre-generated keyframes + Kling `image_tail`**
Before any video is rendered, every transition point in the scene is generated as a static T2I image. A scene with N shots has N+1 keyframes (`kf0 … kfN`). Each I2V call then receives both `image` (the shot's start keyframe) and `image_tail` (the shot's end keyframe) — Kling interpolates motion between them. This is the only approach that can realistically execute the "camera moves from street exterior through the window into Mary's POV" kind of cross-shot camera move.

Kling `image_tail` support must be confirmed at implementation time. If the currently configured model (`kling-v3`) does not support it, the implementation switches to the nearest model that does (likely `kling-v1-6`) for the I2V calls, while continuing to use `kling-v3` for T2I keyframes.

### Decision 4 — Workflow shape: **two-stage pipeline with storyboard review gate**
Rendering is split into two CLI stages:

- `pipeline.py --scene <id> --stage storyboard` — LLM produces the storyboard JSON, T2I generates all keyframes, a contact sheet is assembled for human review. Cheap (T2I only).
- `pipeline.py --scene <id> --stage video` — reads the (approved) storyboard + keyframes, runs N × I2V calls, stitches the resulting clips into a single per-scene mp4. Expensive (I2V).

A single `--regen-keyframe <shot_id>_<first|tail>` command lets a human re-roll just the offending keyframe between the two stages. No keyframes are regenerated automatically during `--stage video`.

## Architecture

```
scripts/video_gen/
  config.py          ← + CINEMATIC_SIGNALS whitelist, storyboard schema constants
  kling_client.py    ← + image_tail parameter support on image_to_video()
  scene_extractor.py ← unchanged
  pipeline.py        ← refactored: --stage {storyboard, video, all}
  storyboard.py      ← NEW: LLM storyboard generation, JSON schema validation
  keyframes.py       ← NEW: T2I keyframe generation with Mary subject-reference locking
  shots.py           ← NEW: per-shot I2V using (first_kf, tail_kf) pairs
  concat.py          ← NEW: ffmpeg concat of per-shot mp4s into per-scene mp4
  contact_sheet.py   ← NEW: assemble all keyframes of a scene into one review image

output/videos/mary/day1/morning_commute/
  storyboard.json              ← storyboard for this scene
  keyframes/kf0.png … kfN.png  ← per-transition keyframes
  keyframes/contact_sheet.png  ← reviewable grid of all keyframes
  shots/shot1.mp4 … shotN.mp4  ← per-shot I2V outputs
  scene.mp4                    ← final stitched output
```

The **per-scene mp4** (`scene.mp4`) replaces today's `morning_commute.mp4`, `arriving_home.mp4`, `toll_parking_entry.mp4` as the canonical deliverable for the scene. The old flat mp4s are kept untouched for comparison but are no longer produced by the pipeline.

## Data Contracts

### Storyboard JSON Schema

```json
{
  "scene_id": "day1_morning_commute",
  "scene_summary": "Sunny 8:15 AM departure from a Paris residential street; Mary sets navigation, chooses fastest route, adjusts AC to 22°C, starts Tech Daily podcast.",
  "keyframes": [
    {
      "id": "kf0",
      "role": "establishing_exterior",
      "prompt": "<≤480 chars T2I prompt — subject, setting, lighting, composition; ends with STYLE_SUFFIX_REF>"
    },
    { "id": "kf1", "role": "pov_transition", "prompt": "…" },
    { "id": "kf2", "role": "action_nav_destination", "prompt": "…" },
    { "id": "kf3", "role": "action_route_fastest",   "prompt": "…" },
    { "id": "kf4", "role": "action_hvac_22c",        "prompt": "…" },
    { "id": "kf5", "role": "action_media_play",      "prompt": "…" }
  ],
  "shots": [
    {
      "id": "shot1",
      "from_kf": "kf0",
      "to_kf": "kf1",
      "duration": "5",
      "narrative_role": "exterior_to_pov_dolly",
      "motion_prompt": "<≤2400 chars I2V prompt describing the motion between the two keyframes>"
    },
    { "id": "shot2", "from_kf": "kf1", "to_kf": "kf2", "duration": "5", "narrative_role": "pov_reach_nav_screen", "motion_prompt": "…" },
    { "id": "shot3", "from_kf": "kf2", "to_kf": "kf3", "duration": "5", "narrative_role": "select_fastest_route", "motion_prompt": "…" },
    { "id": "shot4", "from_kf": "kf3", "to_kf": "kf4", "duration": "5", "narrative_role": "hand_to_hvac_dial",    "motion_prompt": "…" },
    { "id": "shot5", "from_kf": "kf4", "to_kf": "kf5", "duration": "5", "narrative_role": "pulling_away_podcast", "motion_prompt": "…" }
  ]
}
```

Invariants enforced at load time (fail-fast, not best-effort):

- Every `shots[i].from_kf` and `shots[i].to_kf` must resolve to an entry in `keyframes[]`.
- For consecutive shots, `shots[i].to_kf == shots[i+1].from_kf` (no visual gap between adjacent shots).
- `len(keyframes) == len(shots) + 1`.
- Each `keyframes[i].prompt` is ≤ 280 chars. Rationale: final assembled T2I prompt = 120-char character preamble + keyframe.prompt + ~70-char `STYLE_SUFFIX_REF` ≤ 500 (Kling T2I hard limit).
- Each `shots[i].motion_prompt` is ≤ 2400 chars (safety margin under Kling I2V's 2500-char limit — motion_prompt is used as-is, no wrapping).

### Cinematic signal whitelist (lives in `config.py`)

```python
CINEMATIC_SIGNALS = {
    # signal_name: {"role_prefix": str, "priority": int}
    "engine_status":     {"role_prefix": "action_engine",     "priority": 10},
    "gear_position":     {"role_prefix": "action_gear",       "priority": 20},
    "nav_destination":   {"role_prefix": "action_nav",        "priority": 30},
    "nav_route_pref":    {"role_prefix": "action_route",      "priority": 31},
    "hvac_temp_target":  {"role_prefix": "action_hvac",       "priority": 40},
    "media_content_id":  {"role_prefix": "action_media",      "priority": 50},
    "drive_mode":        {"role_prefix": "action_drive_mode", "priority": 60},
    "wiper_state":       {"role_prefix": "action_wiper",      "priority": 70},
    "window_position":   {"role_prefix": "action_window",     "priority": 71},
    "door_status":       {"role_prefix": "action_door",       "priority": 80},
    "keyless_entry":     {"role_prefix": "action_keyless",    "priority": 81},
}

# Signals explicitly filtered out as non-visual:
#   hvac_fan_speed, hvac_power, seat_heating, media_source,
#   media_volume, acc_distance, vehicle_speed
```

The LLM receives only signals whose name is a key of `CINEMATIC_SIGNALS`. It is free to reorder them subject to narrative logic (e.g. engine_status=on usually belongs before nav_destination), but priority is a hint, not a hard order.

## Pipeline Flow

### Stage 1 — Storyboard + Keyframes (`--stage storyboard`)

1. Load `scenes_manifest.json`, pick the target scene by `--scene <id>`.
2. Filter `scene["actions"]` through `CINEMATIC_SIGNALS` to get the cinematic action list.
3. Call OpenAI (single call, GPT-4o structured output) with:
   - System prompt: storyboard writer role, output schema, style rules.
   - User prompt: scene metadata + filtered actions + palette hint.
   - Expected output: the full JSON object defined under "Storyboard JSON Schema" above.
4. Validate storyboard JSON against invariants. Fail fast on any violation.
5. Persist `storyboard.json` to the scene output directory.
6. For each keyframe:
   - Build final T2I prompt: `"Mary, a 30-year-old European woman with light brown hair in a casual updo, wearing a light blue jacket over a cream top and thin wool scarf. "` + keyframe's `prompt` + `STYLE_SUFFIX_REF`.
   - Call `kling.text_to_image` with `image_reference=reference.png` (subject reference), `human_fidelity=0.7`, `image_fidelity=0.5`.
   - Download to `keyframes/kfN.png`.
7. Assemble `keyframes/contact_sheet.png`: a horizontal filmstrip of all keyframes labelled with their `role`, so a human can scan all transitions at once.
8. Print the path to the contact sheet and exit.

`--regen-keyframe <kf_id>` re-runs step 6 for a single keyframe only.

### Stage 2 — Shots + Stitch (`--stage video`)

1. Read `storyboard.json` and verify all referenced keyframe PNGs exist.
2. For each shot:
   - Load `from_kf` and `to_kf` images, base64-encode both.
   - Call `kling.image_to_video(image=from_b64, image_tail=to_b64, prompt=motion_prompt, …)`.
   - Download to `shots/shotN.mp4`.
   - Resume-safe: if `shots/shotN.mp4` already exists, skip unless `--force`.
3. `ffmpeg -f concat -safe 0 -i list.txt -c copy scene.mp4` to stitch all shot mp4s into the final scene output.
4. Print duration + size of `scene.mp4`.

### `--stage all`
Runs storyboard then video back-to-back with no review gate. Intended for regressions/smoke tests, not for the curated demo.

## Mary Face Consistency Strategy

The single largest risk in this design is character drift across 6+ T2I keyframes per scene. Three reinforcing mitigations:

1. **Subject reference on every T2I call.** `reference.png` is passed as `image_reference` with `subject_reference` type on every keyframe generation, so Kling locks the face to the reference portrait. `human_fidelity` is raised to 0.7 (vs. 0.45 used for the reference itself) to bias toward likeness over prompt creativity.
2. **Fixed character preamble.** Every keyframe T2I prompt is prefixed with the same 120-character subject description of Mary (hair, eyes, jacket, scarf) before any shot-specific content. This protects against Kling silently ignoring the reference when it has to generate a non-portrait composition (e.g. a POV-from-behind shot).
3. **Deterministic style suffix.** `STYLE_SUFFIX_REF` is appended to every keyframe prompt unchanged. This is the same suffix used for the reference portrait itself, so the watercolour / ink-line treatment stays identical across keyframes.

Remaining residual risk: Kling does not expose a seed parameter, so two runs of the same prompt give different images. This is absorbed by the review gate — a drifting face in `kf3` just means `--regen-keyframe kf3` until it matches.

## Risks and Open Questions

| Risk | Mitigation |
|------|------------|
| `kling-v3` does not support `image_tail` | Confirm support during first implementation step; fall back to `kling-v1-6` for I2V only if needed. Block on this before touching shot code. |
| LLM produces storyboard with invariant violations (kf mismatch, too many chars) | Strict JSON schema validation at load time. Retry once with the validation error fed back into the prompt; fail hard on second violation. |
| Mary face drift beyond what subject-reference can hold | Review gate catches it; `--regen-keyframe` is cheap (T2I, not I2V). |
| ffmpeg concat fails on codec mismatches between Kling mp4s | Kling returns consistent codec per account; if a mismatch appears, transcode all shots to a normalised format (libx264 yuv420p 24fps) before concat. Implement only if observed. |
| Cost explosion across 15 scenes × ~6 shots | User accepted cost trade-off. Storyboard stage is cheap; expensive video stage can be run per-scene in a curated order rather than all at once. |

## Testing

Each new module (`storyboard.py`, `keyframes.py`, `shots.py`, `concat.py`) gets a dedicated `tests/test_video_gen_*.py` file. Tests do not call live Kling or OpenAI APIs — external calls are mocked at the `KlingClient` / `openai.OpenAI.chat.completions` boundary.

Minimum coverage:

- `test_storyboard_schema_validation.py` — valid storyboards pass, invariants are enforced (kf mismatch, prompt too long, dangling kf refs each produce a specific error).
- `test_cinematic_filter.py` — given a scene's raw actions, only whitelisted signals pass to the LLM.
- `test_keyframes_prompt_assembly.py` — character preamble + keyframe.prompt + style suffix assembled in the correct order and within char limits.
- `test_shots_image_tail_wiring.py` — `shots.py` passes the correct keyframe pair to `kling_client.image_to_video` for each shot.
- `test_concat_invariants.py` — concat fails fast if any shot mp4 is missing.

No end-to-end live test. A manual smoke test against `day1_morning_commute` serves as the acceptance gate before the other 14 scenes are re-rendered.

## Out of Scope for This Design

- Audio track (music, narration, ambience) — separate future spec.
- Cross-scene continuity (e.g. Day 1 evening leading into Day 2 morning) — each scene is still a self-contained video here.
- Migration of already-rendered flat `.mp4`s — they remain on disk untouched; the new per-scene output lives at `mary/dayN/sceneType/scene.mp4` alongside them.
