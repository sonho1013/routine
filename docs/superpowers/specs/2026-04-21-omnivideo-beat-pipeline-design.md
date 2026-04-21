# OmniVideo Beat Pipeline — Design

**Status:** Active. Replaces the Kling v1.6 I2V shot-sequencing design at
`docs/superpowers/specs/2026-04-20-video-shot-sequencing-design.md`.

**Trigger:** The 2026-04-20 design shipped end-to-end, produced
`output/videos/mary/day1/morning_commute/scene.mp4`, and was rejected by the
user on viewing. Four specific failures:

1. Narrative sequence physically impossible — driving shots preceded engine-on;
   no door-opening/boarding shot; exterior-drive shot was placed first, not last.
2. Too many redundant "driving-looking" shots; named actions (e.g. AC adjust)
   not visibly rendered.
3. Kling v1.6 I2V interpolated between independent keyframes without physics
   awareness — IVI/dashboard elements flew around, gear shifter animated
   incoherently.
4. Viewpoint logic inverted — no clean 3rd-person → 1st-person → 3rd-person
   bookend arc.

The user directed us to adopt Kling OmniVideo (`kling-video-o1`) as the
generation primitive.

## Goal

Produce a multi-beat cinematic scene video where (a) the narrative arc is
physically plausible and causally ordered, (b) every cinematic driving action
is individually visible, and (c) generated motion does not violate the world's
physics.

## Architecture

Four stages, driven by a two-stage CLI (`--stage storyboard|video|all`), mirroring
the existing structure but replacing keyframes/shots with **beats** and the
v1.6 I2V call with **OmniVideo multi-image**.

```
         ┌────────────────┐
         │ reference      │  (project-wide, once, cacheable)
         │ stage          │  → output/videos/refs/*.png  (3 images)
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ storyboard     │  (per scene)
         │ stage          │  GPT-4o → storyboard.json (beats, not shots)
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ video stage    │  (per beat, independent OmniVideo calls)
         │                │  → beats/beat{N}_{type}.mp4
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ concat stage   │  ffmpeg demuxer → scene.mp4
         └────────────────┘
```

**Bookend narrative invariant** (enforced by storyboard validator):

```
exterior_boarding  →  cabin_pov (×N)  →  exterior_driveaway
    (3rd-person)       (1st-person)          (3rd-person)
    Mary enters car    hands on controls     car drives off
    actions=[]         actions=[...]         actions=[]
```

`N = ceil(action_count / 3)`. Each POV beat covers up to 3 cinematic actions
in causally plausible order (engine → gear → driving-adjacent like HVAC/nav/
media). Every cinematic action appears in exactly one POV beat.

Each beat is a single OmniVideo call at fixed 5s duration. Beats are generated
independently (no first/end-frame anchoring between beats, no video-extend
chaining). Concat at the end stitches them losslessly.

## Reference images (shared, cacheable)

Three scene-agnostic T2I keyframes are generated once per project run and
reused across every scene and every beat. Stored at `output/videos/refs/`;
reused on subsequent runs unless `--regen-refs` is passed.

| File | Prompt role | Aspect | Used in beat types |
| --- | --- | --- | --- |
| `mary_ref.png` | Mary character lock (portrait) | 1:1 | all |
| `car_exterior_ref.png` | Renault 3/4 exterior + losange grille | 16:9 | `exterior_boarding`, `exterior_driveaway` |
| `car_interior_ref.png` | POV dashboard, steering wheel losange, controls | 16:9 | `cabin_pov` |

Each prompt is hardcoded in `config.py` with an `assert len(...) < 500` guard
(Kling T2I limit). `mary_ref` prompt is the current `CHARACTER_REF_PROMPT`
renamed.

**Per-beat reference selection:**

```python
REF_SELECTION = {
    "exterior_boarding":   ("mary_ref", "car_exterior_ref"),
    "cabin_pov":           ("mary_ref", "car_interior_ref"),
    "exterior_driveaway":  ("mary_ref", "car_exterior_ref"),
}
```

The two images are passed to OmniVideo in `image_list` order. The LLM-written
motion prompt references them via `<<<image_1>>>` (Mary) and `<<<image_2>>>`
(setting).

## Storyboard schema

```python
@dataclass(frozen=True)
class Beat:
    id: str                        # "beat1", "beat2", ...
    beat_type: Literal[
        "exterior_boarding", "cabin_pov", "exterior_driveaway"
    ]
    actions: tuple[str, ...]       # signal_ids from CINEMATIC_SIGNALS
                                   # must be empty for exterior beats,
                                   # 1..3 items for cabin_pov
    duration: Literal["5"]         # OmniVideo supports 5/7/10; fixed to 5
    motion_prompt: str             # ≤2400 chars, targets OmniVideo

@dataclass(frozen=True)
class Storyboard:
    scene_id: str
    scene_summary: str
    beats: tuple[Beat, ...]
```

**Validator signature and rules:**

```python
def load_storyboard(
    raw: dict,
    expected_actions: list[str] | None = None,
) -> Storyboard
```

`expected_actions` is the canonical list of `signal` ids the LLM was asked to
cover (drawn from the scene's filtered cinematic action list before the call).
When `None` (schema-only check used by unit tests), rules 4 and 6 are skipped.

Rejects:

1. First beat not `exterior_boarding` or last beat not `exterior_driveaway`.
2. Either exterior beat with non-empty `actions`.
3. Any middle beat not of type `cabin_pov`.
4. Middle beat count ≠ `ceil(len(expected_actions) / 3)`.
5. POV beat with 0 actions or > 3 actions.
6. The multiset of actions across POV beats ≠ `expected_actions`
   (each expected action must appear exactly once; no duplicates; no extras).
7. `action` entries not in `CINEMATIC_SIGNALS` whitelist.
8. `motion_prompt` > 2400 chars.
9. `duration != "5"`.

## Storyboard LLM contract

New `STORYBOARD_SYSTEM_PROMPT` in `config.py`. The prompt tells GPT-4o:

- Produce a beat list for a French-animated Parisian commute scene featuring
  Mary driving a Renault electric car.
- **Always open** with one `exterior_boarding` beat: Mary outside the car,
  walking up, opening the door, getting in. 3rd-person camera, `actions=[]`.
- **Always close** with one `exterior_driveaway` beat: the Renault pulling
  away into Paris traffic. 3rd-person, `actions=[]`.
- Between them, write `ceil(N_actions/3)` `cabin_pov` beats. POV means
  over-shoulder / first-person of Mary's hands on controls. Mary herself may
  appear partially (hands, sleeve, scarf) but never in a frontal portrait shot.
- Distribute the provided cinematic actions across POV beats, max 3 per beat,
  in physically plausible causal order: `engine_status` before `gear_position`;
  `gear_position` before any driving-adjacent action like
  `hvac_temp_target` / `nav_destination` / `media_content_id` / `drive_mode` /
  `wiper_state`. Never place an action before its prerequisite.
- In each `motion_prompt`, when describing the character use
  `<<<image_1>>>`; when describing the setting use `<<<image_2>>>`. For
  exterior beats `image_2` is the Renault exterior; for POV beats it is the
  Renault dashboard interior.
- When a POV beat shows the steering wheel, mention the silver Renault
  losange logo on the hub.
- End every `motion_prompt` with the style tail: `French animation style,
  watercolor textures, soft pastel palette, ink linework`.
- Output strict JSON matching the beat schema. No markdown fences.

## OmniVideo client

New method on `KlingClient`:

```python
def omni_video(
    self,
    prompt: str,
    image_list: list[str],           # base64 of each reference image, in order
    duration: str = "5",
    mode: str = "pro",
    aspect_ratio: str = "16:9",
    model_name: str = "kling-video-o1",
) -> str:
    """Submit an OmniVideo multi-image task, poll, return the result video URL."""
```

**Endpoint:** `POST /v1/videos/omni-video` → `GET /v1/videos/omni-video/{task_id}`
(same polling pattern as existing `image_to_video`).

**Request body shape** (per the ComfyUI OmniVideo reference at
`ComfyUI-KLingAI-OmniVideo/kling_nodes.py`):

```json
{
  "model_name": "kling-video-o1",
  "mode": "pro",
  "duration": "5",
  "aspect_ratio": "16:9",
  "prompt": "...<<<image_1>>>...<<<image_2>>>...",
  "image_list": [
    {"image_url": "<base64 of mary_ref>"},
    {"image_url": "<base64 of setting_ref>"}
  ]
}
```

**Response shape:** `data.task_result.videos[0].url` (same pattern as I2V).

**Prompt-length guard:** OmniVideo's prompt limit is not documented in the
ComfyUI node; we defensively cap at 2400 chars (matching I2V limit) in the
client. If Kling returns a length error at run-time we raise `KlingError`
and the smoke test will surface it — at that point we tighten the cap.

The existing v1.6 `image_to_video` method is retained (not deleted) — it is
harmless dead code and keeping it avoids a large diff. If future cleanup
removes it, that is unrelated to this redesign.

## Beat generation module

New module `scripts/video_gen/beats.py` replaces `shots.py`:

```python
def generate_beats(
    kling: KlingClient,
    storyboard: Storyboard,
    out_dir: str,
    ref_paths: dict[str, str],       # from build_references()
    force: bool = False,
) -> list[str]:
    """For each beat, pick 2 refs, call omni_video, download to
    {out_dir}/beats/beat{N}_{type}.mp4. Skip if file exists and !force.
    Returns list of beat mp4 paths in beat order."""
```

Helper:

```python
def _select_refs(beat: Beat, ref_paths: dict[str, str]) -> list[str]:
    names = REF_SELECTION[beat.beat_type]
    return [ref_paths[name] for name in names]
```

The selected ref paths are base64-encoded via the existing
`encode_image_b64` helper in `kling_client.py` and handed to `omni_video`.

## Pipeline CLI

`pipeline.py` changes minimally:

- `run_storyboard_stage`: emits the new beat-shaped `storyboard.json`.
  Contact sheet generation is removed from this stage (no keyframes to show);
  replaced by a 3-cell filmstrip showing the 3 reference images in
  `output/videos/refs/contact_sheet.png` produced by the reference stage.
- `run_video_stage`: calls `build_references` first (with `--regen-refs`
  support), then `generate_beats`, then `concat_videos`.
- `ensure_reference_image` (Mary-only) is removed; replaced by
  `build_references` (3 refs).
- New CLI flag `--regen-refs` (force refs rebuild). Existing `--force` still
  forces beats rebuild.
- `--regen-keyframe` is dropped (no keyframes). If a user mistypes it the
  argparse error suffices.

## File layout

```
output/videos/
├── refs/                                  # project-wide, run once
│   ├── mary_ref.png
│   ├── car_exterior_ref.png
│   ├── car_interior_ref.png
│   └── contact_sheet.png                  # 3-cell filmstrip
└── mary/day1/morning_commute/
    ├── storyboard.json                    # new beat schema
    ├── beats/
    │   ├── beat1_exterior_boarding.mp4
    │   ├── beat2_cabin_pov.mp4
    │   ├── beat3_cabin_pov.mp4            # if N_actions > 3
    │   └── beat4_exterior_driveaway.mp4
    └── scene.mp4
```

A scene with 4 cinematic actions produces 2 POV beats → 4 total beats × 5s =
**20s scene**. 6 actions → 2 POV beats → 20s. 7 actions → 3 POV beats → 25s.

## Testing

- **`tests/test_storyboard_schema.py`** (rewrite): covers new validator
  rules 1–9 above.
- **`tests/test_storyboard_generation.py`** (rewrite): fake OpenAI returns
  beat JSON; success on first try, retry-then-succeed, raise after
  `max_retries` exhausted, markdown-fence stripping.
- **`tests/test_references.py`** (new): `build_references` makes 3 T2I calls
  on empty dir, 0 calls when files exist, 3 calls with `force=True`.
- **`tests/test_beats.py`** (new, replaces `test_shots.py`): for a
  3-beat storyboard, `generate_beats` calls `omni_video` 3 times with the
  correct ref pairs per beat type; skips existing files; `force=True`
  regenerates.
- **`tests/test_kling_omnivideo.py`** (new): `omni_video` sends the correct
  body shape to `/v1/videos/omni-video`, polls `/v1/videos/omni-video/{id}`,
  returns the URL from `data.task_result.videos[0].url`, raises on failed
  task status. Uses `responses` or `unittest.mock` on `self.session`.

All existing passing tests continue to pass (config constants used by other
modules are untouched; the I2V method on `KlingClient` is retained).

## Smoke test (end-to-end)

After implementation, run on the same scene that failed last time:

```bash
python -m scripts.video_gen.pipeline \
    --scene day1_morning_commute --stage all
```

**Success criteria** (subjective; user-reviewed):

1. Scene opens with Mary visibly outside the Renault, approaching and
   entering the driver's seat.
2. Every cinematic action from the scene's action list is visibly rendered
   in a POV beat (user can point at the moment each action occurs).
3. No physics violations: the IVI screen does not fly; the gear shifter
   moves only once per gear change; the steering wheel moves only when the
   car is driving.
4. Scene closes with an exterior 3rd-person shot of the Renault driving away.

If any criterion fails, the failure is documented and the next iteration is
a prompt-tuning pass (LLM contract or ref prompt), not a re-architecture.

## Out of scope

- Aliyun OSS integration (only needed for OmniVideo video-edit / video-extend
  modes, which are not used).
- Audio / narration / music.
- Motion blur or physics-aware post-processing.
- Removing the legacy v1.6 I2V method or `keyframes.py`/`shots.py` modules.
  (Cleanup can happen in a later pass once the new pipeline is stable.)
