# Video Shot Sequencing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the video generation pipeline to produce a per-action multi-shot sequence per scene, with pre-generated T2I keyframes and Kling `image_tail` frame-interpolation so adjacent shots are continuous.

**Architecture:** Two-stage CLI (`--stage storyboard` / `--stage video`). Stage 1 calls OpenAI to produce a storyboard JSON (N+1 keyframes + N shots), T2I-generates each keyframe, and assembles a contact sheet for human review. Stage 2 reads the approved storyboard, runs N × Kling I2V calls each seeded with `image` (from-keyframe) + `image_tail` (to-keyframe), then ffmpeg-concats into a single per-scene mp4.

**Tech Stack:** Python 3.11, Kling REST API (T2I + I2V with `image_tail`), OpenAI GPT-4o (structured output), Pillow for contact sheet, ffmpeg CLI for frame extraction + concat, pytest.

**Spec:** `docs/superpowers/specs/2026-04-20-video-shot-sequencing-design.md`

---

## Task 1: Verify Kling `image_tail` support and extend `KlingClient.image_to_video`

The spec's core continuity mechanism depends on Kling accepting an `image_tail` parameter on I2V. This is a blocker — if the currently configured model doesn't support it, we must know now (not after writing three modules around it).

**Files:**
- Modify: `scripts/video_gen/kling_client.py:141-183` (`image_to_video` signature + body)
- Create: `tests/test_kling_client_image_tail.py`

- [ ] **Step 1: Write failing test for image_tail pass-through**

```python
# tests/test_kling_client_image_tail.py
from unittest.mock import MagicMock, patch

from scripts.video_gen.kling_client import KlingClient


def test_image_to_video_sends_image_tail_when_provided():
    client = KlingClient("ak", "sk", "https://example.test")
    fake_post_resp = {"data": {"task_id": "t1"}}
    fake_poll_resp = {"data": {"task_status": "succeed",
                                "task_result": {"videos": [{"url": "http://v"}]}}}

    with patch.object(client, "_post", return_value=fake_post_resp) as mpost, \
         patch.object(client, "_get", return_value=fake_poll_resp):
        client.image_to_video(
            prompt="move camera",
            image_b64="AAAA",
            image_tail_b64="BBBB",
            model_name="kling-v1-6",
            mode="pro",
            duration="5",
        )

    _, kwargs = mpost.call_args
    body = kwargs["body"] if "body" in kwargs else mpost.call_args.args[1]
    assert body["image"] == "AAAA"
    assert body["image_tail"] == "BBBB"


def test_image_to_video_omits_image_tail_when_none():
    client = KlingClient("ak", "sk", "https://example.test")
    fake_post_resp = {"data": {"task_id": "t1"}}
    fake_poll_resp = {"data": {"task_status": "succeed",
                                "task_result": {"videos": [{"url": "http://v"}]}}}

    with patch.object(client, "_post", return_value=fake_post_resp) as mpost, \
         patch.object(client, "_get", return_value=fake_poll_resp):
        client.image_to_video(prompt="p", image_b64="AAAA", model_name="kling-v3")

    body = mpost.call_args.args[1]
    assert "image_tail" not in body
```

- [ ] **Step 2: Run test, verify it fails**

Run: `pytest tests/test_kling_client_image_tail.py -v`
Expected: FAIL — `image_to_video()` has no `image_tail_b64` parameter.

- [ ] **Step 3: Extend `image_to_video` signature**

Replace lines 141-172 of `scripts/video_gen/kling_client.py` with:

```python
    def image_to_video(
        self,
        prompt: str,
        image_b64: str,
        image_tail_b64: str | None = None,
        negative_prompt: str = "",
        model_name: str = "kling-v2-1-master",
        cfg_scale: float = 0.5,
        mode: str = "pro",
        aspect_ratio: str = "16:9",
        duration: str = "5",
    ) -> str:
        """Submit an I2V task, poll until done, return the first video URL.

        If `image_tail_b64` is provided, Kling interpolates motion between
        the two keyframes. The caller is responsible for choosing a model
        that supports `image_tail` (e.g. kling-v1-6); this method does not
        validate model capability.
        """
        if len(prompt) > 2500:
            raise KlingError(
                f"I2V prompt is {len(prompt)} chars; Kling limit is 2500"
            )
        if negative_prompt and len(negative_prompt) > 2500:
            raise KlingError(
                f"I2V negative_prompt is {len(negative_prompt)} chars; "
                f"Kling limit is 2500"
            )
        body = {
            "model_name": model_name,
            "image": image_b64,
            "prompt": prompt,
            "cfg_scale": cfg_scale,
            "mode": mode,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }
        if image_tail_b64:
            body["image_tail"] = image_tail_b64
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_kling_client_image_tail.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Manual live-API probe (NOT a unit test — run once, by hand)**

Run a tiny one-off script against the real Kling API to confirm `image_tail` is accepted on the model we plan to use for I2V. This is a pre-commit sanity check; it is not committed.

```bash
python - <<'PY'
import base64, os, sys
sys.path.insert(0, ".")
from scripts.video_gen.kling_client import KlingClient, encode_image_b64
from scripts.video_gen.config import KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_API_BASE

c = KlingClient(KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_API_BASE)
ref = encode_image_b64("output/videos/mary/reference.png")
# Probe kling-v1-6 first (known to support image_tail per public docs)
url = c.image_to_video(
    prompt="slow push in, soft morning light",
    image_b64=ref,
    image_tail_b64=ref,
    model_name="kling-v1-6",
    mode="pro",
    duration="5",
)
print("OK, model accepted image_tail. Video URL:", url)
PY
```

If the call succeeds, update `config.py` `KLING_VIDEO_MODEL` to `"kling-v1-6"` (I2V only; T2I stays on `kling-v3`) in Step 6. If the call is rejected, resolve by consulting Kling docs for the current image_tail-capable model and substitute; do NOT proceed past this task on a model that silently ignores `image_tail`.

- [ ] **Step 6: Update config.py I2V model to the verified image_tail-capable model**

Modify `scripts/video_gen/config.py:97`:

```python
KLING_VIDEO_MODEL = "kling-v1-6"  # I2V: must support image_tail
```

Leave `KLING_IMAGE_MODEL = "kling-v3"` unchanged (T2I quality is better on v3 and image_tail is irrelevant there).

- [ ] **Step 7: Commit**

```bash
git add scripts/video_gen/kling_client.py scripts/video_gen/config.py tests/test_kling_client_image_tail.py
git commit -m "feat(video-gen): add image_tail support to KlingClient.image_to_video"
```

---

## Task 2: Add `CINEMATIC_SIGNALS` whitelist to config

Data-only change. Defines which signal types get a dedicated shot vs. are filtered out as non-visual before the LLM sees the action list.

**Files:**
- Modify: `scripts/video_gen/config.py` (append new section)

- [ ] **Step 1: Append whitelist to config.py**

Append to `scripts/video_gen/config.py`:

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/video_gen/config.py
git commit -m "feat(video-gen): add CINEMATIC_SIGNALS whitelist and Mary preamble"
```

---

## Task 3: Cinematic action filter

Pure function that filters a scene's raw actions down to cinematic-capable ones, preserving the `signal` name so downstream (LLM + storyboard) can reason about them.

**Files:**
- Modify: `scripts/video_gen/scene_extractor.py` (add filter helper + pass-through of raw signals on manifest entries)
- Create: `tests/test_cinematic_filter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cinematic_filter.py
from scripts.video_gen.scene_extractor import filter_cinematic_actions


def test_filters_out_non_visual_signals():
    signals = [
        {"signal": "engine_status",    "value": "on"},
        {"signal": "seat_heating",     "value": 2},        # filtered
        {"signal": "nav_destination",  "value": "work"},
        {"signal": "acc_distance",     "value": "medium"}, # filtered
        {"signal": "hvac_temp_target", "value": 22},
    ]
    kept = filter_cinematic_actions(signals)
    kept_names = [s["signal"] for s in kept]
    assert kept_names == ["engine_status", "nav_destination", "hvac_temp_target"]


def test_deduplicates_repeated_signals_keeping_last_value():
    signals = [
        {"signal": "hvac_temp_target", "value": 21},
        {"signal": "hvac_temp_target", "value": 22},
    ]
    kept = filter_cinematic_actions(signals)
    assert len(kept) == 1
    assert kept[0]["value"] == 22
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/test_cinematic_filter.py -v`
Expected: FAIL — `filter_cinematic_actions` is not defined.

- [ ] **Step 3: Implement filter in scene_extractor.py**

Add to `scripts/video_gen/scene_extractor.py` (near the top, after imports):

```python
from scripts.video_gen.config import CINEMATIC_SIGNALS


def filter_cinematic_actions(signals: list[dict]) -> list[dict]:
    """Return only signals whose name is in CINEMATIC_SIGNALS.

    For signals that repeat within the scene (e.g. hvac_temp_target set
    twice), keep only the last occurrence — the final state is what gets
    shown on screen.
    """
    last_by_name: dict[str, dict] = {}
    order: list[str] = []
    for sig in signals:
        name = sig.get("signal", "")
        if name not in CINEMATIC_SIGNALS:
            continue
        if name not in last_by_name:
            order.append(name)
        last_by_name[name] = sig
    return [last_by_name[n] for n in order]
```

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_cinematic_filter.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/scene_extractor.py tests/test_cinematic_filter.py
git commit -m "feat(video-gen): add cinematic action filter"
```

---

## Task 4: Storyboard JSON schema + validator

Defines the dataclass / validation layer. No LLM call yet — this task just ensures a given JSON dict either validates or raises with a specific error.

**Files:**
- Create: `scripts/video_gen/storyboard.py`
- Create: `tests/test_storyboard_schema.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storyboard_schema.py
import pytest

from scripts.video_gen.storyboard import (
    Storyboard,
    StoryboardValidationError,
    load_storyboard,
)


VALID = {
    "scene_id": "day1_morning_commute",
    "scene_summary": "ok",
    "keyframes": [
        {"id": "kf0", "role": "establishing_exterior", "prompt": "p0"},
        {"id": "kf1", "role": "pov_transition",        "prompt": "p1"},
        {"id": "kf2", "role": "action_nav",            "prompt": "p2"},
    ],
    "shots": [
        {"id": "shot1", "from_kf": "kf0", "to_kf": "kf1",
         "duration": "5", "narrative_role": "r1", "motion_prompt": "m1"},
        {"id": "shot2", "from_kf": "kf1", "to_kf": "kf2",
         "duration": "5", "narrative_role": "r2", "motion_prompt": "m2"},
    ],
}


def test_valid_storyboard_parses():
    sb = load_storyboard(VALID)
    assert isinstance(sb, Storyboard)
    assert len(sb.shots) == 2
    assert len(sb.keyframes) == 3


def test_rejects_dangling_kf_reference():
    bad = {**VALID}
    bad["shots"] = [
        {"id": "shot1", "from_kf": "kf0", "to_kf": "kf_missing",
         "duration": "5", "narrative_role": "r", "motion_prompt": "m"},
    ]
    with pytest.raises(StoryboardValidationError, match="kf_missing"):
        load_storyboard(bad)


def test_rejects_discontinuous_shots():
    bad = {**VALID}
    bad["shots"] = [
        {"id": "shot1", "from_kf": "kf0", "to_kf": "kf1",
         "duration": "5", "narrative_role": "r", "motion_prompt": "m"},
        # gap: should start at kf1, not kf2
        {"id": "shot2", "from_kf": "kf2", "to_kf": "kf2",
         "duration": "5", "narrative_role": "r", "motion_prompt": "m"},
    ]
    with pytest.raises(StoryboardValidationError, match="discontinuous"):
        load_storyboard(bad)


def test_rejects_wrong_keyframe_count():
    bad = {**VALID}
    bad["keyframes"] = VALID["keyframes"][:2]  # 2 kf but 2 shots → should be 3
    with pytest.raises(StoryboardValidationError, match="keyframe count"):
        load_storyboard(bad)


def test_rejects_keyframe_prompt_over_280_chars():
    bad = {"scene_id": "x", "scene_summary": "s",
           "keyframes": [
               {"id": "kf0", "role": "r", "prompt": "x" * 281},
               {"id": "kf1", "role": "r", "prompt": "p"},
           ],
           "shots": [
               {"id": "s1", "from_kf": "kf0", "to_kf": "kf1",
                "duration": "5", "narrative_role": "r", "motion_prompt": "m"}
           ]}
    with pytest.raises(StoryboardValidationError, match="280"):
        load_storyboard(bad)


def test_rejects_motion_prompt_over_2400_chars():
    bad = {"scene_id": "x", "scene_summary": "s",
           "keyframes": [
               {"id": "kf0", "role": "r", "prompt": "p"},
               {"id": "kf1", "role": "r", "prompt": "p"},
           ],
           "shots": [
               {"id": "s1", "from_kf": "kf0", "to_kf": "kf1",
                "duration": "5", "narrative_role": "r",
                "motion_prompt": "x" * 2401}
           ]}
    with pytest.raises(StoryboardValidationError, match="2400"):
        load_storyboard(bad)
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_storyboard_schema.py -v`
Expected: FAIL — `scripts.video_gen.storyboard` does not exist.

- [ ] **Step 3: Implement storyboard.py**

Create `scripts/video_gen/storyboard.py`:

```python
"""Storyboard data model + validator for the multi-shot video pipeline.

A Storyboard describes one scene as N+1 keyframes (static images at every
transition point) + N shots (each shot = I2V from keyframes[i] to keyframes[i+1]).
Validation is strict and fail-fast: a malformed LLM response is a bug, not a
state to handle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class StoryboardValidationError(ValueError):
    """Raised when a storyboard dict violates the schema invariants."""


@dataclass(frozen=True)
class Keyframe:
    id: str
    role: str
    prompt: str


@dataclass(frozen=True)
class Shot:
    id: str
    from_kf: str
    to_kf: str
    duration: str
    narrative_role: str
    motion_prompt: str


@dataclass(frozen=True)
class Storyboard:
    scene_id: str
    scene_summary: str
    keyframes: tuple[Keyframe, ...]
    shots: tuple[Shot, ...]


_KF_PROMPT_LIMIT = 280
_MOTION_PROMPT_LIMIT = 2400


def load_storyboard(data: dict) -> Storyboard:
    """Validate + build a Storyboard from a dict (typically parsed LLM JSON).

    Raises StoryboardValidationError on any invariant violation, with a
    message that names the offending field/value.
    """
    for key in ("scene_id", "scene_summary", "keyframes", "shots"):
        if key not in data:
            raise StoryboardValidationError(f"missing required field: {key}")

    keyframes = tuple(
        Keyframe(id=kf["id"], role=kf["role"], prompt=kf["prompt"])
        for kf in data["keyframes"]
    )
    shots = tuple(
        Shot(
            id=s["id"],
            from_kf=s["from_kf"],
            to_kf=s["to_kf"],
            duration=s["duration"],
            narrative_role=s["narrative_role"],
            motion_prompt=s["motion_prompt"],
        )
        for s in data["shots"]
    )

    if len(keyframes) != len(shots) + 1:
        raise StoryboardValidationError(
            f"keyframe count mismatch: got {len(keyframes)} keyframes for "
            f"{len(shots)} shots; expected {len(shots) + 1}"
        )

    kf_ids = {kf.id for kf in keyframes}
    for s in shots:
        if s.from_kf not in kf_ids:
            raise StoryboardValidationError(
                f"shot {s.id} references unknown from_kf={s.from_kf}"
            )
        if s.to_kf not in kf_ids:
            raise StoryboardValidationError(
                f"shot {s.id} references unknown to_kf={s.to_kf}"
            )

    for i in range(len(shots) - 1):
        if shots[i].to_kf != shots[i + 1].from_kf:
            raise StoryboardValidationError(
                f"shots {shots[i].id}->{shots[i + 1].id} are discontinuous: "
                f"{shots[i].to_kf} != {shots[i + 1].from_kf}"
            )

    for kf in keyframes:
        if len(kf.prompt) > _KF_PROMPT_LIMIT:
            raise StoryboardValidationError(
                f"keyframe {kf.id}.prompt is {len(kf.prompt)} chars; "
                f"limit is {_KF_PROMPT_LIMIT}"
            )
    for s in shots:
        if len(s.motion_prompt) > _MOTION_PROMPT_LIMIT:
            raise StoryboardValidationError(
                f"shot {s.id}.motion_prompt is {len(s.motion_prompt)} chars; "
                f"limit is {_MOTION_PROMPT_LIMIT}"
            )

    return Storyboard(
        scene_id=data["scene_id"],
        scene_summary=data["scene_summary"],
        keyframes=keyframes,
        shots=shots,
    )


def load_storyboard_file(path: str | Path) -> Storyboard:
    with open(path, "r", encoding="utf-8") as f:
        return load_storyboard(json.load(f))


def dump_storyboard(sb: Storyboard) -> dict:
    """Serialize a Storyboard back to a plain dict (inverse of load_storyboard)."""
    return {
        "scene_id": sb.scene_id,
        "scene_summary": sb.scene_summary,
        "keyframes": [{"id": kf.id, "role": kf.role, "prompt": kf.prompt}
                      for kf in sb.keyframes],
        "shots": [{"id": s.id, "from_kf": s.from_kf, "to_kf": s.to_kf,
                   "duration": s.duration, "narrative_role": s.narrative_role,
                   "motion_prompt": s.motion_prompt}
                  for s in sb.shots],
    }
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_storyboard_schema.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/storyboard.py tests/test_storyboard_schema.py
git commit -m "feat(video-gen): add Storyboard dataclass + strict validator"
```

---

## Task 5: LLM storyboard generator

Wraps the OpenAI call. Given a scene manifest entry + cinematic actions list, produces a validated Storyboard. One retry on validation failure (feed the error back into the next prompt), then fail hard.

**Files:**
- Modify: `scripts/video_gen/storyboard.py` (add `generate_storyboard()`)
- Modify: `scripts/video_gen/config.py` (add `STORYBOARD_SYSTEM_PROMPT`)
- Create: `tests/test_storyboard_generation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storyboard_generation.py
import json
from unittest.mock import MagicMock

import pytest

from scripts.video_gen.storyboard import (
    StoryboardValidationError,
    generate_storyboard,
)


def _fake_openai(responses: list[str]):
    """Build a fake OpenAI client that returns the given JSON strings in order."""
    client = MagicMock()
    calls = iter(responses)

    def _create(**_kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=next(calls)))]
        return resp
    client.chat.completions.create.side_effect = _create
    return client


VALID_JSON = json.dumps({
    "scene_id": "day1_morning_commute",
    "scene_summary": "s",
    "keyframes": [
        {"id": "kf0", "role": "establishing_exterior", "prompt": "p0"},
        {"id": "kf1", "role": "pov_transition",        "prompt": "p1"},
    ],
    "shots": [
        {"id": "shot1", "from_kf": "kf0", "to_kf": "kf1",
         "duration": "5", "narrative_role": "r", "motion_prompt": "m"},
    ],
})


def test_generates_valid_storyboard_on_first_try():
    sb = generate_storyboard(
        openai_client=_fake_openai([VALID_JSON]),
        scene={"id": "day1_morning_commute", "llm_user_prompt": "hello"},
        cinematic_actions=[{"signal": "engine_status", "value": "on"}],
    )
    assert sb.scene_id == "day1_morning_commute"
    assert len(sb.shots) == 1


def test_retries_once_on_validation_error_then_succeeds():
    bad = json.dumps({"scene_id": "x", "scene_summary": "s",
                      "keyframes": [], "shots": []})
    sb = generate_storyboard(
        openai_client=_fake_openai([bad, VALID_JSON]),
        scene={"id": "day1_morning_commute", "llm_user_prompt": "hello"},
        cinematic_actions=[],
    )
    assert sb.scene_id == "day1_morning_commute"


def test_raises_after_second_validation_failure():
    bad = json.dumps({"scene_id": "x", "scene_summary": "s",
                      "keyframes": [], "shots": []})
    with pytest.raises(StoryboardValidationError):
        generate_storyboard(
            openai_client=_fake_openai([bad, bad]),
            scene={"id": "x", "llm_user_prompt": "hello"},
            cinematic_actions=[],
        )


def test_strips_markdown_fences_from_llm_output():
    fenced = "```json\n" + VALID_JSON + "\n```"
    sb = generate_storyboard(
        openai_client=_fake_openai([fenced]),
        scene={"id": "day1_morning_commute", "llm_user_prompt": "x"},
        cinematic_actions=[],
    )
    assert sb.scene_id == "day1_morning_commute"
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_storyboard_generation.py -v`
Expected: FAIL — `generate_storyboard` not defined.

- [ ] **Step 3: Add STORYBOARD_SYSTEM_PROMPT to config.py**

Append to `scripts/video_gen/config.py`:

```python
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
- Each keyframe.prompt ≤ 280 chars (hard limit; shorter is better).
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
```

- [ ] **Step 4: Add generate_storyboard() to storyboard.py**

Append to `scripts/video_gen/storyboard.py`:

```python
import logging

log = logging.getLogger(__name__)


def _strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def _build_user_prompt(scene: dict, cinematic_actions: list[dict]) -> str:
    lines = [scene["llm_user_prompt"], "", "Cinematic actions for this scene:"]
    if not cinematic_actions:
        lines.append("(none — produce a 2-shot storyboard: exterior, then POV)")
    else:
        for a in cinematic_actions:
            lines.append(f"- {a['signal']}: {a.get('value', '')}")
    return "\n".join(lines)


def generate_storyboard(
    openai_client,
    scene: dict,
    cinematic_actions: list[dict],
    model: str = "gpt-4o",
    max_retries: int = 1,
) -> Storyboard:
    """Call OpenAI to produce a validated Storyboard for the given scene.

    On validation failure, retries up to max_retries times with the error
    message appended to the user prompt. After the final retry, re-raises
    StoryboardValidationError.
    """
    from scripts.video_gen.config import STORYBOARD_SYSTEM_PROMPT

    user_prompt = _build_user_prompt(scene, cinematic_actions)
    last_error: StoryboardValidationError | None = None

    for attempt in range(max_retries + 1):
        prompt = user_prompt
        if last_error is not None:
            prompt += (
                f"\n\nYour previous response failed validation: {last_error}\n"
                "Fix the issue and return a valid storyboard."
            )

        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": STORYBOARD_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=2500,
        )
        raw = _strip_markdown_fences(resp.choices[0].message.content)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = StoryboardValidationError(f"LLM returned non-JSON: {e}")
            log.warning(f"attempt {attempt}: {last_error}")
            continue

        try:
            return load_storyboard(data)
        except StoryboardValidationError as e:
            last_error = e
            log.warning(f"attempt {attempt}: storyboard invalid: {e}")

    assert last_error is not None
    raise last_error
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/test_storyboard_generation.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/video_gen/storyboard.py scripts/video_gen/config.py tests/test_storyboard_generation.py
git commit -m "feat(video-gen): LLM storyboard generation with validation retry"
```

---

## Task 6: Keyframe (T2I) module

Given a validated Storyboard, generate each keyframe as a PNG, with character-reference locking for Mary-face consistency.

**Files:**
- Create: `scripts/video_gen/keyframes.py`
- Create: `tests/test_keyframes.py`
- Modify: `scripts/video_gen/kling_client.py` (add `image_reference` param to `text_to_image`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_keyframes.py
import os
from unittest.mock import MagicMock

from scripts.video_gen.keyframes import (
    build_keyframe_prompt,
    generate_keyframes,
)
from scripts.video_gen.storyboard import Keyframe, Shot, Storyboard


def test_build_keyframe_prompt_composes_preamble_body_and_suffix():
    kf = Keyframe(id="kf0", role="establishing_exterior",
                  prompt="parked Renault on a sunny Paris street, 8am light")
    out = build_keyframe_prompt(kf)
    assert out.startswith("Mary, a 30yo European woman")
    assert "parked Renault" in out
    assert out.endswith(
        "French animation style, watercolour textures, "
        "soft pastel palette, ink linework"
    )
    assert len(out) <= 500


def _sb(n_shots=2):
    kfs = tuple(Keyframe(id=f"kf{i}", role="r", prompt=f"p{i}")
                for i in range(n_shots + 1))
    shots = tuple(Shot(id=f"s{i}", from_kf=f"kf{i}", to_kf=f"kf{i+1}",
                        duration="5", narrative_role="r", motion_prompt="m")
                   for i in range(n_shots))
    return Storyboard(scene_id="x", scene_summary="s",
                      keyframes=kfs, shots=shots)


def test_generate_keyframes_calls_kling_once_per_keyframe(tmp_path):
    kling = MagicMock()
    kling.text_to_image.return_value = "http://example/img.png"
    kling.download.side_effect = lambda url, path: open(path, "wb").write(b"\x89PNG")
    sb = _sb(n_shots=2)

    paths = generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=str(tmp_path),
        ref_image_path="/tmp/ref.png",
    )

    assert kling.text_to_image.call_count == 3  # n_shots + 1
    assert [os.path.basename(p) for p in paths] == ["kf0.png", "kf1.png", "kf2.png"]


def test_generate_keyframes_skips_existing_when_not_forced(tmp_path):
    kling = MagicMock()
    sb = _sb(n_shots=1)
    # Pre-create both expected PNGs
    for i in range(2):
        (tmp_path / f"kf{i}.png").write_bytes(b"existing")

    generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=str(tmp_path),
        ref_image_path="/tmp/ref.png",
        force=False,
    )
    kling.text_to_image.assert_not_called()


def test_generate_keyframes_single_target_regenerates_one(tmp_path):
    kling = MagicMock()
    kling.text_to_image.return_value = "http://example/img.png"
    kling.download.side_effect = lambda url, path: open(path, "wb").write(b"\x89PNG")
    sb = _sb(n_shots=2)
    for i in range(3):
        (tmp_path / f"kf{i}.png").write_bytes(b"stale")

    generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=str(tmp_path),
        ref_image_path="/tmp/ref.png",
        only_kf_id="kf1",
    )
    assert kling.text_to_image.call_count == 1
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_keyframes.py -v`
Expected: FAIL — `scripts.video_gen.keyframes` does not exist.

- [ ] **Step 3: Extend `text_to_image` with `image_reference` param**

Modify `scripts/video_gen/kling_client.py` `text_to_image` signature (around lines 97-137). Replace with:

```python
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_name: str = "kling-v2",
        aspect_ratio: str = "1:1",
        n: int = 1,
        image_fidelity: float = 0.5,
        human_fidelity: float = 0.45,
        image_reference_b64: str | None = None,
        reference_type: str = "subject",
    ) -> str:
        """Submit a T2I task, poll until done, return the first image URL.

        When `image_reference_b64` is provided, Kling uses it as a subject/style
        anchor. `reference_type` is "subject" (lock face/identity) or "face".
        """
        if len(prompt) > 500:
            raise KlingError(
                f"T2I prompt is {len(prompt)} chars; Kling limit is 500"
            )
        if negative_prompt and len(negative_prompt) > 200:
            raise KlingError(
                f"T2I negative_prompt is {len(negative_prompt)} chars; "
                f"Kling limit is 200"
            )
        body = {
            "model_name": model_name,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "n": n,
            "image_fidelity": image_fidelity,
            "human_fidelity": human_fidelity,
        }
        if negative_prompt:
            body["negative_prompt"] = negative_prompt
        if image_reference_b64:
            body["image_reference"] = image_reference_b64
            body["reference_type"] = reference_type

        log.info(f"  Kling T2I submit ({model_name}): {prompt[:60]}...")
        resp = self._post("/v1/images/generations", body)
        task_id = resp["data"]["task_id"]
        log.info(f"  T2I task {task_id} submitted, polling...")

        result = self._poll_task("/v1/images/generations", task_id,
                                 poll_interval=8, timeout=600)
        images = result["data"]["task_result"]["images"]
        if not images:
            raise KlingError("T2I succeeded but returned no images")
        return images[0]["url"]
```

- [ ] **Step 4: Implement keyframes.py**

Create `scripts/video_gen/keyframes.py`:

```python
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
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/test_keyframes.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/video_gen/keyframes.py scripts/video_gen/kling_client.py tests/test_keyframes.py
git commit -m "feat(video-gen): T2I keyframe module with subject-reference face lock"
```

---

## Task 7: Contact sheet assembler

Given a keyframes directory, produce a single `contact_sheet.png` for human review.

**Files:**
- Create: `scripts/video_gen/contact_sheet.py`
- Create: `tests/test_contact_sheet.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_contact_sheet.py
from pathlib import Path

from PIL import Image

from scripts.video_gen.contact_sheet import build_contact_sheet
from scripts.video_gen.storyboard import Keyframe, Shot, Storyboard


def _write_png(path: Path, color: tuple[int, int, int]):
    Image.new("RGB", (320, 180), color).save(path)


def _sb(n=3):
    kfs = tuple(Keyframe(id=f"kf{i}", role=f"role{i}", prompt="p")
                for i in range(n))
    shots = tuple(Shot(id=f"s{i}", from_kf=f"kf{i}", to_kf=f"kf{i+1}",
                        duration="5", narrative_role="r", motion_prompt="m")
                   for i in range(n - 1))
    return Storyboard(scene_id="x", scene_summary="s",
                      keyframes=kfs, shots=shots)


def test_contact_sheet_width_is_sum_of_thumbnails(tmp_path):
    sb = _sb(n=3)
    for i in range(3):
        _write_png(tmp_path / f"kf{i}.png", (i * 80, 0, 0))

    out = build_contact_sheet(sb, str(tmp_path))
    img = Image.open(out)
    # 3 thumbs × 320 wide, plus labels row. Just assert it's wide enough.
    assert img.size[0] >= 320 * 3
    assert out.endswith("contact_sheet.png")
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/test_contact_sheet.py -v`
Expected: FAIL — `scripts.video_gen.contact_sheet` missing.

- [ ] **Step 3: Implement contact_sheet.py**

Create `scripts/video_gen/contact_sheet.py`:

```python
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
```

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/test_contact_sheet.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/contact_sheet.py tests/test_contact_sheet.py
git commit -m "feat(video-gen): contact sheet assembler for keyframe review"
```

---

## Task 8: Shots (I2V) module

Given a validated Storyboard + existing keyframe PNGs, render one mp4 per shot using Kling I2V with `image` + `image_tail`.

**Files:**
- Create: `scripts/video_gen/shots.py`
- Create: `tests/test_shots.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_shots.py
import os
from unittest.mock import MagicMock

from scripts.video_gen.shots import generate_shots
from scripts.video_gen.storyboard import Keyframe, Shot, Storyboard


def _sb(n=2):
    kfs = tuple(Keyframe(id=f"kf{i}", role="r", prompt="p")
                for i in range(n + 1))
    shots = tuple(Shot(id=f"shot{i+1}", from_kf=f"kf{i}", to_kf=f"kf{i+1}",
                        duration="5", narrative_role="r", motion_prompt=f"m{i}")
                   for i in range(n))
    return Storyboard(scene_id="x", scene_summary="s",
                      keyframes=kfs, shots=shots)


def test_generate_shots_passes_both_images_to_i2v(tmp_path):
    sb = _sb(n=2)
    # Pre-create the 3 required keyframe PNGs
    import PIL.Image
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    for i in range(3):
        PIL.Image.new("RGB", (16, 9), (i, 0, 0)).save(kf_dir / f"kf{i}.png")
    shots_dir = tmp_path / "shots"

    kling = MagicMock()
    kling.image_to_video.return_value = "http://v/shot.mp4"
    kling.download.side_effect = lambda url, path: open(path, "wb").write(b"vid")

    generate_shots(kling=kling, storyboard=sb,
                   keyframes_dir=str(kf_dir), shots_dir=str(shots_dir))

    assert kling.image_to_video.call_count == 2
    first_call_kwargs = kling.image_to_video.call_args_list[0].kwargs
    assert first_call_kwargs["image_b64"]  # non-empty
    assert first_call_kwargs["image_tail_b64"]
    assert first_call_kwargs["prompt"] == "m0"


def test_generate_shots_skips_existing_mp4(tmp_path):
    sb = _sb(n=1)
    import PIL.Image
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    for i in range(2):
        PIL.Image.new("RGB", (16, 9), (i, 0, 0)).save(kf_dir / f"kf{i}.png")
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    (shots_dir / "shot1.mp4").write_bytes(b"done")

    kling = MagicMock()
    generate_shots(kling=kling, storyboard=sb,
                   keyframes_dir=str(kf_dir), shots_dir=str(shots_dir))
    kling.image_to_video.assert_not_called()


def test_generate_shots_fails_fast_when_keyframe_missing(tmp_path):
    sb = _sb(n=1)
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    # Only create kf0, not kf1
    import PIL.Image
    PIL.Image.new("RGB", (16, 9), (0, 0, 0)).save(kf_dir / "kf0.png")

    import pytest
    with pytest.raises(FileNotFoundError, match="kf1.png"):
        generate_shots(kling=MagicMock(), storyboard=sb,
                       keyframes_dir=str(kf_dir),
                       shots_dir=str(tmp_path / "shots"))
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_shots.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement shots.py**

Create `scripts/video_gen/shots.py`:

```python
"""Per-shot I2V rendering. Reads approved storyboard + keyframes, produces
one mp4 per shot, skips existing files on restart."""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    KLING_VIDEO_ASPECT,
    KLING_VIDEO_CFG,
    KLING_VIDEO_MODEL,
    KLING_VIDEO_MODE,
    NEGATIVE_PROMPT,
)
from scripts.video_gen.kling_client import encode_image_b64
from scripts.video_gen.storyboard import Storyboard

log = logging.getLogger(__name__)


def generate_shots(
    kling,
    storyboard: Storyboard,
    keyframes_dir: str,
    shots_dir: str,
    force: bool = False,
) -> list[str]:
    """Run I2V for each shot. Returns the list of mp4 paths in scene order.

    Fails fast with FileNotFoundError if any required keyframe PNG is missing.
    Skips a shot whose mp4 already exists unless force=True.
    """
    os.makedirs(shots_dir, exist_ok=True)

    # 1. Resolve + verify all keyframe paths up front
    kf_paths: dict[str, str] = {}
    for kf in storyboard.keyframes:
        p = os.path.join(keyframes_dir, f"{kf.id}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"keyframe image missing: {p} — run --stage storyboard first"
            )
        kf_paths[kf.id] = p

    # 2. Render each shot
    out_paths: list[str] = []
    for shot in storyboard.shots:
        out_path = os.path.join(shots_dir, f"{shot.id}.mp4")
        out_paths.append(out_path)

        if os.path.exists(out_path) and not force:
            log.info(f"  {shot.id} — SKIP (exists)")
            continue

        from_b64 = encode_image_b64(kf_paths[shot.from_kf])
        tail_b64 = encode_image_b64(kf_paths[shot.to_kf])

        log.info(f"  {shot.id} ({shot.narrative_role}) — I2V "
                 f"{shot.from_kf}→{shot.to_kf}")
        url = kling.image_to_video(
            prompt=shot.motion_prompt,
            image_b64=from_b64,
            image_tail_b64=tail_b64,
            negative_prompt=NEGATIVE_PROMPT,
            model_name=KLING_VIDEO_MODEL,
            cfg_scale=KLING_VIDEO_CFG,
            mode=KLING_VIDEO_MODE,
            aspect_ratio=KLING_VIDEO_ASPECT,
            duration=shot.duration,
        )
        kling.download(url, out_path)

    return out_paths
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_shots.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/shots.py tests/test_shots.py
git commit -m "feat(video-gen): per-shot I2V renderer with image_tail continuity"
```

---

## Task 9: ffmpeg concat module

Concatenate per-shot mp4s into one per-scene mp4. Uses ffmpeg's concat demuxer (`-f concat -c copy`) for lossless join.

**Files:**
- Create: `scripts/video_gen/concat.py`
- Create: `tests/test_concat.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_concat.py
import os
import subprocess
from unittest.mock import patch

import pytest

from scripts.video_gen.concat import concat_shots


def test_concat_builds_correct_list_file_and_invokes_ffmpeg(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    for name in ("shot1.mp4", "shot2.mp4"):
        (shots_dir / name).write_bytes(b"x")
    out = tmp_path / "scene.mp4"

    recorded = {}

    def fake_run(cmd, **kwargs):
        recorded["cmd"] = cmd
        list_file_idx = cmd.index("-i") + 1
        list_path = cmd[list_file_idx]
        recorded["list"] = open(list_path).read()
        # simulate ffmpeg writing the output
        open(cmd[-1], "wb").write(b"stitched")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with patch("subprocess.run", side_effect=fake_run):
        concat_shots(
            shot_paths=[str(shots_dir / "shot1.mp4"), str(shots_dir / "shot2.mp4")],
            out_path=str(out),
        )

    assert os.path.exists(out)
    assert "shot1.mp4" in recorded["list"]
    assert "shot2.mp4" in recorded["list"]
    assert recorded["cmd"][0] == "ffmpeg"


def test_concat_raises_when_any_shot_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.mp4"):
        concat_shots(shot_paths=[str(tmp_path / "missing.mp4")],
                     out_path=str(tmp_path / "out.mp4"))
```

- [ ] **Step 2: Run tests, verify fail**

Run: `pytest tests/test_concat.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement concat.py**

Create `scripts/video_gen/concat.py`:

```python
"""ffmpeg concat wrapper. Stitches per-shot mp4s into a single per-scene mp4."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

log = logging.getLogger(__name__)


def concat_shots(shot_paths: list[str], out_path: str) -> None:
    """Concat shot_paths (in order) into out_path using ffmpeg concat demuxer.

    Assumes all shots share the same codec/resolution/fps (Kling's output is
    consistent per account). If that assumption breaks in production, the
    implementation should switch to re-encoding via `-c:v libx264 -c:a aac`.
    """
    for p in shot_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"shot mp4 missing: {p}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for p in shot_paths:
            # ffmpeg concat demuxer requires absolute paths + escaped single quotes
            escaped = os.path.abspath(p).replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")
        list_file = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", out_path,
        ]
        log.info(f"  concat {len(shot_paths)} shots → {out_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed: {result.stderr[-500:]}"
            )
    finally:
        try:
            os.unlink(list_file)
        except OSError:
            pass
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_concat.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/concat.py tests/test_concat.py
git commit -m "feat(video-gen): ffmpeg concat module"
```

---

## Task 10: Pipeline refactor — two-stage CLI

Wire everything together. Replace the existing single-scene monolith in `pipeline.py` with three CLI modes: `--stage storyboard`, `--stage video`, `--stage all`.

**Files:**
- Modify: `scripts/video_gen/pipeline.py` (full refactor)
- Create: `tests/test_pipeline_cli.py`

- [ ] **Step 1: Write failing test for CLI routing**

```python
# tests/test_pipeline_cli.py
from unittest.mock import patch

from scripts.video_gen.pipeline import build_arg_parser


def test_parser_requires_stage_or_refonly_or_scene():
    parser = build_arg_parser()
    args = parser.parse_args(["--scene", "day1_morning_commute", "--stage", "storyboard"])
    assert args.scene == "day1_morning_commute"
    assert args.stage == "storyboard"


def test_parser_accepts_regen_keyframe():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--scene", "day1_morning_commute",
        "--stage", "storyboard",
        "--regen-keyframe", "kf2",
    ])
    assert args.regen_keyframe == "kf2"


def test_parser_stage_all_is_valid():
    parser = build_arg_parser()
    args = parser.parse_args(["--scene", "day1_morning_commute", "--stage", "all"])
    assert args.stage == "all"
```

- [ ] **Step 2: Run test, verify fail**

Run: `pytest tests/test_pipeline_cli.py -v`
Expected: FAIL — `build_arg_parser` not found.

- [ ] **Step 3: Refactor pipeline.py**

Replace the entire contents of `scripts/video_gen/pipeline.py` with:

```python
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

    # (a) Storyboard JSON — generate only if missing OR force OR not regen-only
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

    # (b) Keyframes — T2I
    log.info(f"=== {scene['id']} — generate keyframes ({len(sb.keyframes)}) ===")
    generate_keyframes(
        kling=kling,
        storyboard=sb,
        out_dir=kf_dir,
        ref_image_path=REF_IMAGE_PATH,
        force=force,
        only_kf_id=regen_keyframe,
    )

    # (c) Contact sheet
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


# ── Reference image (re-used from old pipeline) ──

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
    if not OPENAI_API_KEY and not args.ref_only:
        sys.exit("OPENAI_API_KEY not set (required unless --ref-only)")

    kling = KlingClient(KLING_ACCESS_KEY, KLING_SECRET_KEY, KLING_API_BASE)

    ensure_reference_image(kling, regen=args.regen_ref)
    if args.ref_only:
        return

    if not args.scene or not args.stage:
        sys.exit("--scene and --stage are required (unless --ref-only)")

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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_pipeline_cli.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Full test-suite regression**

Run: `pytest tests/test_kling_client_image_tail.py tests/test_cinematic_filter.py tests/test_storyboard_schema.py tests/test_storyboard_generation.py tests/test_keyframes.py tests/test_contact_sheet.py tests/test_shots.py tests/test_concat.py tests/test_pipeline_cli.py -v`
Expected: all new tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/video_gen/pipeline.py tests/test_pipeline_cli.py
git commit -m "feat(video-gen): two-stage pipeline CLI with storyboard review gate"
```

---

## Task 11: Smoke test with live APIs on `day1_morning_commute`

End-to-end validation against real Kling + OpenAI. Not a unit test — this is the acceptance gate before running the other 14 scenes.

**Files:** none (manual run)

- [ ] **Step 1: Run storyboard stage**

```bash
python -m scripts.video_gen.pipeline --scene day1_morning_commute --stage storyboard
```

Expected:
- `output/videos/mary/day1/morning_commute/storyboard.json` exists and is valid JSON
- `output/videos/mary/day1/morning_commute/keyframes/kf0.png` … `kfN.png` exist
- `output/videos/mary/day1/morning_commute/keyframes/contact_sheet.png` exists

- [ ] **Step 2: Human review of contact sheet**

Open `contact_sheet.png`. For each keyframe, verify:
- Mary's face looks consistent with `output/videos/mary/reference.png`
- Cabin / car / environment match the scene description
- French watercolour style is intact (not photorealistic, not anime)

For any keyframe that fails review, re-roll it:

```bash
python -m scripts.video_gen.pipeline --scene day1_morning_commute \
    --stage storyboard --regen-keyframe kf3
```

Rebuild contact sheet automatically (the command re-runs contact_sheet.py at the end) and re-review. Iterate until all keyframes pass.

- [ ] **Step 3: Run video stage**

```bash
python -m scripts.video_gen.pipeline --scene day1_morning_commute --stage video
```

Expected:
- `output/videos/mary/day1/morning_commute/shots/shot1.mp4` … `shotN.mp4` exist
- `output/videos/mary/day1/morning_commute/scene.mp4` exists
- Log line reports the final size

- [ ] **Step 4: Play scene.mp4 and verify**

- Total duration ≈ N × 5 seconds
- Adjacent shot boundaries are visually continuous (no harsh cut)
- Each cinematic action from the manifest is visible in some shot
- Style is consistent throughout

- [ ] **Step 5: Document smoke test result**

Append a short note to `docs/superpowers/specs/2026-04-20-video-shot-sequencing-design.md` under a new "Smoke Test 2026-04-20" section recording: number of keyframe re-rolls needed, any visible continuity defects, any model or prompt changes required. This feeds the next iteration.

- [ ] **Step 6: Commit the doc update**

```bash
git add docs/superpowers/specs/2026-04-20-video-shot-sequencing-design.md
git commit -m "docs(video-gen): record day1_morning_commute smoke test result"
```
