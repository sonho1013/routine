# OmniVideo Beat Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed Kling v1.6 I2V shot pipeline with a bookended beat pipeline (exterior_boarding → cabin_pov×N → exterior_driveaway) driven by Kling OmniVideo multi-image.

**Architecture:** Three shared reference images (Mary portrait, Renault exterior, Renault interior) are generated once per run. Per scene, GPT-4o writes a beat-based storyboard; each beat becomes one OmniVideo call with the 2 relevant refs. Beat mp4s are concatenated losslessly into `scene.mp4`. No per-beat T2I keyframes; no v1.6 I2V interpolation.

**Tech Stack:** Python 3.11, pytest, Kling REST (`/v1/videos/omni-video`), OpenAI GPT-4o, Pillow, ffmpeg (already wired via `concat.py`). All changes live under `scripts/video_gen/` and `tests/`.

**Spec:** `docs/superpowers/specs/2026-04-21-omnivideo-beat-pipeline-design.md`.

---

## File Structure

**Created:**
- `scripts/video_gen/references.py` — builds the 3 shared reference PNGs
- `scripts/video_gen/beats.py` — OmniVideo beat generator (replaces `shots.py` in the pipeline; `shots.py` itself stays on disk as dead code)
- `tests/test_kling_omnivideo.py` — unit tests for `KlingClient.omni_video`
- `tests/test_references.py` — unit tests for `references.build_references`
- `tests/test_beats.py` — unit tests for `beats.generate_beats`

**Modified:**
- `scripts/video_gen/kling_client.py` — adds `omni_video(prompt, image_list, duration, mode, aspect_ratio, model_name)` method
- `scripts/video_gen/config.py` — renames `CHARACTER_REF_PROMPT` → also aliased as `MARY_REF_PROMPT`; adds `CAR_EXTERIOR_REF_PROMPT`, `CAR_INTERIOR_REF_PROMPT`, `KLING_OMNI_MODEL`, and replaces `STORYBOARD_SYSTEM_PROMPT`
- `scripts/video_gen/storyboard.py` — full rewrite of dataclasses + `load_storyboard` + `generate_storyboard` (beat schema)
- `scripts/video_gen/pipeline.py` — rewires `run_storyboard_stage` and `run_video_stage` to beats; replaces `ensure_reference_image` with `ensure_references`; updates CLI flags (`--regen-refs`, drop `--regen-keyframe` and `--regen-ref`)
- `tests/test_storyboard_schema.py` — rewritten for beat validator
- `tests/test_storyboard_generation.py` — rewritten for beat LLM generator

**Kept, but no longer called by pipeline:**
- `scripts/video_gen/keyframes.py`, `scripts/video_gen/shots.py`, `scripts/video_gen/contact_sheet.py`, `tests/test_keyframes.py`, `tests/test_shots.py`, `tests/test_contact_sheet.py`. These still reference `Storyboard.keyframes`/`Storyboard.shots`, which will no longer exist after Task 3. **Task 3 therefore deletes these obsolete files** so test collection stays green.

---

## Conventions

- Run tests via the project's existing runner: `pytest tests/<file>.py::<test> -v` from the repo root.
- Python 3.11 is assumed (existing codebase uses `from __future__ import annotations` freely).
- All files use 4-space indent, no tabs.
- Commits follow the existing repo style (no scope prefix, imperative: `feat: ...`, `test: ...`, `refactor: ...`). Co-author trailer is optional.
- When a step says "run the failing test first", that is a real step — don't skip verification that the test fails for the expected reason.
- **TDD rhythm:** write test → see it fail → implement → see it pass → commit.

---

### Task 1: OmniVideo client method

**Files:**
- Modify: `scripts/video_gen/kling_client.py` (add method after `image_to_video`)
- Create: `tests/test_kling_omnivideo.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_kling_omnivideo.py`:

```python
"""Unit tests for KlingClient.omni_video."""
from unittest.mock import MagicMock, patch

import pytest

from scripts.video_gen.kling_client import KlingClient, KlingError


def _make_client() -> KlingClient:
    return KlingClient(
        access_key="ak", secret_key="sk",
        base_url="https://api.klingai.com",
    )


def _mock_response(payload: dict, status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = ""
    return resp


def test_omni_video_submits_multi_image_body_and_returns_url():
    client = _make_client()
    submit_resp = _mock_response({"code": 0, "data": {"task_id": "T1"}})
    poll_resp = _mock_response({"code": 0, "data": {
        "task_status": "succeed",
        "task_result": {"videos": [{"url": "https://cdn/video.mp4"}]},
    }})
    with patch.object(client.session, "post", return_value=submit_resp) as mpost, \
         patch.object(client.session, "get",  return_value=poll_resp) as mget:
        url = client.omni_video(
            prompt="Mary <<<image_1>>> near <<<image_2>>>",
            image_list=["B64A", "B64B"],
            duration="5",
            mode="pro",
            aspect_ratio="16:9",
            model_name="kling-video-o1",
        )
    assert url == "https://cdn/video.mp4"

    # POST went to the OmniVideo endpoint
    (post_url,), post_kwargs = mpost.call_args
    assert post_url.endswith("/v1/videos/omni-video")
    body = post_kwargs["json"]
    assert body["model_name"] == "kling-video-o1"
    assert body["mode"] == "pro"
    assert body["duration"] == "5"
    assert body["aspect_ratio"] == "16:9"
    assert body["prompt"] == "Mary <<<image_1>>> near <<<image_2>>>"
    assert body["image_list"] == [
        {"image_url": "B64A"},
        {"image_url": "B64B"},
    ]

    # GET polled the OmniVideo task endpoint with the returned task_id
    (get_url,), _ = mget.call_args
    assert get_url.endswith("/v1/videos/omni-video/T1")


def test_omni_video_raises_on_failed_task():
    client = _make_client()
    submit_resp = _mock_response({"code": 0, "data": {"task_id": "T2"}})
    poll_resp = _mock_response({"code": 0, "data": {
        "task_status": "failed",
        "task_status_msg": "prompt rejected",
    }})
    with patch.object(client.session, "post", return_value=submit_resp), \
         patch.object(client.session, "get",  return_value=poll_resp):
        with pytest.raises(KlingError, match="prompt rejected"):
            client.omni_video(
                prompt="x", image_list=["B64"], duration="5",
            )


def test_omni_video_rejects_prompt_over_2400_chars():
    client = _make_client()
    with pytest.raises(KlingError, match="2400"):
        client.omni_video(prompt="x" * 2401, image_list=["B64"])


def test_omni_video_rejects_empty_image_list():
    client = _make_client()
    with pytest.raises(KlingError, match="image_list"):
        client.omni_video(prompt="p", image_list=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kling_omnivideo.py -v`
Expected: FAIL with `AttributeError: 'KlingClient' object has no attribute 'omni_video'`.

- [ ] **Step 3: Implement `omni_video` on `KlingClient`**

Edit `scripts/video_gen/kling_client.py`. After the `image_to_video` method and before the `_poll_task` method, add:

```python
    # ── OmniVideo (multi-image storytelling) ──

    def omni_video(
        self,
        prompt: str,
        image_list: list[str],
        duration: str = "5",
        mode: str = "pro",
        aspect_ratio: str = "16:9",
        model_name: str = "kling-video-o1",
    ) -> str:
        """Submit an OmniVideo multi-image task, poll until done, return
        the first video URL.

        `image_list` holds base64-encoded reference images in the order
        the prompt binds them via <<<image_1>>>, <<<image_2>>>, ...
        """
        if not image_list:
            raise KlingError("omni_video requires a non-empty image_list")
        if len(prompt) > 2400:
            raise KlingError(
                f"OmniVideo prompt is {len(prompt)} chars; cap is 2400"
            )
        body = {
            "model_name": model_name,
            "mode": mode,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "prompt": prompt,
            "image_list": [{"image_url": b64} for b64 in image_list],
        }
        log.info(
            f"  Kling OmniVideo submit ({model_name}, {mode}, "
            f"{duration}s, {len(image_list)} refs)"
        )
        resp = self._post("/v1/videos/omni-video", body)
        task_id = resp["data"]["task_id"]
        log.info(f"  OmniVideo task {task_id} submitted, polling...")

        result = self._poll_task(
            "/v1/videos/omni-video", task_id,
            poll_interval=16, timeout=900,
        )
        videos = result["data"]["task_result"]["videos"]
        if not videos:
            raise KlingError("OmniVideo succeeded but returned no videos")
        return videos[0]["url"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kling_omnivideo.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kling_omnivideo.py scripts/video_gen/kling_client.py
git commit -m "feat: add KlingClient.omni_video for multi-image storytelling"
```

---

### Task 2: Reference-image prompts in config

Adds three hardcoded T2I prompts (each < 500 chars) and the OmniVideo model constant. Does not yet change anything that uses them.

**Files:**
- Modify: `scripts/video_gen/config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference_prompts.py`:

```python
"""Config-level checks on the 3 shared T2I reference prompts."""
from scripts.video_gen import config


def test_mary_ref_prompt_is_alias_for_character_ref_prompt():
    assert config.MARY_REF_PROMPT == config.CHARACTER_REF_PROMPT


def test_ref_prompts_fit_kling_500_char_limit():
    for name in ("MARY_REF_PROMPT",
                 "CAR_EXTERIOR_REF_PROMPT",
                 "CAR_INTERIOR_REF_PROMPT"):
        prompt = getattr(config, name)
        assert len(prompt) < 500, f"{name} is {len(prompt)} chars"


def test_car_exterior_prompt_mentions_renault():
    assert "Renault" in config.CAR_EXTERIOR_REF_PROMPT
    assert "losange" in config.CAR_EXTERIOR_REF_PROMPT


def test_car_interior_prompt_is_pov_and_mentions_losange():
    p = config.CAR_INTERIOR_REF_PROMPT
    assert "POV" in p or "first-person" in p
    assert "losange" in p
    # Must NOT include a person (interior ref has to be un-populated so the
    # LLM-bound <<<image_1>>> Mary reference dictates who appears).
    assert "Mary" not in p


def test_omni_model_constant():
    assert config.KLING_OMNI_MODEL == "kling-video-o1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reference_prompts.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'MARY_REF_PROMPT'`.

- [ ] **Step 3: Update `config.py`**

Edit `scripts/video_gen/config.py`. After the existing `CHARACTER_REF_PROMPT` block (which ends with `assert len(CHARACTER_REF_PROMPT) < 500, ...`), add:

```python
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
```

Then, near the existing `KLING_VIDEO_MODEL = "kling-v1-6"` line, add:

```python
KLING_OMNI_MODEL = "kling-video-o1"   # OmniVideo multi-image storytelling
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reference_prompts.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_reference_prompts.py scripts/video_gen/config.py
git commit -m "feat: add shared Renault exterior/interior T2I ref prompts and OmniVideo model const"
```

---

### Task 3: Beat schema + validator (rewrite `storyboard.py`)

This task replaces the keyframe/shot data model with the beat model. The change breaks `keyframes.py`, `shots.py`, and `contact_sheet.py` because they import `Keyframe`/`Shot`/`Storyboard.keyframes`/`Storyboard.shots`. These modules and their tests are deleted in this task — the pipeline no longer uses them and keeping them would block test collection.

**Files:**
- Modify: `scripts/video_gen/storyboard.py` (rewrite dataclasses + `load_storyboard` + `dump_storyboard` + `load_storyboard_file`; leave `generate_storyboard` untouched here — next task updates it)
- Delete: `scripts/video_gen/keyframes.py`, `scripts/video_gen/shots.py`, `scripts/video_gen/contact_sheet.py`
- Delete: `tests/test_keyframes.py`, `tests/test_shots.py`, `tests/test_contact_sheet.py`
- Rewrite: `tests/test_storyboard_schema.py`

- [ ] **Step 1: Write the failing test (rewrite `test_storyboard_schema.py`)**

Overwrite `tests/test_storyboard_schema.py` with:

```python
"""Validator tests for the beat-based Storyboard schema."""
import pytest

from scripts.video_gen.storyboard import (
    Beat,
    Storyboard,
    StoryboardValidationError,
    load_storyboard,
)


def _bookend_boarding():
    return {
        "id": "beat1", "beat_type": "exterior_boarding",
        "actions": [], "duration": "5",
        "motion_prompt": "Mary walks to the Renault and opens the door.",
    }


def _bookend_driveaway():
    return {
        "id": "beatN", "beat_type": "exterior_driveaway",
        "actions": [], "duration": "5",
        "motion_prompt": "The Renault drives off into Paris traffic.",
    }


def _pov(bid: str, actions: list[str]):
    return {
        "id": bid, "beat_type": "cabin_pov",
        "actions": actions, "duration": "5",
        "motion_prompt": "POV of Mary's hands operating controls.",
    }


def _valid_storyboard(pov_beats: list[dict]) -> dict:
    return {
        "scene_id": "day1_morning_commute",
        "scene_summary": "Mary drives to work on a bright Paris morning.",
        "beats": [_bookend_boarding(), *pov_beats, _bookend_driveaway()],
    }


def test_valid_three_beat_storyboard_parses():
    raw = _valid_storyboard([_pov("beat2", ["engine_status",
                                            "gear_position",
                                            "nav_destination"])])
    sb = load_storyboard(raw, expected_actions=[
        "engine_status", "gear_position", "nav_destination",
    ])
    assert isinstance(sb, Storyboard)
    assert len(sb.beats) == 3
    assert isinstance(sb.beats[0], Beat)
    assert sb.beats[0].beat_type == "exterior_boarding"
    assert sb.beats[-1].beat_type == "exterior_driveaway"


def test_valid_four_beat_storyboard_with_two_pov_beats():
    raw = _valid_storyboard([
        _pov("beat2", ["engine_status", "gear_position"]),
        _pov("beat3", ["nav_destination", "hvac_temp_target"]),
    ])
    sb = load_storyboard(raw, expected_actions=[
        "engine_status", "gear_position",
        "nav_destination", "hvac_temp_target",
    ])
    assert len(sb.beats) == 4


def test_rejects_first_beat_not_exterior_boarding():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][0]["beat_type"] = "cabin_pov"
    with pytest.raises(StoryboardValidationError, match="exterior_boarding"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_last_beat_not_exterior_driveaway():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][-1]["beat_type"] = "cabin_pov"
    with pytest.raises(StoryboardValidationError, match="exterior_driveaway"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_exterior_beat_with_actions():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][0]["actions"] = ["engine_status"]
    with pytest.raises(StoryboardValidationError, match="exterior.*actions"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_middle_beat_not_cabin_pov():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][1]["beat_type"] = "exterior_boarding"
    with pytest.raises(StoryboardValidationError, match="cabin_pov"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_wrong_middle_beat_count():
    # 4 actions → expect ceil(4/3)=2 POV beats, but we supply only 1
    raw = _valid_storyboard([_pov("beat2", ["engine_status",
                                            "gear_position",
                                            "nav_destination",
                                            "hvac_temp_target"])])
    with pytest.raises(StoryboardValidationError, match="POV beat count"):
        load_storyboard(raw, expected_actions=[
            "engine_status", "gear_position",
            "nav_destination", "hvac_temp_target",
        ])


def test_rejects_pov_beat_with_too_many_actions():
    raw = _valid_storyboard([_pov("beat2", [
        "engine_status", "gear_position",
        "nav_destination", "hvac_temp_target",
    ])])
    with pytest.raises(StoryboardValidationError, match="at most 3"):
        load_storyboard(raw, expected_actions=[
            "engine_status", "gear_position",
            "nav_destination", "hvac_temp_target",
        ])


def test_rejects_empty_pov_beat():
    raw = _valid_storyboard([_pov("beat2", [])])
    with pytest.raises(StoryboardValidationError,
                       match="at least 1"):
        load_storyboard(raw, expected_actions=[])


def test_rejects_action_multiset_mismatch():
    raw = _valid_storyboard([_pov("beat2", ["engine_status",
                                            "engine_status",
                                            "gear_position"])])
    with pytest.raises(StoryboardValidationError, match="action"):
        load_storyboard(raw, expected_actions=[
            "engine_status", "gear_position", "nav_destination",
        ])


def test_rejects_non_whitelisted_signal():
    raw = _valid_storyboard([_pov("beat2", ["totally_made_up"])])
    with pytest.raises(StoryboardValidationError, match="totally_made_up"):
        load_storyboard(raw, expected_actions=["totally_made_up"])


def test_rejects_motion_prompt_over_2400_chars():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][1]["motion_prompt"] = "x" * 2401
    with pytest.raises(StoryboardValidationError, match="2400"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_duration_other_than_5():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][1]["duration"] = "10"
    with pytest.raises(StoryboardValidationError, match="duration"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_expected_actions_optional_for_schema_only_check():
    """When expected_actions is None, rules 4 and 6 are skipped so the
    generator's internal validation can reuse this function before it
    knows the full action list."""
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    sb = load_storyboard(raw, expected_actions=None)
    assert len(sb.beats) == 3
```

- [ ] **Step 2: Delete the obsolete modules and their tests**

```bash
git rm scripts/video_gen/keyframes.py \
      scripts/video_gen/shots.py \
      scripts/video_gen/contact_sheet.py \
      tests/test_keyframes.py \
      tests/test_shots.py \
      tests/test_contact_sheet.py
```

Also remove their imports from `pipeline.py`. Edit `scripts/video_gen/pipeline.py` and delete these lines (they will be replaced in Task 7, so for now we just prevent a collect error):

```python
from scripts.video_gen.contact_sheet import build_contact_sheet
from scripts.video_gen.keyframes import generate_keyframes
from scripts.video_gen.shots import generate_shots
```

This will break `pipeline.py`'s own body at runtime (it still references those names), but Task 7 rewires it. In the meantime, `pipeline.py` only breaks when run as a script — `pytest` collection won't touch the unused code paths.

- [ ] **Step 3: Run the new test file to verify it fails**

Run: `pytest tests/test_storyboard_schema.py -v`
Expected: FAIL — every test fails with `ImportError: cannot import name 'Beat'` (since the old module doesn't export `Beat` yet).

- [ ] **Step 4: Rewrite `storyboard.py` data model + validator**

Overwrite the top of `scripts/video_gen/storyboard.py` (keep the existing `generate_storyboard` at the bottom for now — it still compiles because it calls `load_storyboard`, which now accepts the new schema). The new file body, up through `load_storyboard_file`:

```python
"""Storyboard data model + validator for the beat-based video pipeline.

A Storyboard describes one scene as a sequence of Beats. The sequence is
strictly bookended:

    beats[0]   : beat_type == "exterior_boarding"   (actions = [])
    beats[1..-2]: beat_type == "cabin_pov"          (1..3 actions each)
    beats[-1]  : beat_type == "exterior_driveaway"  (actions = [])

Every input cinematic action appears in exactly one POV beat. Validation is
strict and fail-fast: a malformed LLM response is a bug, not a state to
handle.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.video_gen.config import CINEMATIC_SIGNALS

log = logging.getLogger(__name__)


class StoryboardValidationError(ValueError):
    """Raised when a storyboard dict violates the schema invariants."""


BeatType = Literal["exterior_boarding", "cabin_pov", "exterior_driveaway"]
_BEAT_TYPES: frozenset[str] = frozenset(
    ("exterior_boarding", "cabin_pov", "exterior_driveaway")
)

_MOTION_PROMPT_LIMIT = 2400
_POV_MAX_ACTIONS = 3


@dataclass(frozen=True)
class Beat:
    id: str
    beat_type: BeatType
    actions: tuple[str, ...]
    duration: Literal["5"]
    motion_prompt: str


@dataclass(frozen=True)
class Storyboard:
    scene_id: str
    scene_summary: str
    beats: tuple[Beat, ...]


def _build_beat(b: dict) -> Beat:
    return Beat(
        id=b["id"],
        beat_type=b["beat_type"],
        actions=tuple(b["actions"]),
        duration=b["duration"],
        motion_prompt=b["motion_prompt"],
    )


def load_storyboard(
    data: dict,
    expected_actions: list[str] | None = None,
) -> Storyboard:
    """Validate and build a Storyboard from a dict (typically parsed LLM JSON).

    When `expected_actions` is None, rules 4 and 6 (middle-beat count and
    action-set match) are skipped. Callers that know the full action list
    (e.g. the generation driver) must pass it.
    """
    for key in ("scene_id", "scene_summary", "beats"):
        if key not in data:
            raise StoryboardValidationError(f"missing required field: {key}")

    if not isinstance(data["beats"], list) or len(data["beats"]) < 2:
        raise StoryboardValidationError(
            "beats must be a list with at least 2 entries (bookend)"
        )

    for b in data["beats"]:
        if b.get("beat_type") not in _BEAT_TYPES:
            raise StoryboardValidationError(
                f"beat {b.get('id')!r} has unknown beat_type "
                f"{b.get('beat_type')!r}"
            )

    beats = tuple(_build_beat(b) for b in data["beats"])

    # 1. bookends
    if beats[0].beat_type != "exterior_boarding":
        raise StoryboardValidationError(
            f"first beat must be exterior_boarding, got {beats[0].beat_type}"
        )
    if beats[-1].beat_type != "exterior_driveaway":
        raise StoryboardValidationError(
            f"last beat must be exterior_driveaway, got {beats[-1].beat_type}"
        )

    # 2. exterior beats carry no actions
    for b in (beats[0], beats[-1]):
        if b.actions:
            raise StoryboardValidationError(
                f"exterior beat {b.id} must have empty actions, got {list(b.actions)}"
            )

    # 3. middle beats are all cabin_pov
    middle = beats[1:-1]
    for b in middle:
        if b.beat_type != "cabin_pov":
            raise StoryboardValidationError(
                f"middle beat {b.id} must be cabin_pov, got {b.beat_type}"
            )

    # 5. per-beat action-count bounds
    for b in middle:
        if len(b.actions) < 1:
            raise StoryboardValidationError(
                f"POV beat {b.id} must have at least 1 action"
            )
        if len(b.actions) > _POV_MAX_ACTIONS:
            raise StoryboardValidationError(
                f"POV beat {b.id} has {len(b.actions)} actions; "
                f"at most {_POV_MAX_ACTIONS} allowed"
            )

    # 7. whitelist
    for b in middle:
        for a in b.actions:
            if a not in CINEMATIC_SIGNALS:
                raise StoryboardValidationError(
                    f"beat {b.id} references non-whitelisted signal {a!r}"
                )

    # 8. motion prompt length
    for b in beats:
        if len(b.motion_prompt) > _MOTION_PROMPT_LIMIT:
            raise StoryboardValidationError(
                f"beat {b.id}.motion_prompt is {len(b.motion_prompt)} chars; "
                f"limit is {_MOTION_PROMPT_LIMIT}"
            )

    # 9. duration fixed at "5"
    for b in beats:
        if b.duration != "5":
            raise StoryboardValidationError(
                f"beat {b.id} duration must be '5', got {b.duration!r}"
            )

    # 4 + 6 require expected_actions
    if expected_actions is not None:
        expected_count = math.ceil(len(expected_actions) / _POV_MAX_ACTIONS) \
            if expected_actions else 0
        if len(middle) != expected_count:
            raise StoryboardValidationError(
                f"POV beat count mismatch: got {len(middle)}, "
                f"expected {expected_count} for {len(expected_actions)} actions"
            )
        covered = [a for b in middle for a in b.actions]
        if Counter(covered) != Counter(expected_actions):
            raise StoryboardValidationError(
                f"action coverage mismatch: beats cover {covered}, "
                f"expected {expected_actions}"
            )

    return Storyboard(
        scene_id=data["scene_id"],
        scene_summary=data["scene_summary"],
        beats=beats,
    )


def load_storyboard_file(
    path: str | Path,
    expected_actions: list[str] | None = None,
) -> Storyboard:
    """Load and validate a Storyboard from a JSON file on disk."""
    with open(path, "r", encoding="utf-8") as f:
        return load_storyboard(json.load(f), expected_actions=expected_actions)


def dump_storyboard(sb: Storyboard) -> dict:
    """Serialize a Storyboard back to a plain dict (inverse of load_storyboard)."""
    return {
        "scene_id": sb.scene_id,
        "scene_summary": sb.scene_summary,
        "beats": [
            {"id": b.id, "beat_type": b.beat_type,
             "actions": list(b.actions), "duration": b.duration,
             "motion_prompt": b.motion_prompt}
            for b in sb.beats
        ],
    }
```

**Keep everything below the original `dump_storyboard` function (markdown-fence stripper, `_build_user_prompt`, `generate_storyboard`) untouched for now.** Task 4 updates the generator.

- [ ] **Step 5: Run storyboard schema tests**

Run: `pytest tests/test_storyboard_schema.py -v`
Expected: 14 passed.

- [ ] **Step 6: Run the full test suite to check for collateral breakage**

Run: `pytest -x --ignore=tests/test_storyboard_generation.py -q`
Expected: all tests pass except that `tests/test_storyboard_generation.py` may fail (we'll fix it in Task 4 — that's why we skip it here). If any *other* file fails, stop and investigate.

- [ ] **Step 7: Commit**

```bash
git add -A scripts/video_gen/storyboard.py scripts/video_gen/pipeline.py \
          tests/test_storyboard_schema.py
git commit -m "refactor: replace keyframe/shot storyboard schema with beat model"
```

---

### Task 4: Beat-aware storyboard LLM generator

Rewrites `STORYBOARD_SYSTEM_PROMPT` and adapts `generate_storyboard` to pass `expected_actions` into the validator. Rewrites `tests/test_storyboard_generation.py`.

**Files:**
- Modify: `scripts/video_gen/config.py` (replace `STORYBOARD_SYSTEM_PROMPT`)
- Modify: `scripts/video_gen/storyboard.py` (update `_build_user_prompt` + `generate_storyboard` bottom half)
- Rewrite: `tests/test_storyboard_generation.py`

- [ ] **Step 1: Write the failing test (rewrite `test_storyboard_generation.py`)**

Overwrite `tests/test_storyboard_generation.py`:

```python
"""Generator tests for the beat-based storyboard LLM flow."""
import json
from unittest.mock import MagicMock

import pytest

from scripts.video_gen.storyboard import (
    StoryboardValidationError,
    generate_storyboard,
)


def _fake_openai(responses: list[str]):
    """OpenAI stub that returns `responses` in order from chat.completions.create."""
    client = MagicMock()
    calls = iter(responses)

    def _create(**_kwargs):
        r = MagicMock()
        r.choices = [MagicMock(message=MagicMock(content=next(calls)))]
        return r
    client.chat.completions.create.side_effect = _create
    return client


def _beat_json(pov_actions: list[list[str]]) -> str:
    beats = [
        {"id": "beat1", "beat_type": "exterior_boarding",
         "actions": [], "duration": "5",
         "motion_prompt": "Mary walks up to the Renault and opens the door."},
    ]
    for i, acts in enumerate(pov_actions, start=2):
        beats.append({
            "id": f"beat{i}", "beat_type": "cabin_pov",
            "actions": acts, "duration": "5",
            "motion_prompt": "POV of Mary's hands on the controls.",
        })
    beats.append({
        "id": f"beat{len(pov_actions) + 2}",
        "beat_type": "exterior_driveaway",
        "actions": [], "duration": "5",
        "motion_prompt": "The Renault drives off down the street.",
    })
    return json.dumps({
        "scene_id": "day1_morning_commute",
        "scene_summary": "Mary's morning commute.",
        "beats": beats,
    })


SCENE = {"id": "day1_morning_commute", "llm_user_prompt": "Morning commute."}


def test_generates_valid_storyboard_on_first_try():
    sb = generate_storyboard(
        openai_client=_fake_openai([_beat_json([["engine_status",
                                                 "gear_position",
                                                 "nav_destination"]])]),
        scene=SCENE,
        cinematic_actions=[
            {"signal": "engine_status", "value": "on"},
            {"signal": "gear_position", "value": "D"},
            {"signal": "nav_destination", "value": "office"},
        ],
    )
    assert sb.scene_id == "day1_morning_commute"
    assert len(sb.beats) == 3


def test_retries_once_on_validation_error_then_succeeds():
    bad = json.dumps({"scene_id": "x", "scene_summary": "s", "beats": []})
    good = _beat_json([["engine_status"]])
    sb = generate_storyboard(
        openai_client=_fake_openai([bad, good]),
        scene=SCENE,
        cinematic_actions=[{"signal": "engine_status", "value": "on"}],
    )
    assert sb.scene_id == "day1_morning_commute"


def test_raises_after_final_validation_failure():
    bad = json.dumps({"scene_id": "x", "scene_summary": "s", "beats": []})
    with pytest.raises(StoryboardValidationError):
        generate_storyboard(
            openai_client=_fake_openai([bad, bad]),
            scene=SCENE,
            cinematic_actions=[{"signal": "engine_status", "value": "on"}],
            max_retries=1,
        )


def test_strips_markdown_fences_from_llm_output():
    fenced = "```json\n" + _beat_json([["engine_status"]]) + "\n```"
    sb = generate_storyboard(
        openai_client=_fake_openai([fenced]),
        scene=SCENE,
        cinematic_actions=[{"signal": "engine_status", "value": "on"}],
    )
    assert sb.scene_id == "day1_morning_commute"


def test_user_prompt_includes_expected_beat_count_hint():
    """The user prompt the LLM sees should tell it how many POV beats to emit."""
    client = _fake_openai([_beat_json([["engine_status", "gear_position"],
                                       ["nav_destination", "hvac_temp_target"]])])
    generate_storyboard(
        openai_client=client,
        scene=SCENE,
        cinematic_actions=[
            {"signal": "engine_status", "value": "on"},
            {"signal": "gear_position", "value": "D"},
            {"signal": "nav_destination", "value": "office"},
            {"signal": "hvac_temp_target", "value": 22},
        ],
    )
    user_msg = client.chat.completions.create.call_args_list[0].kwargs["messages"][1]
    # 4 actions → ceil(4/3) = 2 POV beats
    assert "2 cabin_pov beats" in user_msg["content"]
    assert "Total beats: 4" in user_msg["content"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_storyboard_generation.py -v`
Expected: every test fails — existing generator sends the old `_build_user_prompt` format which does not include the count hint; also the returned JSON uses the new beat schema but the generator did not know to pass `expected_actions`, so the multiset check fires.

- [ ] **Step 3: Replace `STORYBOARD_SYSTEM_PROMPT` in `config.py`**

Delete the current `STORYBOARD_SYSTEM_PROMPT = """..."""` block (starting at the comment `# ── Storyboard LLM Prompt ──`) and replace it with:

```python
# ── Storyboard LLM Prompt ──
STORYBOARD_SYSTEM_PROMPT = """\
You are a cinematic storyboard writer for French-style animated short films \
about Mary's daily commute in a Renault electric car in Paris. You write \
OmniVideo motion prompts that render through Kling's multi-image storytelling model.

Given a scene's metadata plus a filtered list of "cinematic" driving actions, \
produce a *beat-based* storyboard with a strict bookend arc:

  1. exterior_boarding   (3rd-person, Mary outside; no actions)
  2. cabin_pov x N       (1st-person POV; 1..3 actions per beat)
  3. exterior_driveaway  (3rd-person, car drives away; no actions)

Requirements:
- Total beats = N + 2, where N = ceil(action_count / 3).
- Distribute ALL provided actions across the POV beats, at most 3 per beat, \
  with no duplicates and in physically plausible causal order: engine_status \
  must precede gear_position; gear_position must precede any driving-adjacent \
  action (hvac_temp_target, nav_destination, nav_route_pref, media_content_id, \
  drive_mode, wiper_state, window_position, door_status, keyless_entry).
- Each beat.duration MUST be "5".
- POV means over-shoulder or first-person of Mary's hands on controls. Mary \
  may be partially visible (hand, sleeve, scarf) but never in a frontal \
  portrait shot.
- Never describe Mary's physical appearance — <<<image_1>>> locks it.
- In motion_prompt, reference the character via <<<image_1>>> and the setting \
  via <<<image_2>>>. For exterior beats, <<<image_2>>> is the Renault 3/4 \
  exterior; for POV beats it is the Renault dashboard interior (IVI, gear \
  shifter, steering wheel losange).
- Mention the silver Renault diamond losange logo when a POV beat shows the \
  steering wheel.
- Each motion_prompt ≤ 2400 chars; target 2-5 sentences.
- End every motion_prompt with the style tail: \
  "French animation style, watercolor textures, soft pastel palette, ink linework."

Output STRICT JSON (no markdown fences) matching:
{
  "scene_id": "<echo the scene id>",
  "scene_summary": "<one sentence>",
  "beats": [
    {"id": "beat1", "beat_type": "exterior_boarding",
     "actions": [], "duration": "5", "motion_prompt": "..."},
    {"id": "beat2", "beat_type": "cabin_pov",
     "actions": ["signal_a", ...], "duration": "5", "motion_prompt": "..."},
    ...,
    {"id": "beatN", "beat_type": "exterior_driveaway",
     "actions": [], "duration": "5", "motion_prompt": "..."}
  ]
}
"""
```

- [ ] **Step 4: Update `generate_storyboard` and `_build_user_prompt` in `storyboard.py`**

At the bottom of `scripts/video_gen/storyboard.py`, replace the current `_strip_markdown_fences`, `_build_user_prompt`, and `generate_storyboard` with:

```python
def _strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def _build_user_prompt(scene: dict, cinematic_actions: list[dict]) -> str:
    n_pov = math.ceil(len(cinematic_actions) / _POV_MAX_ACTIONS) \
        if cinematic_actions else 0
    total = n_pov + 2
    lines = [
        scene["llm_user_prompt"],
        "",
        f"Cinematic actions ({len(cinematic_actions)}):",
    ]
    if not cinematic_actions:
        # Degenerate case: no POV beats, bookend only. This shouldn't happen
        # in production scenes (the manifest filter always has ≥1 action),
        # but we make the behavior explicit rather than raise.
        lines.append("(none — produce only the 2 exterior bookend beats)")
    else:
        for a in cinematic_actions:
            lines.append(f"  - {a['signal']}: {a.get('value', '')}")
    lines += [
        "",
        f"Produce {n_pov} cabin_pov beats between the two exterior bookends. "
        f"Total beats: {total}.",
    ]
    return "\n".join(lines)


def generate_storyboard(
    openai_client,
    scene: dict,
    cinematic_actions: list[dict],
    model: str = "gpt-4o",
    max_retries: int = 3,
) -> Storyboard:
    """Call OpenAI to produce a validated beat-based Storyboard.

    On validation failure, retries up to max_retries times with the error
    message appended to the user prompt. After the final retry, re-raises
    StoryboardValidationError.
    """
    from scripts.video_gen.config import STORYBOARD_SYSTEM_PROMPT

    expected_actions = [a["signal"] for a in cinematic_actions]
    user_prompt = _build_user_prompt(scene, cinematic_actions)
    last_error: StoryboardValidationError | None = None

    for attempt in range(max_retries + 1):
        prompt = user_prompt
        if last_error is not None:
            prompt += (
                f"\n\nYour previous response failed validation: {last_error}\n"
                "Fix the issue and return a valid beat-based storyboard."
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
            return load_storyboard(data, expected_actions=expected_actions)
        except StoryboardValidationError as e:
            last_error = e
            log.warning(f"attempt {attempt}: storyboard invalid: {e}")

    assert last_error is not None
    raise last_error
```

Remove the old `import logging` / `log = ...` pair at the bottom of the file (the new top-of-file block already defines them).

- [ ] **Step 5: Run the new tests**

Run: `pytest tests/test_storyboard_generation.py -v`
Expected: 5 passed.

- [ ] **Step 6: Run full suite to verify no regressions**

Run: `pytest -q`
Expected: all tests pass. (Tests from tasks 1–3 should still pass; obsolete tests were deleted.)

- [ ] **Step 7: Commit**

```bash
git add scripts/video_gen/config.py scripts/video_gen/storyboard.py \
        tests/test_storyboard_generation.py
git commit -m "feat: beat-based storyboard LLM generator with expected-actions validator"
```

---

### Task 5: Reference image builder (`references.py`)

**Files:**
- Create: `scripts/video_gen/references.py`
- Create: `tests/test_references.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_references.py`:

```python
"""Unit tests for references.build_references — the 3-image shared ref builder."""
from unittest.mock import MagicMock

from scripts.video_gen.references import build_references


def _fake_kling():
    k = MagicMock()
    k.text_to_image.return_value = "https://cdn/img.png"
    k.download.side_effect = lambda url, dest: open(dest, "wb").write(b"x")
    return k


def test_build_references_calls_t2i_three_times_on_empty_dir(tmp_path):
    kling = _fake_kling()
    paths = build_references(kling, out_dir=str(tmp_path), force=False)
    assert set(paths.keys()) == {
        "mary_ref", "car_exterior_ref", "car_interior_ref",
    }
    assert kling.text_to_image.call_count == 3
    for p in paths.values():
        assert p.startswith(str(tmp_path))
        assert p.endswith(".png")
        # Each entry was downloaded (file exists and is non-empty)
        import os
        assert os.path.getsize(p) > 0


def test_build_references_is_noop_when_files_exist(tmp_path):
    # Pre-create all 3 files
    for name in ("mary_ref", "car_exterior_ref", "car_interior_ref"):
        (tmp_path / f"{name}.png").write_bytes(b"stale")
    kling = _fake_kling()
    paths = build_references(kling, out_dir=str(tmp_path), force=False)
    assert kling.text_to_image.call_count == 0
    assert len(paths) == 3
    # Existing files are not overwritten
    assert (tmp_path / "mary_ref.png").read_bytes() == b"stale"


def test_build_references_force_regenerates_all(tmp_path):
    for name in ("mary_ref", "car_exterior_ref", "car_interior_ref"):
        (tmp_path / f"{name}.png").write_bytes(b"stale")
    kling = _fake_kling()
    build_references(kling, out_dir=str(tmp_path), force=True)
    assert kling.text_to_image.call_count == 3


def test_each_ref_uses_correct_aspect_ratio(tmp_path):
    kling = _fake_kling()
    build_references(kling, out_dir=str(tmp_path), force=False)
    calls_by_prompt_prefix = {
        c.kwargs["prompt"][:20]: c.kwargs["aspect_ratio"]
        for c in kling.text_to_image.call_args_list
    }
    # Mary uses 1:1; both car refs use 16:9
    assert set(calls_by_prompt_prefix.values()) == {"1:1", "16:9"}
    ratios = list(calls_by_prompt_prefix.values())
    assert ratios.count("1:1") == 1
    assert ratios.count("16:9") == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_references.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.video_gen.references'`.

- [ ] **Step 3: Implement `references.py`**

Create `scripts/video_gen/references.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_references.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/references.py tests/test_references.py
git commit -m "feat: build_references — generate 3 shared T2I anchor images"
```

---

### Task 6: Beat generator (`beats.py`)

**Files:**
- Create: `scripts/video_gen/beats.py`
- Create: `tests/test_beats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_beats.py`:

```python
"""Unit tests for beats.generate_beats."""
import os
from unittest.mock import MagicMock

from scripts.video_gen.beats import REF_SELECTION, generate_beats
from scripts.video_gen.storyboard import Beat, Storyboard


def _ref_paths(tmp_path) -> dict:
    paths = {}
    for name in ("mary_ref", "car_exterior_ref", "car_interior_ref"):
        p = tmp_path / f"{name}.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        paths[name] = str(p)
    return paths


def _three_beat_storyboard() -> Storyboard:
    return Storyboard(
        scene_id="day1_morning_commute",
        scene_summary="test",
        beats=(
            Beat(id="beat1", beat_type="exterior_boarding", actions=(),
                 duration="5",
                 motion_prompt="Mary <<<image_1>>> near <<<image_2>>>."),
            Beat(id="beat2", beat_type="cabin_pov",
                 actions=("engine_status", "gear_position"),
                 duration="5",
                 motion_prompt="POV hands on wheel <<<image_2>>>."),
            Beat(id="beat3", beat_type="exterior_driveaway", actions=(),
                 duration="5",
                 motion_prompt="Car drives off <<<image_2>>>."),
        ),
    )


def _fake_kling_with_video():
    k = MagicMock()
    k.omni_video.return_value = "https://cdn/v.mp4"
    k.download.side_effect = lambda url, dest: open(dest, "wb").write(b"v")
    return k


def test_ref_selection_covers_all_three_beat_types():
    assert set(REF_SELECTION.keys()) == {
        "exterior_boarding", "cabin_pov", "exterior_driveaway",
    }
    # POV beat uses the interior ref; both exterior beats use the exterior ref
    assert REF_SELECTION["cabin_pov"] == ("mary_ref", "car_interior_ref")
    assert REF_SELECTION["exterior_boarding"] == ("mary_ref", "car_exterior_ref")
    assert REF_SELECTION["exterior_driveaway"] == ("mary_ref", "car_exterior_ref")


def test_generate_beats_calls_omni_once_per_beat_with_correct_refs(tmp_path):
    kling = _fake_kling_with_video()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    refs = _ref_paths(tmp_path)

    paths = generate_beats(
        kling=kling, storyboard=sb,
        out_dir=str(out_dir), ref_paths=refs,
    )

    assert kling.omni_video.call_count == 3
    # First call (exterior_boarding) uses mary_ref + car_exterior_ref
    call_args = kling.omni_video.call_args_list
    assert len(call_args[0].kwargs["image_list"]) == 2
    # File names encode beat type
    assert paths[0].endswith("beat1_exterior_boarding.mp4")
    assert paths[1].endswith("beat2_cabin_pov.mp4")
    assert paths[2].endswith("beat3_exterior_driveaway.mp4")
    for p in paths:
        assert os.path.getsize(p) > 0


def test_generate_beats_skips_existing_files(tmp_path):
    kling = _fake_kling_with_video()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    beats_dir = out_dir / "beats"
    beats_dir.mkdir(parents=True)
    # Pre-write all 3 beat mp4s
    for name in ("beat1_exterior_boarding.mp4",
                 "beat2_cabin_pov.mp4",
                 "beat3_exterior_driveaway.mp4"):
        (beats_dir / name).write_bytes(b"cached")

    refs = _ref_paths(tmp_path)
    generate_beats(kling=kling, storyboard=sb,
                   out_dir=str(out_dir), ref_paths=refs)
    assert kling.omni_video.call_count == 0


def test_generate_beats_force_regenerates(tmp_path):
    kling = _fake_kling_with_video()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    beats_dir = out_dir / "beats"
    beats_dir.mkdir(parents=True)
    for name in ("beat1_exterior_boarding.mp4",
                 "beat2_cabin_pov.mp4",
                 "beat3_exterior_driveaway.mp4"):
        (beats_dir / name).write_bytes(b"cached")

    refs = _ref_paths(tmp_path)
    generate_beats(kling=kling, storyboard=sb,
                   out_dir=str(out_dir), ref_paths=refs, force=True)
    assert kling.omni_video.call_count == 3


def test_generate_beats_raises_if_ref_missing(tmp_path):
    kling = _fake_kling_with_video()
    sb = _three_beat_storyboard()
    out_dir = tmp_path / "scene"
    # Only provide 1 of the 3 refs
    refs = {"mary_ref": str(tmp_path / "does_not_exist.png")}
    try:
        generate_beats(kling=kling, storyboard=sb,
                       out_dir=str(out_dir), ref_paths=refs)
    except KeyError as e:
        assert "car_exterior_ref" in str(e)
    else:
        raise AssertionError("expected KeyError for missing ref")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_beats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.video_gen.beats'`.

- [ ] **Step 3: Implement `beats.py`**

Create `scripts/video_gen/beats.py`:

```python
"""Per-beat OmniVideo rendering. Reads a validated Storyboard + 3 shared
reference images, produces one mp4 per beat, skips existing files on restart.
"""
from __future__ import annotations

import logging
import os

from scripts.video_gen.config import (
    KLING_OMNI_MODEL,
    KLING_VIDEO_ASPECT,
    KLING_VIDEO_MODE,
)
from scripts.video_gen.kling_client import encode_image_b64
from scripts.video_gen.storyboard import Beat, Storyboard

log = logging.getLogger(__name__)

REF_SELECTION: dict[str, tuple[str, str]] = {
    "exterior_boarding":  ("mary_ref", "car_exterior_ref"),
    "cabin_pov":          ("mary_ref", "car_interior_ref"),
    "exterior_driveaway": ("mary_ref", "car_exterior_ref"),
}


def _select_refs(beat: Beat, ref_paths: dict[str, str]) -> list[str]:
    """Return the 2 base64 strings for the beat's type, in <<<image_N>>> order."""
    b64s: list[str] = []
    for name in REF_SELECTION[beat.beat_type]:
        if name not in ref_paths:
            raise KeyError(
                f"reference {name!r} required for beat_type "
                f"{beat.beat_type!r} not found in ref_paths"
            )
        b64s.append(encode_image_b64(ref_paths[name]))
    return b64s


def generate_beats(
    kling,
    storyboard: Storyboard,
    out_dir: str,
    ref_paths: dict[str, str],
    force: bool = False,
) -> list[str]:
    """Render one OmniVideo mp4 per beat. Returns paths in beat order.

    Skips a beat whose mp4 already exists unless force=True.
    """
    beats_dir = os.path.join(out_dir, "beats")
    os.makedirs(beats_dir, exist_ok=True)

    out_paths: list[str] = []
    for beat in storyboard.beats:
        filename = f"{beat.id}_{beat.beat_type}.mp4"
        out_path = os.path.join(beats_dir, filename)
        out_paths.append(out_path)

        if os.path.exists(out_path) and not force:
            log.info(f"  {beat.id} ({beat.beat_type}) — SKIP (exists)")
            continue

        image_list = _select_refs(beat, ref_paths)
        log.info(f"  {beat.id} ({beat.beat_type}) — OmniVideo")
        url = kling.omni_video(
            prompt=beat.motion_prompt,
            image_list=image_list,
            duration=beat.duration,
            mode=KLING_VIDEO_MODE,
            aspect_ratio=KLING_VIDEO_ASPECT,
            model_name=KLING_OMNI_MODEL,
        )
        kling.download(url, out_path)

    return out_paths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_beats.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/beats.py tests/test_beats.py
git commit -m "feat: generate_beats — one OmniVideo call per beat"
```

---

### Task 7: Pipeline rewire

Updates `pipeline.py` to use `build_references` + `generate_beats` + `concat_shots`. Drops the `--regen-keyframe` and `--regen-ref` flags; adds `--regen-refs`. Drops contact sheet (no per-scene keyframes anymore).

**Files:**
- Modify: `scripts/video_gen/pipeline.py`
- Modify: `scripts/video_gen/concat.py` (verify it accepts `list[str]` of any mp4 paths — rename internal variable if it mentions shots specifically; we keep the public `concat_shots` function name because the tests still import it)

- [ ] **Step 1: Check `concat.py` still works with beat paths**

Run: `pytest tests/test_concat.py -v`
Expected: pass. The concat module is path-agnostic; no code change needed. If the test suite does not currently exist skip this step.

- [ ] **Step 2: Rewrite `pipeline.py`**

Overwrite `scripts/video_gen/pipeline.py` with:

```python
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
```

- [ ] **Step 3: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass. No failures.

- [ ] **Step 4: Sanity-check CLI argparse**

Run: `python -m scripts.video_gen.pipeline --help`
Expected: help text shows `--scene`, `--stage`, `--force`, `--regen-refs`, `--refs-only` and NOT `--regen-keyframe` or `--regen-ref`.

- [ ] **Step 5: Commit**

```bash
git add scripts/video_gen/pipeline.py
git commit -m "refactor: wire pipeline.py to OmniVideo beat stages"
```

---

### Task 8: Live smoke test on `day1_morning_commute`

End-to-end run against the real Kling API. Produces artefacts the user reviews. This task has no pytest assertions — success is user judgement against the spec's acceptance criteria.

**Files:**
- No code changes. Outputs written to `output/videos/refs/` and `output/videos/mary/day1/morning_commute/`.

- [ ] **Step 1: Check that KLING + OPENAI env are set (or fall back to hardcoded)**

Run: `echo "kling=${KLING_ACCESS_KEY:0:4}... openai=${OPENAI_API_KEY:0:4}..."`
Expected: OPENAI key shows a prefix; KLING key is hardcoded in `config.py` (fallback value) so missing env is fine.

- [ ] **Step 2: Clear any stale artefacts from the previous (rejected) run**

```bash
rm -rf output/videos/refs
rm -rf output/videos/mary/day1/morning_commute/storyboard.json \
       output/videos/mary/day1/morning_commute/beats \
       output/videos/mary/day1/morning_commute/scene.mp4 \
       output/videos/mary/day1/morning_commute/keyframes \
       output/videos/mary/day1/morning_commute/shots
```

(The `keyframes/` and `shots/` directories are leftovers from the deleted pipeline; safe to wipe.)

- [ ] **Step 3: Build the 3 shared references once**

Run: `python -m scripts.video_gen.pipeline --refs-only`
Expected: 3 T2I calls, 3 PNGs under `output/videos/refs/`. Inspect the PNGs visually before proceeding. If any ref looks wrong, re-run with `--regen-refs`.

- [ ] **Step 4: Run the end-to-end pipeline for `day1_morning_commute`**

Run: `python -m scripts.video_gen.pipeline --scene day1_morning_commute --stage all`
Expected:
- Storyboard generation logs show N (≥1) POV beats between the two exterior bookends.
- `storyboard.json` written, containing only the new beat schema (no `keyframes`/`shots` keys).
- OmniVideo calls fire once per beat; each downloads to `output/videos/mary/day1/morning_commute/beats/beatN_<type>.mp4`.
- Final `scene.mp4` is written. Its duration is `5 × (N + 2)` seconds.

- [ ] **Step 5: Document the smoke test result**

Append to `docs/superpowers/specs/2026-04-21-omnivideo-beat-pipeline-design.md` a new section `## Smoke test result (YYYY-MM-DD)` containing:

- Storyboard beat count, scene duration, scene.mp4 size.
- Pass/fail against each acceptance criterion in the spec's "Smoke test" section (user-supplied judgement).
- Any prompt regressions observed (e.g. "losange invisible on steering wheel") with a short note of whether the fix is a prompt-tuning iteration or a deeper re-architecture.

Commit:

```bash
git add docs/superpowers/specs/2026-04-21-omnivideo-beat-pipeline-design.md
git commit -m "docs: record OmniVideo smoke test result on day1_morning_commute"
```
