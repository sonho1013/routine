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

    # 4. middle-beat count (requires expected_actions)
    if expected_actions is not None:
        expected_count = math.ceil(len(expected_actions) / _POV_MAX_ACTIONS) \
            if expected_actions else 0
        if len(middle) != expected_count:
            raise StoryboardValidationError(
                f"POV beat count mismatch: got {len(middle)}, "
                f"expected {expected_count} for {len(expected_actions)} actions"
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

    # 6. action coverage (requires expected_actions)
    if expected_actions is not None:
        covered = [a for b in middle for a in b.actions]
        if Counter(covered) != Counter(expected_actions):
            raise StoryboardValidationError(
                f"action coverage mismatch: beats cover {covered}, "
                f"expected {expected_actions}"
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
    max_retries: int = 3,
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
