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

    Validation order (each step assumes prior steps passed):
      1. Required top-level fields exist
      2. keyframe count == len(shots) + 1
      3. Every shot's from_kf / to_kf resolves to a known keyframe id
      4. Consecutive shots are continuous (shots[i].to_kf == shots[i+1].from_kf)
      5. Prompt length limits (keyframe ≤ 280, motion_prompt ≤ 2400)
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

    # 2. keyframe count invariant
    if len(keyframes) != len(shots) + 1:
        raise StoryboardValidationError(
            f"keyframe count mismatch: got {len(keyframes)} keyframes for "
            f"{len(shots)} shots; expected {len(shots) + 1}"
        )

    # 3. kf-id resolution
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

    # 4. continuity
    for i in range(len(shots) - 1):
        if shots[i].to_kf != shots[i + 1].from_kf:
            raise StoryboardValidationError(
                f"shots {shots[i].id}->{shots[i + 1].id} are discontinuous: "
                f"{shots[i].to_kf} != {shots[i + 1].from_kf}"
            )

    # 5. prompt length limits
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
    """Load and validate a Storyboard from a JSON file on disk."""
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
