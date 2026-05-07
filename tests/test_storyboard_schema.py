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


def _pov(bid: str, actions: list[str], keyframe_prompt: str = "Dashboard with controls highlighted."):
    return {
        "id": bid, "beat_type": "cabin_pov",
        "actions": actions, "duration": "5",
        "motion_prompt": "POV of Mary's hands operating controls.",
        "keyframe_prompt": keyframe_prompt,
    }


def _valid_storyboard(pov_beats: list[dict]) -> dict:
    return {
        "scene_id": "day1_morning_commute",
        "scene_summary": "Mary drives to work on a bright Paris morning.",
        "beats": [_bookend_boarding(), *pov_beats, _bookend_driveaway()],
    }


def test_valid_three_beat_storyboard_parses():
    raw = _valid_storyboard([_pov("beat2", ["engine_status",
                                            "gear_position"])])
    sb = load_storyboard(raw, expected_actions=[
        "engine_status", "gear_position",
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
    # 4 actions → expect ceil(4/2)=2 POV beats; supply 1 beat with 2 actions
    # so rule 5 (per-beat bounds) passes cleanly and only rule 4 fires.
    raw = _valid_storyboard([_pov("beat2", ["engine_status", "gear_position"])])
    with pytest.raises(StoryboardValidationError, match="POV beat count"):
        load_storyboard(raw, expected_actions=[
            "engine_status", "gear_position",
            "nav_destination", "hvac_temp_target",
        ])


def test_rejects_pov_beat_with_too_many_actions():
    # Schema-only check (expected_actions=None) isolates rule 5 from rule 4.
    raw = _valid_storyboard([_pov("beat2", [
        "engine_status", "gear_position", "nav_destination",
    ])])
    with pytest.raises(StoryboardValidationError, match="at most 2"):
        load_storyboard(raw, expected_actions=None)


def test_rejects_empty_pov_beat():
    # 1 action → expect 1 POV beat; supply 1 beat with 0 actions so rule 4
    # passes (count matches) and only rule 5 fires on the per-beat bound.
    raw = _valid_storyboard([_pov("beat2", [])])
    with pytest.raises(StoryboardValidationError,
                       match="at least 1"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_action_multiset_mismatch():
    # 3 actions expected → ceil(3/2)=2 POV beats. Supply 2 POV beats with
    # per-beat counts ≤2 so rules 4 and 5 pass; only rule 6 (multiset) fires.
    raw = _valid_storyboard([
        _pov("beat2", ["engine_status", "engine_status"]),
        _pov("beat3", ["gear_position"]),
    ])
    with pytest.raises(StoryboardValidationError, match="coverage mismatch"):
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


def test_rejects_pov_beat_without_keyframe_prompt():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"], keyframe_prompt="")])
    with pytest.raises(StoryboardValidationError, match="keyframe_prompt"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_keyframe_prompt_over_limit():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"],
                                  keyframe_prompt="x" * 341)])
    with pytest.raises(StoryboardValidationError, match="340"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_rejects_exterior_beat_with_keyframe_prompt():
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    raw["beats"][0]["keyframe_prompt"] = "should not be here"
    with pytest.raises(StoryboardValidationError, match="exterior.*keyframe"):
        load_storyboard(raw, expected_actions=["engine_status"])


def test_expected_actions_optional_for_schema_only_check():
    """When expected_actions is None, rules 4 and 6 are skipped so the
    generator's internal validation can reuse this function before it
    knows the full action list."""
    raw = _valid_storyboard([_pov("beat2", ["engine_status"])])
    sb = load_storyboard(raw, expected_actions=None)
    assert len(sb.beats) == 3
