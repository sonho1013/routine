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
