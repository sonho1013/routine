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
    bad = {
        "scene_id": "x",
        "scene_summary": "s",
        "keyframes": [
            {"id": "kf0", "role": "r", "prompt": "p"},
            {"id": "kf1", "role": "r", "prompt": "p"},
        ],
        "shots": [
            {"id": "shot1", "from_kf": "kf0", "to_kf": "kf_missing",
             "duration": "5", "narrative_role": "r", "motion_prompt": "m"},
        ],
    }
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
