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
