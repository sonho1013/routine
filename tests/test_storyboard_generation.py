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


def test_raises_after_last_validation_failure():
    bad = json.dumps({"scene_id": "x", "scene_summary": "s",
                      "keyframes": [], "shots": []})
    with pytest.raises(StoryboardValidationError):
        generate_storyboard(
            openai_client=_fake_openai([bad, bad]),
            scene={"id": "x", "llm_user_prompt": "hello"},
            cinematic_actions=[],
            max_retries=1,
        )


def test_strips_markdown_fences_from_llm_output():
    fenced = "```json\n" + VALID_JSON + "\n```"
    sb = generate_storyboard(
        openai_client=_fake_openai([fenced]),
        scene={"id": "day1_morning_commute", "llm_user_prompt": "x"},
        cinematic_actions=[],
    )
    assert sb.scene_id == "day1_morning_commute"
