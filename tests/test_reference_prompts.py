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
