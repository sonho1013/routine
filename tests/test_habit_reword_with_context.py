"""Verify reword_cluster passes structured context into the LLM prompt.

Pre-fix the prompt only saw fact texts, so habit names lost the situation
(time-of-day, geofence, vehicle state). This test pins the wiring contract:
when a context is passed, the prompt sent to the LLM must include the relevant
fields, and when context is None / mostly unknown, the prompt must say so.
"""
from unittest.mock import MagicMock

from panoramix_core.clustering.habits_detector import (
    HabitsDetector, _format_context_for_prompt,
)
from panoramix_core.models.fact import StructuredContext


def _detector_with_mock_llm(reply: str = "morning office gate driver window habit,0.9"):
    d = HabitsDetector(llm_client=None)        # init reads real prompt template
    d.llm = MagicMock()
    d.llm.invoke.return_value = reply
    return d


# ── _format_context_for_prompt ──

def test_format_context_skips_unknown_fields():
    ctx = StructuredContext(
        time_bucket="evening",
        vehicle_state="unknown",       # filtered out
        geofence=None,                 # filtered out
        wiper_state="unknown",         # filtered out
    )
    out = _format_context_for_prompt(ctx)
    assert "time_bucket: evening" in out
    assert "vehicle_state" not in out
    assert "geofence" not in out
    assert "wiper_state" not in out


def test_format_context_renders_relevant_fields():
    ctx = StructuredContext(
        time_bucket="evening",
        hour=18,
        weekday=True,
        vehicle_state="crawling",
        geofence="office_gate_01",
        poi_type="site_entrance",
        window_state="open",
    )
    out = _format_context_for_prompt(ctx)
    assert "time_bucket: evening" in out
    assert "hour: 18" in out
    assert "weekday: workday" in out
    assert "vehicle_state: crawling" in out
    assert "geofence: office_gate_01" in out
    assert "poi_type: site_entrance" in out
    assert "window_state: open" in out


def test_format_context_handles_none():
    assert _format_context_for_prompt(None) == "(no useful context provided)"


def test_format_context_handles_all_unknown():
    ctx = StructuredContext()  # all defaults are "unknown" / None
    assert _format_context_for_prompt(ctx) == "(no useful context provided)"


def test_format_weekday_false_is_weekend():
    ctx = StructuredContext(time_bucket="midday", weekday=False)
    out = _format_context_for_prompt(ctx)
    assert "weekday: weekend" in out


# ── reword_cluster passes context to LLM ──

def test_reword_cluster_includes_context_in_prompt():
    d = _detector_with_mock_llm()
    ctx = StructuredContext(
        time_bucket="evening",
        weekday=True,
        geofence="office_gate_01",
        poi_type="site_entrance",
        vehicle_state="crawling",
    )
    d.reword_cluster(["driver window opened"] * 5, context=ctx)

    prompt_sent = d.llm.invoke.call_args.args[0]
    assert "geofence: office_gate_01" in prompt_sent
    assert "poi_type: site_entrance" in prompt_sent
    assert "time_bucket: evening" in prompt_sent
    assert "weekday: workday" in prompt_sent
    assert "driver window opened" in prompt_sent


def test_reword_cluster_no_context_uses_placeholder():
    d = _detector_with_mock_llm()
    d.reword_cluster(["climate set to 23"] * 3, context=None)
    prompt_sent = d.llm.invoke.call_args.args[0]
    assert "(no useful context provided)" in prompt_sent
