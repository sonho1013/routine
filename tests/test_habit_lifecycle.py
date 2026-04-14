"""HabitLifecycleManager.classify_candidate 集成测试 — spec §7.2"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from engine.habit_lifecycle import (
    ClassificationResult,
    ModifyDrift,
    NewPending,
    Reinforce,
    classify_candidate,
    compute_structural_key,
)
from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard


def _make_habit(
    signal_name: str = "hvac_temp_target",
    mean: float = 22.0,
    time_bucket: str = "early_morning",
    vehicle_state: str = "engine_started",
    geofence: str = "home",
) -> Habit:
    return Habit(
        username="tester",
        batch_id=2,
        text=f"set temperature to {mean}",
        signal_category="numeric",
        signal_name=signal_name,
        structural_key="placeholder",
        context_time_bucket=time_bucket,
        context_vehicle_state=vehicle_state,
        context_geofence=geofence,
        context_weekday=1,
        raw_value_stats={
            "type": "numeric", "mean": mean, "std": 0.5,
            "min": mean - 1, "max": mean + 1, "count": 10,
            "habit_text": f"set temperature to {mean}",
        },
        member_fact_ids=[],
    )


def _make_accepted_card(
    structural_key: str,
    frozen_habits: list[dict],
) -> SceneCard:
    return SceneCard(
        username="tester",
        status="accepted",
        structural_key=structural_key,
        display_name="Morning Home Routine",
        frozen_content_snapshot={
            "snapshot_batch_id": 1,
            "snapshot_at": "2026-04-08T10:00:00",
            "habits": frozen_habits,
            "dominant_context": {
                "time_bucket": "early_morning",
                "vehicle_state": "engine_started",
                "geofence": "home",
                "weekday": True,
            },
        },
        first_seen_batch_id=1,
        last_reinforced_batch_id=1,
        created_at=datetime.fromisoformat("2026-04-08T10:00:00"),
        accepted_at=datetime.fromisoformat("2026-04-08T10:05:00"),
    )


@pytest.fixture
def rule_engine_temp():
    m = MagicMock()
    m.get_rule.side_effect = lambda s: {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    } if s == "hvac_temp_target" else None
    return m


# ── NewPending：无匹配 accepted ──

def test_classify_no_accepted_returns_new_pending(rule_engine_temp):
    candidate = [_make_habit(mean=22.0)]
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, NewPending)
    assert result.structural_key == compute_structural_key(candidate)


def test_classify_structural_key_mismatch_returns_new_pending(rule_engine_temp):
    candidate = [_make_habit(geofence="workplace")]  # 不同 geofence
    accepted = _make_accepted_card(
        structural_key="different_key",
        frozen_habits=[{
            "signal": "hvac_temp_target",
            "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[accepted],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, NewPending)


# ── Reinforce：匹配且无 drift ──

def test_classify_reinforce_when_mean_within_threshold(rule_engine_temp):
    candidate = [_make_habit(mean=22.5)]  # +0.5 < 1.5
    key = compute_structural_key(candidate)
    accepted = _make_accepted_card(
        structural_key=key,
        frozen_habits=[{
            "signal": "hvac_temp_target",
            "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[accepted],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, Reinforce)
    assert result.card_id == accepted.card_id


# ── ModifyDrift：匹配但 drift ──

def test_classify_modify_drift_when_mean_exceeds_threshold(rule_engine_temp):
    candidate = [_make_habit(mean=24.0)]  # +2.0 > 1.5
    key = compute_structural_key(candidate)
    accepted = _make_accepted_card(
        structural_key=key,
        frozen_habits=[{
            "signal": "hvac_temp_target",
            "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[accepted],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, ModifyDrift)
    assert result.card_id == accepted.card_id
    assert "hvac_temp_target" in result.drifted_signals


# ── 多张 accepted 匹配同一 key → NewPending + warning (DCM-3) ──

def test_classify_multiple_accepted_matches_logged_as_new_pending(
    rule_engine_temp, caplog
):
    candidate = [_make_habit(mean=22.0)]
    key = compute_structural_key(candidate)
    a1 = _make_accepted_card(
        structural_key=key, frozen_habits=[{
            "signal": "hvac_temp_target", "habit_text": "t",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    a2 = _make_accepted_card(
        structural_key=key, frozen_habits=[{
            "signal": "hvac_temp_target", "habit_text": "t",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    with caplog.at_level("WARNING"):
        result = classify_candidate(
            candidate_habits=candidate,
            accepted_cards=[a1, a2],
            signal_rule_engine=rule_engine_temp,
        )
    assert isinstance(result, NewPending)
    assert "multiple accepted" in caplog.text.lower() or "DCM-3" in caplog.text


# ── 不相关的 accepted 卡被忽略 ──

def test_classify_ignores_unrelated_accepted(rule_engine_temp):
    candidate = [_make_habit(mean=22.0)]
    key = compute_structural_key(candidate)
    matching = _make_accepted_card(
        structural_key=key,
        frozen_habits=[{
            "signal": "hvac_temp_target", "habit_text": "set temperature to 22.0",
            "raw_value_stats": {"type": "numeric", "mean": 22.0},
        }],
    )
    unrelated = _make_accepted_card(
        structural_key="different",
        frozen_habits=[{
            "signal": "media_source", "habit_text": "music",
            "raw_value_stats": {"type": "categorical", "dominant_value": "music"},
        }],
    )
    result = classify_candidate(
        candidate_habits=candidate,
        accepted_cards=[unrelated, matching],
        signal_rule_engine=rule_engine_temp,
    )
    assert isinstance(result, Reinforce)
    assert result.card_id == matching.card_id
