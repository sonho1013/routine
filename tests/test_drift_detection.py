"""drift_detection 纯函数单测 — spec §7.3 / §7.4"""
from unittest.mock import MagicMock

import pytest

from engine.drift_detection import (
    compute_raw_value_stats,
    cosine_distance,
    is_drifted,
)


# ── is_drifted (numeric_mean_diff) ──

def _rule_engine_with(signal: str, rule: dict):
    m = MagicMock()
    m.get_rule.side_effect = lambda s: rule if s == signal else None
    return m


def test_is_drifted_numeric_within_threshold():
    re = _rule_engine_with("hvac_temp_target", {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    })
    old = {"type": "numeric", "mean": 22.0}
    new = {"type": "numeric", "mean": 23.0}
    assert is_drifted("hvac_temp_target", old, new, re) is False


def test_is_drifted_numeric_exceeds_threshold():
    re = _rule_engine_with("hvac_temp_target", {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    })
    old = {"type": "numeric", "mean": 22.0}
    new = {"type": "numeric", "mean": 24.0}  # +2.0 > 1.5
    assert is_drifted("hvac_temp_target", old, new, re) is True


def test_is_drifted_numeric_boundary_equal_not_drifted():
    """严格大于阈值才算漂移"""
    re = _rule_engine_with("hvac_temp_target", {
        "drift_metric": "numeric_mean_diff",
        "drift_threshold": 1.5,
    })
    old = {"type": "numeric", "mean": 22.0}
    new = {"type": "numeric", "mean": 23.5}
    assert is_drifted("hvac_temp_target", old, new, re) is False


# ── is_drifted (dominant_value_change) ──

def test_is_drifted_categorical_same_dominant():
    re = _rule_engine_with("media_source", {
        "drift_metric": "dominant_value_change",
    })
    old = {"type": "categorical", "dominant_value": "music"}
    new = {"type": "categorical", "dominant_value": "music"}
    assert is_drifted("media_source", old, new, re) is False


def test_is_drifted_categorical_dominant_changed():
    re = _rule_engine_with("media_source", {
        "drift_metric": "dominant_value_change",
    })
    old = {"type": "categorical", "dominant_value": "music"}
    new = {"type": "categorical", "dominant_value": "podcast"}
    assert is_drifted("media_source", old, new, re) is True


# ── is_drifted (embedding_fallback) ──

def test_is_drifted_embedding_fallback_similar():
    """cos距离 < 0.15 认为未漂移"""
    re = MagicMock()
    re.get_rule.return_value = None  # 无规则 → fallback
    old = {"habit_text": "turned on seat heating"}
    new = {"habit_text": "turned on seat heating"}
    fake_embed = MagicMock(return_value=[1.0, 0.0, 0.0])
    assert is_drifted("unknown", old, new, re, embed_fn=fake_embed) is False


def test_is_drifted_embedding_fallback_divergent():
    re = MagicMock()
    re.get_rule.return_value = None
    old = {"habit_text": "first"}
    new = {"habit_text": "second"}

    # 两次 embed 输出几乎正交
    def fake_embed(text):
        return [1.0, 0.0, 0.0] if text == "first" else [0.0, 1.0, 0.0]

    assert is_drifted("unknown", old, new, re, embed_fn=fake_embed) is True


# ── cosine_distance ──

def test_cosine_distance_identical_is_zero():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_cosine_distance_orthogonal_is_one():
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-9)


def test_cosine_distance_opposite_is_two():
    assert cosine_distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, abs=1e-9)


def test_cosine_distance_zero_vector_returns_one():
    """零向量没有方向 → 约定返回 1.0 (maximally distant)"""
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ── compute_raw_value_stats (numeric) ──

def test_compute_raw_value_stats_numeric():
    from panoramix_core.models.fact import Fact
    from panoramix_core.models.fact_enums import FactType

    facts = []
    for val in [22.0, 23.0, 21.5, 22.5]:
        f = MagicMock()
        f.json_metadata = f'{{"signal": "hvac_temp_target", "raw_value": {val}}}'
        facts.append(f)

    stats = compute_raw_value_stats(facts, signal_category="numeric")
    assert stats["type"] == "numeric"
    assert stats["mean"] == pytest.approx(22.25)
    assert stats["min"] == 21.5
    assert stats["max"] == 23.0
    assert stats["count"] == 4


def test_compute_raw_value_stats_numeric_single_value_zero_std():
    facts = [MagicMock()]
    facts[0].json_metadata = '{"signal": "s", "raw_value": 22.0}'
    stats = compute_raw_value_stats(facts, signal_category="numeric")
    assert stats["std"] == 0.0
    assert stats["count"] == 1


# ── compute_raw_value_stats (categorical) ──

def test_compute_raw_value_stats_categorical():
    facts = []
    for val in ["music", "music", "music", "podcast"]:
        f = MagicMock()
        f.json_metadata = f'{{"signal": "media_source", "raw_value": "{val}"}}'
        facts.append(f)

    stats = compute_raw_value_stats(facts, signal_category="categorical")
    assert stats["type"] == "categorical"
    assert stats["dominant_value"] == "music"
    assert stats["value_counts"] == {"music": 3, "podcast": 1}
    assert stats["count"] == 4


def test_compute_raw_value_stats_unknown_category_raises():
    with pytest.raises(ValueError, match="unknown signal_category"):
        compute_raw_value_stats([], signal_category="mystery")
