"""compute_structural_key() 稳定性与边界测试 — spec §3.1"""
import pytest

from engine.habit_lifecycle import compute_structural_key
from panoramix_core.models.habit import Habit


def _make_habit(
    signal_name: str = "hvac_temp_target",
    time_bucket: str = "early_morning",
    vehicle_state: str = "engine_started",
    geofence: str = "home",
    raw_value_stats: dict | None = None,
) -> Habit:
    return Habit(
        username="tester",
        batch_id=1,
        text=f"habit text for {signal_name}",
        signal_category="numeric",
        signal_name=signal_name,
        structural_key="placeholder",
        context_time_bucket=time_bucket,
        context_vehicle_state=vehicle_state,
        context_geofence=geofence,
        context_weekday=1,
        raw_value_stats=raw_value_stats or {"type": "numeric", "mean": 22.0, "count": 5},
        member_fact_ids=[],
    )


def test_empty_habits_raises():
    with pytest.raises(ValueError):
        compute_structural_key([])


def test_single_habit_deterministic():
    h = _make_habit()
    k1 = compute_structural_key([h])
    k2 = compute_structural_key([h])
    assert k1 == k2
    assert len(k1) == 16  # SHA-256 hex[:16]


def test_order_independence():
    """同 cluster 不同顺序应产生相同 key"""
    h1 = _make_habit(signal_name="hvac_temp_target")
    h2 = _make_habit(signal_name="seat_heating")
    assert compute_structural_key([h1, h2]) == compute_structural_key([h2, h1])


def test_dominant_geofence_majority_wins():
    habits = [
        _make_habit(geofence="home"),
        _make_habit(geofence="home"),
        _make_habit(geofence="workplace"),
    ]
    k_home = compute_structural_key(habits)
    # 换成全 workplace → key 必须变
    habits_w = [_make_habit(geofence="workplace") for _ in range(3)]
    assert compute_structural_key(habits_w) != k_home


def test_dominant_tie_broken_alphabetically():
    """平票时按字典序取第一个，确保确定性"""
    h1 = _make_habit(geofence="home")
    h2 = _make_habit(geofence="workplace")
    k_tied_1 = compute_structural_key([h1, h2])
    k_tied_2 = compute_structural_key([h2, h1])
    assert k_tied_1 == k_tied_2  # 顺序无关
    # 再加一个 home 打破平票
    k_home_wins = compute_structural_key([h1, h2, _make_habit(geofence="home")])
    # "home" 原本是字典序第一，所以平票时也选 home；加到 home 胜出时 key 不变
    assert k_tied_1 == k_home_wins


def test_none_geofence_handled():
    habits = [_make_habit(geofence=None), _make_habit(geofence=None)]
    k = compute_structural_key(habits)
    assert len(k) == 16


def test_different_signal_set_different_key():
    h1 = _make_habit(signal_name="hvac_temp_target")
    h2 = _make_habit(signal_name="media_source")
    assert compute_structural_key([h1]) != compute_structural_key([h2])


def test_weekday_not_in_key():
    """Round 5: weekday 故意不参与 key，避免过度切分"""
    h_weekday = _make_habit()
    h_weekend = _make_habit()
    h_weekend.context_weekday = 0
    assert compute_structural_key([h_weekday]) == compute_structural_key([h_weekend])


def test_time_bucket_changes_key():
    h_morning = _make_habit(time_bucket="early_morning")
    h_noon = _make_habit(time_bucket="noon")
    assert compute_structural_key([h_morning]) != compute_structural_key([h_noon])


def test_vehicle_state_changes_key():
    h_started = _make_habit(vehicle_state="engine_started")
    h_off = _make_habit(vehicle_state="engine_off")
    assert compute_structural_key([h_started]) != compute_structural_key([h_off])
