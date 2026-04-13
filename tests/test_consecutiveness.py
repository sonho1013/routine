"""
连续性验证逻辑单元测试

覆盖:
  - verify_cluster_consecutiveness(): 通过/不通过/不足数据
  - _get_dominant_signal / _get_dominant_context / _find_candidates
  - make_cluster_filter(): 与 HabitsDetector 的集成接口
  - 真实 mockup 数据集成验证
"""
import os
import sys
import json
import pytest
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.consecutiveness import (
    verify_cluster_consecutiveness,
    make_cluster_filter,
    ClusterConsecutiveness,
    SignalConsecutiveness,
    _get_dominant_signal,
    _get_dominant_context,
    _find_candidates,
    _extract_signal_name,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _fact(text, signal, value, day, hour=8,
          time_bucket="early_morning", vehicle_state="engine_started",
          geofence=None, weekday=True):
    """构造带完整 metadata 的 Fact"""
    ts = datetime(2025, 10, 5 + day, hour, 15, 0)
    return Fact(
        text=text,
        type=FactType.PREF,
        durability=FactDurability.LONG_TERM,
        time_stamp=ts,
        source=FactSources.SIGNAL,
        context=StructuredContext(
            time_bucket=time_bucket,
            hour=hour,
            weekday=weekday,
            vehicle_state=vehicle_state,
            geofence=geofence,
        ),
        json_metadata=json.dumps({"signal": signal, "raw_value": value}),
    )


# ═══════════════════════════════════════════════════
# 核心场景
# ═══════════════════════════════════════════════════

class TestConsecutivePass:
    """5 天连续相同行为 → 全部在聚类中 → 通过"""

    def test_five_consecutive(self):
        # Day1-5 每天早晨设 AC 22°C
        facts = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in range(1, 6)]
        all_facts = facts[:]  # 全量 = 聚类本身

        result = verify_cluster_consecutiveness(facts, all_facts, required_consecutive=5)
        assert result.is_consecutive is True
        assert result.cluster_size == 5
        assert result.dominant_signal == "hvac_temp_target"
        assert result.signal_results[0].recent_in_cluster == 5

    def test_more_than_required(self):
        """7 天数据，只要最近 5 天连续即可"""
        facts = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in range(1, 8)]
        result = verify_cluster_consecutiveness(facts, facts, required_consecutive=5)
        assert result.is_consecutive is True

    def test_with_lower_threshold(self):
        """3 天数据，required=3 → 通过"""
        facts = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in range(1, 4)]
        result = verify_cluster_consecutiveness(facts, facts, required_consecutive=3)
        assert result.is_consecutive is True


class TestConsecutiveFail:
    """最近 N 次中有非聚类成员 → 不通过"""

    def test_one_interruption(self):
        """Day1,2,4,5 设 22°C (在聚类中), Day3 设 24°C (不在聚类中)"""
        cluster = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in [1, 2, 4, 5]]
        outlier = _fact("set AC to 24", "hvac_temp_target", 24, day=3)
        all_facts = cluster + [outlier]

        result = verify_cluster_consecutiveness(cluster, all_facts, required_consecutive=5)
        assert result.is_consecutive is False
        assert result.signal_results[0].recent_in_cluster == 4
        assert result.signal_results[0].total_candidates == 5

    def test_recent_break(self):
        """Day1-4 在聚类，Day5 中断 → 最近 5 中只有 4 个在聚类"""
        cluster = [_fact("nav to work", "nav_destination", "work", day=d) for d in range(1, 5)]
        outlier = _fact("nav to supermarket", "nav_destination", "supermarket", day=5)
        all_facts = cluster + [outlier]

        result = verify_cluster_consecutiveness(cluster, all_facts, required_consecutive=5)
        assert result.is_consecutive is False

    def test_old_consecutive_new_break(self):
        """Day1-5 连续，但 Day6-7 中断 → 最近 5 条(Day3-7)中只有 3 个在聚类"""
        cluster = [_fact("seat heating 2", "seat_heating", 2, day=d) for d in range(1, 6)]
        outliers = [_fact("seat heating 3", "seat_heating", 3, day=d) for d in [6, 7]]
        all_facts = cluster + outliers

        result = verify_cluster_consecutiveness(cluster, all_facts, required_consecutive=5)
        assert result.is_consecutive is False
        assert result.signal_results[0].recent_in_cluster == 3  # Day3,4,5 in cluster


class TestInsufficientData:
    """候选数不足 required → 不通过"""

    def test_three_facts_need_five(self):
        facts = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in range(1, 4)]
        result = verify_cluster_consecutiveness(facts, facts, required_consecutive=5)
        assert result.is_consecutive is False
        assert "insufficient" in result.signal_results[0].detail

    def test_empty_cluster(self):
        result = verify_cluster_consecutiveness([], [], required_consecutive=5)
        assert result.is_consecutive is False
        assert result.cluster_size == 0


# ═══════════════════════════════════════════════════
# 上下文隔离
# ═══════════════════════════════════════════════════

class TestContextIsolation:
    """不同上下文的相同信号不应互相干扰"""

    def test_different_time_bucket(self):
        """早晨 AC 22°C (5次) + 下午 AC 24°C (2次，不在聚类) → 不混淆"""
        morning_cluster = [
            _fact("set AC to 22", "hvac_temp_target", 22, day=d,
                  time_bucket="early_morning")
            for d in range(1, 6)
        ]
        afternoon = [
            _fact("set AC to 24", "hvac_temp_target", 24, day=d,
                  hour=15, time_bucket="afternoon")
            for d in [1, 2]
        ]
        all_facts = morning_cluster + afternoon

        result = verify_cluster_consecutiveness(
            morning_cluster, all_facts, required_consecutive=5
        )
        # 下午的 facts 不在候选集中，不影响早晨的连续性
        assert result.is_consecutive is True
        assert result.signal_results[0].total_candidates == 5

    def test_different_vehicle_state(self):
        """启动时 AC 和停车时关 AC 是不同行为，不互相干扰"""
        start_cluster = [
            _fact("set AC to 22", "hvac_temp_target", 22, day=d,
                  vehicle_state="engine_started")
            for d in range(1, 6)
        ]
        parked = [
            _fact("turned off AC", "hvac_power", "off", day=d,
                  hour=18, time_bucket="evening", vehicle_state="parked")
            for d in range(1, 6)
        ]
        all_facts = start_cluster + parked

        result = verify_cluster_consecutiveness(
            start_cluster, all_facts, required_consecutive=5
        )
        assert result.is_consecutive is True

    def test_geofence_filtering(self):
        """聚类有特定围栏 → 候选也必须匹配"""
        home_cluster = [
            _fact("close windows", "window_position", 0, day=d,
                  hour=18, time_bucket="evening", vehicle_state="parked",
                  geofence="home")
            for d in range(1, 6)
        ]
        work_close = [
            _fact("close windows", "window_position", 0, day=d,
                  hour=9, time_bucket="morning", vehicle_state="parked",
                  geofence="workplace")
            for d in [1, 2]
        ]
        all_facts = home_cluster + work_close

        result = verify_cluster_consecutiveness(
            home_cluster, all_facts, required_consecutive=5
        )
        assert result.is_consecutive is True
        assert result.signal_results[0].total_candidates == 5

    def test_no_geofence_allows_any(self):
        """聚类无围栏 → 候选集不按围栏过滤"""
        cluster = [
            _fact("eco mode", "drive_mode", "eco", day=d, geofence=None)
            for d in range(1, 6)
        ]
        result = verify_cluster_consecutiveness(cluster, cluster, required_consecutive=5)
        assert result.is_consecutive is True


# ═══════════════════════════════════════════════════
# 内部函数
# ═══════════════════════════════════════════════════

class TestDominantSignal:
    def test_single_signal(self):
        facts = [_fact("x", "hvac_temp_target", 22, day=d) for d in range(1, 4)]
        assert _get_dominant_signal(facts) == "hvac_temp_target"

    def test_mixed_signals(self):
        facts = [
            _fact("x", "hvac_temp_target", 22, day=1),
            _fact("y", "hvac_temp_target", 22, day=2),
            _fact("z", "media_source", "podcast", day=3),
        ]
        assert _get_dominant_signal(facts) == "hvac_temp_target"

    def test_no_metadata(self):
        f = Fact(text="test", type=FactType.PREF, time_stamp=datetime.now())
        assert _extract_signal_name(f) == "unknown"


class TestDominantContext:
    def test_uniform(self):
        facts = [
            _fact("x", "sig", 1, day=d, time_bucket="morning", vehicle_state="parked")
            for d in range(1, 4)
        ]
        ctx = _get_dominant_context(facts)
        assert ctx["time_bucket"] == "morning"
        assert ctx["vehicle_state"] == "parked"
        assert ctx["geofence"] is None

    def test_majority_wins(self):
        facts = [
            _fact("x", "sig", 1, day=1, time_bucket="morning"),
            _fact("x", "sig", 1, day=2, time_bucket="morning"),
            _fact("x", "sig", 1, day=3, time_bucket="afternoon"),
        ]
        ctx = _get_dominant_context(facts)
        assert ctx["time_bucket"] == "morning"


class TestFindCandidates:
    def test_filters_by_signal(self):
        pool = [
            _fact("x", "hvac_temp_target", 22, day=1),
            _fact("y", "seat_heating", 2, day=1),
            _fact("z", "hvac_temp_target", 24, day=2),
        ]
        ctx = {"time_bucket": "early_morning", "vehicle_state": "engine_started", "geofence": None}
        cands = _find_candidates(pool, "hvac_temp_target", ctx)
        assert len(cands) == 2

    def test_filters_by_context(self):
        pool = [
            _fact("x", "hvac_temp_target", 22, day=1, time_bucket="early_morning"),
            _fact("y", "hvac_temp_target", 24, day=2, time_bucket="afternoon", hour=15),
        ]
        ctx = {"time_bucket": "early_morning", "vehicle_state": "engine_started", "geofence": None}
        cands = _find_candidates(pool, "hvac_temp_target", ctx)
        assert len(cands) == 1


# ═══════════════════════════════════════════════════
# make_cluster_filter 接口
# ═══════════════════════════════════════════════════

class TestMakeClusterFilter:
    def test_pass(self):
        facts = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in range(1, 6)]
        fn = make_cluster_filter(required_consecutive=5)
        assert fn(facts, facts) is True

    def test_fail(self):
        cluster = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in [1, 2, 4, 5]]
        outlier = _fact("set AC to 24", "hvac_temp_target", 24, day=3)
        fn = make_cluster_filter(required_consecutive=5)
        assert fn(cluster, cluster + [outlier]) is False


# ═══════════════════════════════════════════════════
# 真实 mockup 数据集成
# ═══════════════════════════════════════════════════

class TestMockupIntegration:
    """使用真实 mockup 数据验证连续性"""

    @pytest.fixture
    def mockup_facts(self):
        from engine.signal_to_fact import signals_to_facts
        data_file = os.path.join(os.path.dirname(__file__), "step1_mockup_data.json")
        if not os.path.exists(data_file):
            pytest.skip("mockup data not found")
        with open(data_file) as f:
            data = json.load(f)
        all_facts = []
        for scene_events in data["scenes"].values():
            for event in scene_events:
                all_facts.extend(signals_to_facts(event))
        return all_facts

    def test_morning_ac_consecutive(self, mockup_facts):
        """早晨 hvac_temp_target 5 天连续 → 通过"""
        morning_ac = [
            f for f in mockup_facts
            if _extract_signal_name(f) == "hvac_temp_target"
            and f.context.time_bucket == "early_morning"
        ]
        assert len(morning_ac) == 5  # 5 days of morning commute
        result = verify_cluster_consecutiveness(morning_ac, mockup_facts, required_consecutive=5)
        assert result.is_consecutive is True

    def test_arriving_home_engine_off(self, mockup_facts):
        """到家 engine_status off 5 天连续 → 通过"""
        home_off = [
            f for f in mockup_facts
            if _extract_signal_name(f) == "engine_status"
            and f.context.vehicle_state == "parked"
            and f.context.time_bucket == "evening"
        ]
        assert len(home_off) == 5
        result = verify_cluster_consecutiveness(home_off, mockup_facts, required_consecutive=5)
        assert result.is_consecutive is True

    def test_toll_window_consecutive(self, mockup_facts):
        """收费站 window_position 5 天连续 → 通过"""
        toll_window = [
            f for f in mockup_facts
            if _extract_signal_name(f) == "window_position"
            and f.context.vehicle_state == "crawling"
        ]
        assert len(toll_window) == 5
        result = verify_cluster_consecutiveness(toll_window, mockup_facts, required_consecutive=5)
        assert result.is_consecutive is True

    def test_noise_insufficient(self, mockup_facts):
        """噪声信号（不足 5 次同上下文）→ insufficient"""
        # 找一个只出现 1-2 次的信号+上下文组合
        noise_nav = [
            f for f in mockup_facts
            if _extract_signal_name(f) == "nav_destination"
            and f.context.time_bucket == "afternoon"
        ]
        if len(noise_nav) >= 2:
            result = verify_cluster_consecutiveness(
                noise_nav, mockup_facts, required_consecutive=5
            )
            assert result.is_consecutive is False


# ═══════════════════════════════════════════════════
# _AGREEMENT_THRESHOLD 行为
# ═══════════════════════════════════════════════════

class TestAgreementThreshold:
    """聚类内 time_bucket 一致性 <80% 时跳过 time_bucket 过滤"""

    def test_low_agreement_skips_time_filter(self):
        """
        5 个 facts: 3 个 early_morning + 2 个 morning → agreement = 0.6 < 0.8
        候选集应包含两种 time_bucket 的 facts
        """
        cluster = [
            _fact("open window", "window_position", 100, day=d,
                  time_bucket="early_morning", vehicle_state="crawling")
            for d in range(1, 4)
        ] + [
            _fact("open window", "window_position", 100, day=d,
                  time_bucket="morning", vehicle_state="crawling")
            for d in range(4, 6)
        ]
        # 额外添加一个不同 time_bucket 但同 signal+vehicle_state 的 fact（不在聚类中）
        extra = _fact("open window", "window_position", 100, day=6,
                      time_bucket="midday", vehicle_state="crawling")
        all_facts = cluster + [extra]

        result = verify_cluster_consecutiveness(cluster, all_facts, required_consecutive=5)
        # 因为 agreement < 0.8, time_bucket 不过滤,
        # 所以 extra (midday) 也会进入候选集, 最近 5 条可能含 extra
        # Day1-5 cluster + Day6 extra → 最近 5 = Day2-6 → Day6 不在聚类 → FAIL
        assert result.is_consecutive is False
        # 候选总数应为 6（不限 time_bucket）
        assert result.signal_results[0].total_candidates == 6

    def test_high_agreement_filters_time(self):
        """
        5 个 facts: 5 个都是 early_morning → agreement = 1.0 ≥ 0.8
        time_bucket 过滤生效 → 不同 time_bucket 的 facts 不进入候选集
        """
        cluster = [
            _fact("set AC to 22", "hvac_temp_target", 22, day=d,
                  time_bucket="early_morning")
            for d in range(1, 6)
        ]
        extra = _fact("set AC to 24", "hvac_temp_target", 24, day=6,
                      time_bucket="afternoon", hour=15)
        all_facts = cluster + [extra]

        result = verify_cluster_consecutiveness(cluster, all_facts, required_consecutive=5)
        assert result.is_consecutive is True
        # time_bucket 过滤生效, afternoon 的不算候选
        assert result.signal_results[0].total_candidates == 5

    def test_exactly_80_percent_filters(self):
        """agreement = 0.8 正好达到阈值 → 过滤生效"""
        # 5 个 facts: 4 个 morning + 1 个 afternoon → agreement = 0.8
        cluster = [
            _fact("eco mode", "drive_mode", "eco", day=d, time_bucket="morning")
            for d in range(1, 5)
        ] + [
            _fact("eco mode", "drive_mode", "eco", day=5, time_bucket="afternoon", hour=15)
        ]
        extra = _fact("eco mode", "drive_mode", "eco", day=6, time_bucket="evening", hour=19)
        all_facts = cluster + [extra]

        ctx = _get_dominant_context(cluster)
        assert ctx["time_bucket"] == "morning"
        assert ctx["time_bucket_agreement"] == pytest.approx(0.8)

        cands = _find_candidates(all_facts, "drive_mode", ctx)
        # 过滤生效：只留 morning 的 4 个
        assert len(cands) == 4


# ═══════════════════════════════════════════════════
# ClusterConsecutiveness.summary 属性
# ═══════════════════════════════════════════════════

class TestSummary:
    def test_pass_summary(self):
        facts = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in range(1, 6)]
        result = verify_cluster_consecutiveness(facts, facts, required_consecutive=5)
        s = result.summary
        assert "✓" in s
        assert "hvac_temp_target" in s

    def test_fail_summary(self):
        cluster = [_fact("set AC to 22", "hvac_temp_target", 22, day=d) for d in [1, 2, 4, 5]]
        outlier = _fact("set AC to 24", "hvac_temp_target", 24, day=3)
        result = verify_cluster_consecutiveness(cluster, cluster + [outlier], required_consecutive=5)
        s = result.summary
        assert "✗" in s
        assert "4/5" in s

    def test_empty_summary(self):
        result = verify_cluster_consecutiveness([], [], required_consecutive=5)
        assert result.summary == ""


# ═══════════════════════════════════════════════════
# _extract_signal_name 边界
# ═══════════════════════════════════════════════════

class TestExtractSignalName:
    def test_normal(self):
        f = _fact("x", "hvac_temp_target", 22, day=1)
        assert _extract_signal_name(f) == "hvac_temp_target"

    def test_no_metadata(self):
        f = Fact(text="test", type=FactType.PREF, time_stamp=datetime.now())
        assert _extract_signal_name(f) == "unknown"

    def test_empty_json_object(self):
        """json_metadata = '{}' → 无 signal key → unknown"""
        f = Fact(text="test", type=FactType.PREF, time_stamp=datetime.now(),
                 json_metadata=json.dumps({}))
        assert _extract_signal_name(f) == "unknown"

    def test_missing_signal_key(self):
        f = Fact(text="test", type=FactType.PREF, time_stamp=datetime.now(),
                 json_metadata=json.dumps({"other": "data"}))
        assert _extract_signal_name(f) == "unknown"


# ═══════════════════════════════════════════════════
# 多信号聚类
# ═══════════════════════════════════════════════════

class TestMultiSignalCluster:
    """聚类包含多种信号时，取主导信号验证"""

    def test_dominant_signal_wins(self):
        """3 个 hvac_temp + 2 个 seat_heating → 主导信号 = hvac_temp"""
        facts = [
            _fact("set AC to 22", "hvac_temp_target", 22, day=d)
            for d in range(1, 4)
        ] + [
            _fact("seat heating 2", "seat_heating", 2, day=d)
            for d in range(4, 6)
        ]
        result = verify_cluster_consecutiveness(facts, facts, required_consecutive=5)
        assert result.dominant_signal == "hvac_temp_target"
