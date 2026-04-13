"""
主动执行器单元测试

覆盖:
  ── parse_habit_actions ──
  - HVAC温度/开关、座椅加热、导航、媒体、车窗、驾驶模式、ACC、无钥匙、引擎
  - 无匹配文本 → 空 dict

  ── RecommendedAction / RecommendationResult ──
  - combined_confidence 计算
  - summary 格式
  - has_recommendations 属性

  ── ProactiveExecutor.recommend ──
  - 精确匹配上下文 → 返回推荐
  - 无 accepted habits → 空结果
  - 超过阈值 → 不推荐
  - 多 habit 按距离排序
  - top_k 限制
  - 相邻时段匹配（soft match）
  - geofence 匹配/不匹配

  ── HabitDemoEngine 集成 ──
  - accept_habit + get_recommendation 闭环
"""
import os
import sys
import json
import pytest
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.proactive_executor import (
    ProactiveExecutor,
    RecommendedAction,
    RecommendationResult,
    parse_habit_actions,
    _context_to_dict,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources
from panoramix_core.config import CONTEXT_MATCH_THRESHOLD


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _ctx(time_bucket="early_morning", vehicle_state="engine_started",
         geofence=None, weekday=True, hour=8):
    return StructuredContext(
        time_bucket=time_bucket, hour=hour, weekday=weekday,
        vehicle_state=vehicle_state, geofence=geofence,
    )


def _habit(text, time_bucket="early_morning", vehicle_state="engine_started",
           geofence=None, weekday=True, hour=8, accepted=True,
           clustering_confidence=0.8, scene_name="Morning Commute"):
    meta = {
        "clustering_confidence": clustering_confidence,
        "scene_name": scene_name,
    }
    return Fact(
        text=text,
        type=FactType.HABIT,
        durability=FactDurability.LONG_TERM,
        time_stamp=datetime.now(),
        source=FactSources.HABITS_DETECTOR,
        accepted=accepted,
        context=_ctx(time_bucket, vehicle_state, geofence, weekday, hour),
        json_metadata=json.dumps(meta),
    )


class FakeFactStore:
    """模拟 FactStore 用于单元测试"""

    def __init__(self, facts=None):
        self._facts = list(facts or [])

    def get_facts(self, username, types=None, **kwargs):
        result = self._facts
        if types:
            type_values = {t.value if hasattr(t, 'value') else t for t in types}
            result = [f for f in result if f.type.value in type_values]
        return result

    def store_facts(self, username, facts):
        self._facts.extend(facts)

    def delete_facts(self, username, fact_ids):
        id_set = set(fact_ids)
        self._facts = [f for f in self._facts if f.id not in id_set]


# ═══════════════════════════════════════════════════
# parse_habit_actions
# ═══════════════════════════════════════════════════

class TestParseHabitActions:
    def test_hvac_temp(self):
        r = parse_habit_actions("set cabin air conditioning temperature to 22 degrees")
        assert r["hvac_temp_target"] == 22

    def test_hvac_temp_variant(self):
        r = parse_habit_actions("set AC temperature to 24 degrees")
        assert r["hvac_temp_target"] == 24

    def test_hvac_power_off(self):
        r = parse_habit_actions("turned off cabin air conditioning")
        assert "hvac_power" in r

    def test_seat_heating(self):
        r = parse_habit_actions("turned on seat heating to level 2")
        assert r["seat_heating"] == 2

    def test_nav_destination(self):
        r = parse_habit_actions("started navigation to work")
        assert r["nav_destination"] == "work"

    def test_nav_route(self):
        r = parse_habit_actions("selected fastest route preference")
        assert "nav_route_pref" in r

    def test_media_podcast(self):
        r = parse_habit_actions("resumed listening to podcast Tech Daily")
        assert r["media_source"] == "tech daily"

    def test_media_volume(self):
        r = parse_habit_actions("adjusted media volume to 58 percent")
        assert r["media_volume"] == 58

    def test_media_off(self):
        r = parse_habit_actions("stopped all media playback")
        assert r["media_off"] is True

    def test_window_lower(self):
        r = parse_habit_actions("lowered driver window to 73 percent")
        assert r["window_position"] == 73

    def test_window_close(self):
        r = parse_habit_actions("closed all vehicle windows")
        assert r["window_position"] == 0

    def test_drive_mode_eco(self):
        r = parse_habit_actions("switched to eco driving mode")
        assert r["drive_mode"] == "eco"

    def test_drive_mode_sport(self):
        r = parse_habit_actions("switched to sport driving mode")
        assert r["drive_mode"] == "sport"

    def test_acc_distance(self):
        r = parse_habit_actions("set ACC following distance to medium")
        assert r["acc_distance"] == "medium"

    def test_keyless_entry(self):
        r = parse_habit_actions("disabled keyless proximity entry")
        assert "keyless_entry" in r

    def test_engine_off(self):
        r = parse_habit_actions("shut down the vehicle engine")
        assert r["engine_status"] == "off"

    def test_no_match(self):
        r = parse_habit_actions("some random text about nothing")
        assert r == {}

    def test_media_content(self):
        r = parse_habit_actions("selected media content FM 98.5")
        assert r["media_content_id"] == "fm 98.5"


# ═══════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════

class TestRecommendedAction:
    def test_combined_confidence(self):
        a = RecommendedAction(
            habit_id="1", habit_text="test",
            context_distance=0.1, match_confidence=0.9,
            clustering_confidence=0.8,
            scene_name="Test", parsed_actions={}, habit_context={},
        )
        assert a.combined_confidence == pytest.approx(0.72, abs=0.01)

    def test_combined_confidence_zero_distance(self):
        a = RecommendedAction(
            habit_id="1", habit_text="test",
            context_distance=0.0, match_confidence=1.0,
            clustering_confidence=0.85,
            scene_name=None, parsed_actions={}, habit_context={},
        )
        assert a.combined_confidence == 0.85


class TestRecommendationResult:
    def test_has_recommendations_false(self):
        r = RecommendationResult(current_context={})
        assert r.has_recommendations is False

    def test_has_recommendations_true(self):
        r = RecommendationResult(
            current_context={},
            actions=[RecommendedAction(
                habit_id="1", habit_text="test", context_distance=0.1,
                match_confidence=0.9, clustering_confidence=0.8,
                scene_name="X", parsed_actions={}, habit_context={},
            )],
        )
        assert r.has_recommendations is True

    def test_summary_no_match(self):
        r = RecommendationResult(current_context={}, candidates_checked=5)
        assert "No match" in r.summary
        assert "5" in r.summary

    def test_summary_with_match(self):
        r = RecommendationResult(
            current_context={},
            actions=[RecommendedAction(
                habit_id="1", habit_text="set AC to 22",
                context_distance=0.05, match_confidence=0.95,
                clustering_confidence=0.8, scene_name="Morning",
                parsed_actions={}, habit_context={},
            )],
        )
        assert "1 recommendations" in r.summary
        assert "set AC to 22" in r.summary


# ═══════════════════════════════════════════════════
# ProactiveExecutor.recommend
# ═══════════════════════════════════════════════════

class TestRecommendExactMatch:
    """完全匹配上下文 → 推荐"""

    def test_exact_match(self):
        h = _habit("set cabin air conditioning temperature to 22 degrees")
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store)

        ctx = _ctx()  # 与 habit 上下文完全相同
        result = executor.recommend(ctx, "user1")

        assert result.has_recommendations
        assert len(result.actions) == 1
        assert result.actions[0].habit_text == h.text
        # geofence=None → "unknown" 贡献 0.25*0.5=0.125
        assert result.actions[0].context_distance == pytest.approx(0.125, abs=0.001)
        assert result.actions[0].match_confidence == pytest.approx(0.875, abs=0.001)
        assert result.actions[0].parsed_actions["hvac_temp_target"] == 22
        assert result.actions[0].scene_name == "Morning Commute"

    def test_multiple_habits_sorted(self):
        """多个 habits 按距离排序"""
        h1 = _habit("set AC to 22", time_bucket="early_morning")
        h2 = _habit("eco mode", time_bucket="morning")  # 稍远
        store = FakeFactStore([h1, h2])
        executor = ProactiveExecutor(store, threshold=0.5)

        ctx = _ctx(time_bucket="early_morning")
        result = executor.recommend(ctx, "user1")

        assert len(result.actions) >= 1
        # h1 应排在前面 (距离更近)
        assert result.actions[0].context_distance <= result.actions[1].context_distance


class TestRecommendNoMatch:
    """无匹配场景"""

    def test_no_accepted_habits(self):
        h = _habit("set AC to 22", accepted=False)
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store)

        result = executor.recommend(_ctx(), "user1")
        assert not result.has_recommendations
        assert result.total_habits == 1
        assert result.candidates_checked == 0

    def test_no_habits_at_all(self):
        store = FakeFactStore([])
        executor = ProactiveExecutor(store)

        result = executor.recommend(_ctx(), "user1")
        assert not result.has_recommendations
        assert result.total_habits == 0

    def test_context_too_different(self):
        """完全不同的上下文 → 超过阈值"""
        h = _habit("set AC to 22",
                    time_bucket="early_morning", vehicle_state="engine_started",
                    weekday=True)
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store)

        # 完全不同的上下文
        ctx = _ctx(time_bucket="night", vehicle_state="parked", weekday=False)
        result = executor.recommend(ctx, "user1")
        assert not result.has_recommendations


class TestRecommendTopK:
    def test_top_k_limits(self):
        habits = [
            _habit(f"habit {i}", time_bucket="early_morning")
            for i in range(5)
        ]
        store = FakeFactStore(habits)
        executor = ProactiveExecutor(store)

        result = executor.recommend(_ctx(), "user1", top_k=2)
        assert len(result.actions) <= 2


class TestRecommendSoftMatch:
    """相邻时段/部分匹配"""

    def test_adjacent_time_bucket(self):
        """morning habit + early_morning context → 小距离"""
        h = _habit("eco mode", time_bucket="morning")
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store, threshold=0.5)

        ctx = _ctx(time_bucket="early_morning")
        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations
        # distance > 0 but < threshold
        assert 0 < result.actions[0].context_distance < 0.5

    def test_geofence_mismatch_increases_distance(self):
        """habit 有 geofence=home, 当前 geofence=workplace → 距离增大"""
        h = _habit("close windows", geofence="home",
                    time_bucket="evening", vehicle_state="parked")
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store, threshold=1.0)  # 放宽阈值看距离

        ctx_home = _ctx(time_bucket="evening", vehicle_state="parked", geofence="home")
        ctx_work = _ctx(time_bucket="evening", vehicle_state="parked", geofence="workplace")

        r_home = executor.recommend(ctx_home, "user1")
        r_work = executor.recommend(ctx_work, "user1")

        assert r_home.actions[0].context_distance < r_work.actions[0].context_distance

    def test_weekday_weekend_difference(self):
        """工作日 habit + 周末 context → 有距离但较小(权重0.10)"""
        h = _habit("eco mode", weekday=True)
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store, threshold=0.5)

        ctx = _ctx(weekday=False)
        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations
        # weekday 权重 0.10 + geofence=None 贡献 0.125 → 0.225
        assert result.actions[0].context_distance == pytest.approx(0.225, abs=0.01)


class TestRecommendMetadata:
    """验证 metadata 正确传递"""

    def test_clustering_confidence_passed(self):
        h = _habit("set AC to 22", clustering_confidence=0.75)
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store)

        result = executor.recommend(_ctx(), "user1")
        assert result.actions[0].clustering_confidence == 0.75

    def test_scene_name_passed(self):
        h = _habit("set AC to 22", scene_name="Weekday Morning Commute")
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store)

        result = executor.recommend(_ctx(), "user1")
        assert result.actions[0].scene_name == "Weekday Morning Commute"

    def test_no_metadata(self):
        h = Fact(
            text="set AC to 22", type=FactType.HABIT,
            durability=FactDurability.LONG_TERM,
            time_stamp=datetime.now(),
            source=FactSources.HABITS_DETECTOR,
            accepted=True,
            context=_ctx(),
        )
        store = FakeFactStore([h])
        executor = ProactiveExecutor(store)

        result = executor.recommend(_ctx(), "user1")
        assert result.actions[0].clustering_confidence == 0.0
        assert result.actions[0].scene_name is None


# ═══════════════════════════════════════════════════
# HabitDemoEngine 集成
# ═══════════════════════════════════════════════════

class TestEngineIntegration:
    """验证 accept_habit + get_recommendation 闭环"""

    def test_accept_and_recommend(self):
        """accept 后 recommend 能找到"""
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username="test-executor", llm_client=None)
        engine.reset()

        # 手动存入一个未 accepted 的 habit
        h = _habit("set cabin air conditioning temperature to 22 degrees",
                    accepted=False)
        engine.fact_store.store_facts(engine.username, [h])

        # 未 accept → 无推荐
        ctx = _ctx()
        result = engine.get_recommendation(ctx)
        assert not result.has_recommendations

        # accept
        engine.accept_habit(h.id)

        # accept 后 → 有推荐
        result = engine.get_recommendation(ctx)
        assert result.has_recommendations
        assert result.actions[0].parsed_actions["hvac_temp_target"] == 22

        engine.reset()
        engine.close()

    def test_reject_habit(self):
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username="test-executor-reject", llm_client=None)
        engine.reset()

        h = _habit("eco mode", accepted=True)
        engine.fact_store.store_facts(engine.username, [h])

        # reject → 删除
        assert engine.reject_habit(h.id) is True
        habits = engine.get_habits()
        assert len(habits) == 0

        engine.reset()
        engine.close()


# ═══════════════════════════════════════════════════
# _context_to_dict
# ═══════════════════════════════════════════════════

class TestContextToDict:
    def test_basic(self):
        ctx = _ctx(geofence="home")
        d = _context_to_dict(ctx)
        assert d["time_bucket"] == "early_morning"
        assert d["vehicle_state"] == "engine_started"
        assert d["geofence"] == "home"
        assert d["weekday"] is True
        assert d["hour"] == 8
