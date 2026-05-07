"""
边界测试 — 习惯中断与拒绝

场景 1: 中间习惯中断 → 重新学习
  - 用户建立习惯模式 → 系统检测到习惯
  - 用户中断习惯（停止重复行为一段时间）
  - 用户恢复同样行为 → 系统能重新学习并检测

场景 2: 用户拒绝推荐（reject）后的系统处理
  - reject 后习惯从存储中删除
  - reject 后相同上下文不再推荐
  - reject 后重新 ingest 相同信号 → 系统能重新检测
  - reject 不存在的 habit_id → 返回 False
  - reject 后其他习惯不受影响

运行:
  python -m pytest tests/test_boundary_interruption_rejection.py -v
"""
import json
import os
import sys
import pytest
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.habit_engine import HabitDemoEngine
from engine.proactive_executor import ProactiveExecutor
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _ctx(time_bucket="early_morning", vehicle_state="engine_started",
         geofence="home", weekday=True, hour=8):
    return StructuredContext(
        time_bucket=time_bucket, hour=hour, weekday=weekday,
        vehicle_state=vehicle_state, geofence=geofence,
    )


def _habit(text, time_bucket="early_morning", vehicle_state="engine_started",
           geofence="home", weekday=True, hour=8, accepted=False,
           clustering_confidence=0.85, scene_name="Morning Commute"):
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


def _pref_fact(text, time_bucket="early_morning", vehicle_state="engine_started",
               geofence="home", weekday=True, hour=8):
    """模拟原始 PREF fact（聚类前的信号 fact）"""
    return Fact(
        text=text,
        type=FactType.PREF,
        durability=FactDurability.LONG_TERM,
        time_stamp=datetime.now(),
        source=FactSources.SIGNAL,
        accepted=False,
        context=_ctx(time_bucket, vehicle_state, geofence, weekday, hour),
    )


class FakeFactStore:
    """模拟 FactStore"""

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

    def get_facts_with_embeddings(self, username, **kwargs):
        return []

    def close(self):
        pass


# ═══════════════════════════════════════════════════
# 场景 1: 习惯中断 → 重新学习
# ═══════════════════════════════════════════════════

class TestHabitInterruptionAndRelearning:
    """
    测试习惯中断后重新学习的系统行为。

    真实场景: 用户每天早上设置空调22度（形成习惯），
    然后停了2周不再做这个操作，之后又恢复每天设置。
    系统应该能重新检测到这个习惯。
    """

    def test_habit_exists_then_deleted_then_relearned(self):
        """
        习惯被检测 → 手动删除（模拟过期/中断） → 重新存入相同模式 → 系统仍可识别
        """
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)

        # Phase 1: 习惯存在且被接受，推荐正常工作
        h = _habit("set cabin air conditioning temperature to 22 degrees",
                    accepted=True)
        store.store_facts("user1", [h])

        ctx = _ctx()
        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations, "Phase 1: 已接受习惯应有推荐"

        # Phase 2: 习惯中断 — 从存储中移除（模拟过期清理或中断丢失）
        store.delete_facts("user1", [h.id])

        result = executor.recommend(ctx, "user1")
        assert not result.has_recommendations, "Phase 2: 习惯删除后应无推荐"

        # Phase 3: 重新学习 — 新的习惯 fact 被检测并存入
        h_new = _habit("set cabin air conditioning temperature to 22 degrees",
                       accepted=True, clustering_confidence=0.90)
        store.store_facts("user1", [h_new])

        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations, "Phase 3: 重新学习后应有推荐"
        assert result.actions[0].clustering_confidence == 0.90

    def test_interrupted_habit_does_not_ghost_recommend(self):
        """
        习惯被删除后，不应出现"幽灵推荐"（数据残留）
        """
        store = FakeFactStore()
        executor = ProactiveExecutor(store)

        h = _habit("eco mode", accepted=True)
        store.store_facts("user1", [h])

        # 删除
        store.delete_facts("user1", [h.id])

        # 确认：store 中 HABIT 类型为空
        remaining = store.get_facts("user1", types=[FactType.HABIT])
        assert len(remaining) == 0, "删除后不应有 HABIT 残留"

        # 推荐也应为空
        result = executor.recommend(_ctx(), "user1")
        assert not result.has_recommendations

    def test_partial_pattern_after_interruption(self):
        """
        中断后只恢复了部分模式（不同上下文），不应匹配原习惯上下文
        """
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)

        # 原习惯: 早上+引擎启动+home
        # 恢复习惯: 同操作但在 evening+parked+workplace
        h_new = _habit("set cabin air conditioning temperature to 22 degrees",
                       accepted=True,
                       time_bucket="evening", vehicle_state="parked",
                       geofence="workplace")
        store.store_facts("user1", [h_new])

        # 用原来的上下文查询 → 不应匹配（上下文距离太大）
        ctx_morning = _ctx(time_bucket="early_morning",
                           vehicle_state="engine_started", geofence="home")
        result = executor.recommend(ctx_morning, "user1")
        assert not result.has_recommendations, "不同上下文的重新学习不应匹配原场景"

        # 用新上下文查询 → 应匹配
        ctx_evening = _ctx(time_bucket="evening", vehicle_state="parked",
                           geofence="workplace")
        result = executor.recommend(ctx_evening, "user1")
        assert result.has_recommendations, "新上下文应匹配重新学习的习惯"

    def test_relearned_habit_gets_new_id(self):
        """
        重新学习的习惯应有新 ID（不复用旧 ID）
        """
        h_old = _habit("set AC to 22", accepted=True)
        old_id = h_old.id

        h_new = _habit("set AC to 22", accepted=True)
        new_id = h_new.id

        assert old_id != new_id, "重新学习的习惯应获得新的 UUID"


# ═══════════════════════════════════════════════════
# 场景 2: 拒绝推荐（reject）后的系统处理
# ═══════════════════════════════════════════════════

class TestRejectHabitBoundary:
    """
    测试 reject_habit 的边界行为
    """

    def test_reject_removes_from_store(self):
        """reject 后习惯从存储中完全删除"""
        store = FakeFactStore()
        h = _habit("set AC to 22", accepted=True)
        store.store_facts("user1", [h])

        # 模拟 engine.reject_habit 逻辑
        habits = store.get_facts("user1", types=[FactType.HABIT])
        target = [x for x in habits if x.id == h.id]
        assert len(target) == 1

        store.delete_facts("user1", [h.id])

        remaining = store.get_facts("user1", types=[FactType.HABIT])
        assert len(remaining) == 0, "reject 后应无 HABIT 残留"

    def test_reject_stops_recommendation(self):
        """reject 后相同上下文不再推荐该习惯"""
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)

        h = _habit("set AC to 22", accepted=True)
        store.store_facts("user1", [h])

        ctx = _ctx()

        # reject 前有推荐
        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations

        # reject
        store.delete_facts("user1", [h.id])

        # reject 后无推荐
        result = executor.recommend(ctx, "user1")
        assert not result.has_recommendations, "reject 后不应再推荐"

    def test_reject_nonexistent_habit_returns_false(self):
        """reject 不存在的 habit_id → 返回 False"""
        engine = HabitDemoEngine(username="test_reject_nonexist", llm_client=None)
        engine.reset()

        result = engine.reject_habit("nonexistent-id-12345")
        assert result is False, "reject 不存在的 ID 应返回 False"

        engine.reset()
        engine.close()

    def test_reject_one_does_not_affect_others(self):
        """reject 一个习惯不影响其他习惯"""
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)

        h1 = _habit("set AC to 22", accepted=True)
        h2 = _habit("turned on seat heating to level 2", accepted=True)
        store.store_facts("user1", [h1, h2])

        ctx = _ctx()

        # 两个都能推荐
        result = executor.recommend(ctx, "user1")
        assert len(result.actions) == 2

        # reject h1
        store.delete_facts("user1", [h1.id])

        # h2 仍然被推荐
        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations
        assert len(result.actions) == 1
        assert result.actions[0].habit_id == h2.id

    def test_reject_then_reingest_same_pattern(self):
        """
        reject 后重新注入相同模式 → 系统能重新检测出习惯。
        这验证 reject 不会产生"黑名单"效应。
        """
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)

        h = _habit("set AC to 22", accepted=True)
        store.store_facts("user1", [h])

        # reject
        store.delete_facts("user1", [h.id])
        assert not executor.recommend(_ctx(), "user1").has_recommendations

        # 重新注入同样的习惯（模拟 re-learn）
        h_new = _habit("set AC to 22", accepted=True,
                       clustering_confidence=0.92)
        store.store_facts("user1", [h_new])

        result = executor.recommend(_ctx(), "user1")
        assert result.has_recommendations, "reject 后重新学习应能推荐"
        assert result.actions[0].clustering_confidence == 0.92

    def test_reject_accepted_vs_unaccepted(self):
        """reject pending 场景卡应成功"""
        from panoramix_core.models.habit import Habit as HabitModel

        engine = HabitDemoEngine(username="test_reject_both", llm_client=None)
        engine.reset()

        structural_key = "test_key_eco"
        habit = HabitModel(
            username=engine.username, batch_id=1,
            text="eco mode", signal_category="categorical",
            signal_name="drive_mode", structural_key=structural_key,
            context_time_bucket="early_morning", context_vehicle_state="engine_started",
            context_weekday=1, raw_value_stats={}, member_fact_ids=[],
        )
        with engine.scene_card_store.transaction() as conn:
            engine.habit_store.insert_many([habit], batch_id=1, conn=conn)
            engine.scene_card_store.upsert_pending_by_structural_key(
                structural_key=structural_key,
                card_data={"display_name": "Test", "content_snapshot": {}},
                batch_id=1, conn=conn,
            )

        result = engine.reject_habit(habit.habit_id)
        assert result is True, "reject pending 场景卡应返回 True"

        engine.reset()
        engine.close()

    def test_double_reject_same_habit(self):
        """连续 reject 同一个 habit → 第二次返回 False"""
        from panoramix_core.models.habit import Habit as HabitModel

        engine = HabitDemoEngine(username="test_double_reject", llm_client=None)
        engine.reset()

        structural_key = "test_key_double"
        habit = HabitModel(
            username=engine.username, batch_id=1,
            text="eco mode", signal_category="categorical",
            signal_name="drive_mode", structural_key=structural_key,
            context_time_bucket="early_morning", context_vehicle_state="engine_started",
            context_weekday=1, raw_value_stats={}, member_fact_ids=[],
        )
        with engine.scene_card_store.transaction() as conn:
            engine.habit_store.insert_many([habit], batch_id=1, conn=conn)
            engine.scene_card_store.upsert_pending_by_structural_key(
                structural_key=structural_key,
                card_data={"display_name": "Test", "content_snapshot": {}},
                batch_id=1, conn=conn,
            )

        assert engine.reject_habit(habit.habit_id) is True
        assert engine.reject_habit(habit.habit_id) is False, "重复 reject 应返回 False"

        engine.reset()
        engine.close()


# ═══════════════════════════════════════════════════
# 组合场景: 中断 + 拒绝 交叉
# ═══════════════════════════════════════════════════

class TestInterruptionRejectionCombined:
    """中断与拒绝的交叉边界场景"""

    def test_reject_then_interrupt_then_relearn(self):
        """
        reject → 中断 → 重新学习 → accept → 推荐恢复

        完整生命周期：
        1. 习惯被检测并接受
        2. 用户 reject（不想要这个推荐了）
        3. 一段时间后用户又开始做同样的事
        4. 系统重新检测到习惯
        5. 用户接受新习惯
        6. 推荐恢复
        """
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)
        ctx = _ctx()

        # Step 1: 习惯存在且已接受
        h1 = _habit("set AC to 22", accepted=True)
        store.store_facts("user1", [h1])
        assert executor.recommend(ctx, "user1").has_recommendations

        # Step 2: reject
        store.delete_facts("user1", [h1.id])
        assert not executor.recommend(ctx, "user1").has_recommendations

        # Step 3-4: 重新检测到习惯（但未 accept）
        h2 = _habit("set AC to 22", accepted=False, clustering_confidence=0.88)
        store.store_facts("user1", [h2])
        assert not executor.recommend(ctx, "user1").has_recommendations, \
            "未 accept 的重新学习习惯不应推荐"

        # Step 5: accept
        h2.accepted = True
        store.delete_facts("user1", [h2.id])
        store.store_facts("user1", [h2])

        # Step 6: 推荐恢复
        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations, "accept 后推荐应恢复"

    def test_multiple_habits_reject_one_interrupt_another(self):
        """
        多习惯场景：reject A，习惯 B 被中断删除
        只有重新学习的 B（accept 后）能推荐
        """
        store = FakeFactStore()
        executor = ProactiveExecutor(store, threshold=0.35)
        ctx = _ctx()

        h_a = _habit("set AC to 22", accepted=True)
        h_b = _habit("turned on seat heating to level 2", accepted=True)
        store.store_facts("user1", [h_a, h_b])

        # reject A
        store.delete_facts("user1", [h_a.id])
        # interrupt B (模拟过期清除)
        store.delete_facts("user1", [h_b.id])

        assert not executor.recommend(ctx, "user1").has_recommendations

        # B 重新学习并 accept
        h_b_new = _habit("turned on seat heating to level 2",
                         accepted=True, clustering_confidence=0.91)
        store.store_facts("user1", [h_b_new])

        result = executor.recommend(ctx, "user1")
        assert result.has_recommendations
        assert len(result.actions) == 1
        assert "seat heating" in result.actions[0].habit_text

    def test_engine_accept_reject_accept_cycle(self):
        """
        通过 HabitDemoEngine 测试 accept → reject → 重新 accept 循环
        """
        from panoramix_core.models.habit import Habit as HabitModel

        engine = HabitDemoEngine(username="cycle_test", llm_client=None)
        engine.reset()
        ctx = _ctx()

        structural_key = "test_key_ac"
        habit = HabitModel(
            username=engine.username, batch_id=1,
            text="set AC to 22", signal_category="numeric",
            signal_name="hvac_temp_target", structural_key=structural_key,
            context_time_bucket="early_morning", context_vehicle_state="engine_started",
            context_geofence="home", context_weekday=1,
            raw_value_stats={"clustering_confidence": 0.8},
            member_fact_ids=[],
        )
        with engine.scene_card_store.transaction() as conn:
            engine.habit_store.insert_many([habit], batch_id=1, conn=conn)
            engine.scene_card_store.upsert_pending_by_structural_key(
                structural_key=structural_key,
                card_data={"display_name": "Morning AC",
                           "content_snapshot": {"habits": []}},
                batch_id=1, conn=conn,
            )

        # accept
        assert engine.accept_habit(habit.habit_id) is True
        result = engine.get_recommendation(ctx)
        assert result.has_recommendations

        # reject (retire accepted card)
        assert engine.reject_habit(habit.habit_id) is True
        result = engine.get_recommendation(ctx)
        assert not result.has_recommendations

        # 重新存入并 accept（模拟重新学习）
        habit2 = HabitModel(
            username=engine.username, batch_id=2,
            text="set AC to 22", signal_category="numeric",
            signal_name="hvac_temp_target", structural_key=structural_key + "_v2",
            context_time_bucket="early_morning", context_vehicle_state="engine_started",
            context_geofence="home", context_weekday=1,
            raw_value_stats={"clustering_confidence": 0.9},
            member_fact_ids=[],
        )
        with engine.scene_card_store.transaction() as conn:
            engine.habit_store.insert_many([habit2], batch_id=2, conn=conn)
            engine.scene_card_store.upsert_pending_by_structural_key(
                structural_key=structural_key + "_v2",
                card_data={"display_name": "Morning AC v2",
                           "content_snapshot": {"habits": []}},
                batch_id=2, conn=conn,
            )
        assert engine.accept_habit(habit2.habit_id) is True

        result = engine.get_recommendation(ctx)
        assert result.has_recommendations, "accept→reject→accept 循环后应能推荐"

        engine.reset()
        engine.close()
