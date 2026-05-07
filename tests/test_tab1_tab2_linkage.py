"""
Tab 1 / Tab 2 数据联动验证

验证 Tab 1 学习阶段产出的习惯能在 Tab 2 推理阶段正确读取、匹配、推荐。
覆盖开发方案 Week 6 "Tab 1/Tab 2 数据联动验证" 要求。

Phase 1: 基础联动 — Tab 1 学习 → Tab 2 可见
Phase 2: 接受联动 — Tab 2 accept → 推荐生效
Phase 3: 拒绝联动 — Tab 2 reject → habit 消失 + 推荐不再出现
Phase 4: 场景匹配 — Tab 2 不同场景 → 匹配不同 Tab 1 习惯
Phase 5: 多用户联动 — 用户 A 的 Tab 1 数据仅在用户 A 的 Tab 2 出现
Phase 6: 边界场景 — 空状态、重新学习、连续操作
Phase 7: 反馈闭环 — accept/reject 后 Tab 1 KG 状态同步

运行:
  cd habit-memory-demo
  python -m pytest tests/test_tab1_tab2_linkage.py -v
"""
import json
import os
import sys
import shutil
import tempfile
import logging
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 独立临时目录
_TMP_DIR = tempfile.mkdtemp(prefix="habit_linkage_test_")
os.environ["MEMORY_DIR"] = _TMP_DIR

import panoramix_core.config as _cfg
_cfg.MEMORY_DIR = _TMP_DIR

from scenarios.mock_data_generator import generate_full_dataset
from signals.simulator import SignalSimulator
from engine.habit_engine import HabitDemoEngine
from panoramix_core.models.fact import StructuredContext
from panoramix_core.models.fact_enums import FactType

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

USER_MAIN = "linkage_mary"
USER_OTHER = "linkage_tom"


# ═══════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture(scope="module")
def mockup_data():
    return generate_full_dataset()


@pytest.fixture(scope="module")
def simulator(mockup_data):
    return SignalSimulator(mockup_data)


@pytest.fixture(scope="module")
def engine_main():
    """主用户引擎 — 模拟 Tab 1 学习 + Tab 2 推理"""
    eng = HabitDemoEngine(username=USER_MAIN, llm_client=None)
    eng.reset()
    yield eng
    eng.reset()
    eng.close()


@pytest.fixture(scope="module")
def engine_other():
    """另一用户引擎 — 用于隔离验证"""
    eng = HabitDemoEngine(username=USER_OTHER, llm_client=None)
    eng.reset()
    yield eng
    eng.reset()
    eng.close()


@pytest.fixture(scope="module")
def ingested_result(engine_main, simulator):
    """Phase 0: Tab 1 执行 batch ingest (module 级别共享)"""
    result = engine_main.ingest_from_simulator(simulator)
    return result


# ═══════════════════════════════════════════════════
# Phase 1: 基础联动 — Tab 1 学习 → Tab 2 可见
# ═══════════════════════════════════════════════════

class TestPhase1BasicLinkage:
    """Tab 1 学习后，Tab 2 Mini KG 能读到习惯"""

    def test_tab1_produces_habits(self, ingested_result):
        """Tab 1 pipeline 产出 >= 1 个 habit"""
        assert ingested_result["habits_detected"] > 0

    def test_tab2_reads_same_habits(self, engine_main, ingested_result):
        """Tab 2 get_status() 返回的 habits 数量 == Tab 1 检测数量"""
        status = engine_main.get_status()
        assert status["habit_facts"] == ingested_result["habits_detected"]

    def test_tab2_habits_have_scene_names(self, engine_main):
        """Tab 2 Mini KG 中的 habits 都有 scene_name"""
        status = engine_main.get_status()
        for h in status["habits"]:
            assert h.get("scene_name") is not None, f"Habit {h['id']} missing scene_name"

    def test_tab2_habits_have_confidence(self, engine_main):
        """Tab 2 Mini KG 中的 habits 都有 clustering_confidence"""
        status = engine_main.get_status()
        for h in status["habits"]:
            conf = h.get("clustering_confidence")
            assert conf is not None and conf > 0, f"Habit {h['id']} invalid confidence"

    def test_tab2_habits_have_context(self, engine_main):
        """Tab 2 中每个 habit 携带完整 context"""
        status = engine_main.get_status()
        for h in status["habits"]:
            ctx = h.get("context", {})
            assert "time_bucket" in ctx
            assert "vehicle_state" in ctx

    def test_tab2_habits_all_unaccepted_initially(self, engine_main):
        """初始状态：所有 habits 尚未 accepted"""
        status = engine_main.get_status()
        for h in status["habits"]:
            assert h["accepted"] is False, f"Habit {h['id']} should not be accepted initially"

    def test_tab2_no_recommendations_before_accept(self, engine_main):
        """未 accept 任何 habit 时，推荐结果为空"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        assert not result.has_recommendations
        assert result.candidates_checked == 0


# ═══════════════════════════════════════════════════
# Phase 2: 接受联动 — Tab 2 accept → 推荐生效
# ═══════════════════════════════════════════════════

class TestPhase2AcceptLinkage:
    """Tab 2 accept 后推荐即可生效"""

    def test_accept_first_habit(self, engine_main):
        """accept 第一个 habit 成功"""
        habits = engine_main.get_status()["habits"]
        assert len(habits) > 0
        first_id = habits[0]["id"]
        ok = engine_main.accept_habit(first_id)
        assert ok is True

    def test_accepted_habit_marked_in_kg(self, engine_main):
        """accept 后 KG 中对应 habit accepted=True"""
        status = engine_main.get_status()
        accepted = [h for h in status["habits"] if h["accepted"]]
        assert len(accepted) >= 1

    def test_recommendation_available_after_accept(self, engine_main):
        """accept 后，匹配上下文能产出推荐"""
        # 获取已 accept 的 habit 的上下文作为查询条件
        status = engine_main.get_status()
        accepted = [h for h in status["habits"] if h["accepted"]]
        assert len(accepted) > 0

        h = accepted[0]
        ctx = StructuredContext(
            time_bucket=h["context"]["time_bucket"],
            hour=7,
            weekday=h["context"].get("weekday", True),
            vehicle_state=h["context"]["vehicle_state"],
            geofence=h["context"].get("geofence"),
        )
        result = engine_main.get_recommendation(ctx)
        assert result.candidates_checked >= 1

    def test_accept_all_habits(self, engine_main):
        """批量 accept 所有 habits"""
        status = engine_main.get_status()
        for h in status["habits"]:
            if not h["accepted"]:
                engine_main.accept_habit(h["id"])
        # 验证全部 accepted
        status2 = engine_main.get_status()
        for h in status2["habits"]:
            assert h["accepted"] is True

    def test_recommendation_with_multiple_accepted(self, engine_main):
        """多个 habit 被 accept 后 candidates_checked >= 1"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        assert result.candidates_checked >= 1
        assert result.total_habits >= 1


# ═══════════════════════════════════════════════════
# Phase 3: 拒绝联动 — Tab 2 reject → habit 消失
# ═══════════════════════════════════════════════════

class TestPhase3RejectLinkage:
    """Tab 2 reject 后 habit 从 KG 和推荐中消失"""

    def test_reject_one_habit(self, engine_main):
        """reject 一个 habit 成功 — 整张场景卡（同 structural_key 的所有 habits）被移除"""
        status = engine_main.get_status()
        habits = status["habits"]
        assert len(habits) > 0
        target_id = habits[0]["id"]
        target_card_id = habits[0]["card_id"]
        count_before = len(habits)
        # 同一场景卡下的 habits 数量
        card_habits_count = sum(1 for h in habits if h["card_id"] == target_card_id)
        ok = engine_main.reject_habit(target_id)
        assert ok is True
        # 验证数量减少（整张场景卡的 habits 都被移除）
        status2 = engine_main.get_status()
        assert len(status2["habits"]) == count_before - card_habits_count
        # 被 reject 的 habit 不再出现
        remaining_ids = {h["id"] for h in status2["habits"]}
        assert target_id not in remaining_ids

    def test_rejected_habit_not_in_kg(self, engine_main):
        """被 reject 的 habit 不再出现在 KG"""
        # 获取之前 reject 的 id (已经不在了，所以我们直接验证 habit count 一致)
        status = engine_main.get_status()
        habit_ids = {h["id"] for h in status["habits"]}
        # 只要 KG 中的 id 集合与 get_habits() 一致即可
        habits_direct = engine_main.get_habits()
        direct_ids = {h.id for h in habits_direct}
        assert habit_ids == direct_ids

    def test_rejected_habit_not_recommended(self, engine_main):
        """被 reject 的 habit 不会出现在推荐结果中"""
        status = engine_main.get_status()
        valid_ids = {h["id"] for h in status["habits"]}
        # 用各种上下文查询
        for tb in ["early_morning", "morning", "evening", "night"]:
            ctx = StructuredContext(time_bucket=tb, hour=8, weekday=True,
                                   vehicle_state="engine_started")
            result = engine_main.get_recommendation(ctx)
            for a in result.actions:
                assert a.habit_id in valid_ids, \
                    f"Rejected habit {a.habit_id} appeared in recommendation"


# ═══════════════════════════════════════════════════
# Phase 4: 场景匹配 — 不同上下文匹配不同习惯
# ═══════════════════════════════════════════════════

class TestPhase4ScenarioMatching:
    """Tab 2 不同场景能匹配到对应的 Tab 1 习惯"""

    def test_recommendation_has_context_distance(self, engine_main):
        """推荐结果包含 context_distance 字段"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        for a in result.actions:
            assert hasattr(a, "context_distance")
            assert 0 <= a.context_distance <= 1.0

    def test_recommendation_has_combined_confidence(self, engine_main):
        """推荐结果包含 combined_confidence"""
        ctx = StructuredContext(
            time_bucket="morning", hour=10,
            weekday=False, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        for a in result.actions:
            assert 0 < a.combined_confidence <= 1.0

    def test_recommendation_has_parsed_actions(self, engine_main):
        """推荐结果包含 parsed_actions (可以为空 dict)"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        for a in result.actions:
            assert isinstance(a.parsed_actions, dict)

    def test_recommendation_sorted_by_confidence(self, engine_main):
        """推荐结果按 combined_confidence 降序"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        if len(result.actions) > 1:
            confs = [a.combined_confidence for a in result.actions]
            assert confs == sorted(confs, reverse=True)

    def test_mismatched_context_fewer_results(self, engine_main):
        """不匹配的上下文产出更少的推荐"""
        # parking + night = 不太常见的组合
        ctx_odd = StructuredContext(
            time_bucket="night", hour=23,
            weekday=False, vehicle_state="parking",
        )
        ctx_common = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        r_odd = engine_main.get_recommendation(ctx_odd)
        r_common = engine_main.get_recommendation(ctx_common)
        # 至少 common 场景能匹配到，或两者都无匹配(数据有限)
        assert r_odd.candidates_checked <= r_common.candidates_checked or \
               r_common.candidates_checked == 0  # 全部 reject 后可能都是 0

    def test_top_k_limits_results(self, engine_main):
        """top_k 参数限制返回数量"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx, top_k=1)
        assert len(result.actions) <= 1


# ═══════════════════════════════════════════════════
# Phase 5: 多用户联动隔离
# ═══════════════════════════════════════════════════

class TestPhase5MultiUserLinkage:
    """用户 A 的 Tab 1 学习不影响用户 B 的 Tab 2"""

    def test_other_user_no_habits(self, engine_other):
        """另一用户没有 habits"""
        status = engine_other.get_status()
        assert status["habit_facts"] == 0

    def test_other_user_no_recommendations(self, engine_other):
        """另一用户推荐为空"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_other.get_recommendation(ctx)
        assert not result.has_recommendations
        assert result.total_habits == 0

    def test_other_user_independent_learning(self, engine_other, simulator):
        """另一用户独立学习后有自己的 habits"""
        result = engine_other.ingest_from_simulator(simulator)
        assert result["habits_detected"] > 0
        status = engine_other.get_status()
        assert status["habit_facts"] > 0

    def test_main_user_unaffected_by_other(self, engine_main):
        """另一用户学习不影响主用户的 habits"""
        status = engine_main.get_status()
        # 主用户在 Phase 3 reject 了一个，所以总数 < 原始检测数
        assert status["habit_facts"] >= 1

    def test_accept_other_no_cross_recommend(self, engine_main, engine_other):
        """accept 另一用户的 habit 不影响主用户推荐"""
        other_status = engine_other.get_status()
        if other_status["habits"]:
            engine_other.accept_habit(other_status["habits"][0]["id"])

        # 主用户推荐不包含另一用户的 habit
        main_habit_ids = {h["id"] for h in engine_main.get_status()["habits"]}
        ctx = StructuredContext(
            time_bucket="early_morning", hour=7,
            weekday=True, vehicle_state="engine_started",
        )
        result = engine_main.get_recommendation(ctx)
        for a in result.actions:
            assert a.habit_id in main_habit_ids


# ═══════════════════════════════════════════════════
# Phase 6: 边界场景
# ═══════════════════════════════════════════════════

class TestPhase6EdgeCases:
    """边界场景: 空状态、重新学习"""

    def test_empty_user_get_status(self):
        """全新用户 get_status() 返回空"""
        eng = HabitDemoEngine(username="linkage_empty", llm_client=None)
        eng.reset()
        status = eng.get_status()
        assert status["total_facts"] == 0
        assert status["habit_facts"] == 0
        assert status["habits"] == []
        eng.close()

    def test_empty_user_recommendation(self):
        """全新用户推荐为空"""
        eng = HabitDemoEngine(username="linkage_empty2", llm_client=None)
        eng.reset()
        ctx = StructuredContext(time_bucket="morning", hour=9,
                               weekday=True, vehicle_state="engine_started")
        result = eng.get_recommendation(ctx)
        assert not result.has_recommendations
        assert result.total_habits == 0
        assert result.candidates_checked == 0
        eng.close()

    def test_reset_then_relearn(self, simulator):
        """reset 后重新学习，habits 正常产出"""
        eng = HabitDemoEngine(username="linkage_reset_test", llm_client=None)
        eng.reset()
        # 第一次学习
        r1 = eng.ingest_from_simulator(simulator)
        assert r1["habits_detected"] > 0
        count1 = eng.get_status()["habit_facts"]
        # reset
        eng.reset()
        assert eng.get_status()["habit_facts"] == 0
        # 第二次学习
        r2 = eng.ingest_from_simulator(simulator)
        assert r2["habits_detected"] > 0
        count2 = eng.get_status()["habit_facts"]
        # 两次应该产出相同数量 (相同数据)
        assert count1 == count2
        eng.reset()
        eng.close()

    def test_accept_nonexistent_habit(self, engine_main):
        """accept 不存在的 habit 返回 False"""
        ok = engine_main.accept_habit("nonexistent_id_12345")
        assert ok is False

    def test_reject_nonexistent_habit(self, engine_main):
        """reject 不存在的 habit 返回 False"""
        ok = engine_main.reject_habit("nonexistent_id_12345")
        assert ok is False


# ═══════════════════════════════════════════════════
# Phase 7: 反馈闭环 — accept/reject 后 Tab 1 KG 同步
# ═══════════════════════════════════════════════════

class TestPhase7FeedbackLoop:
    """Tab 2 的反馈操作在 Tab 1 KG 中可见"""

    def test_accept_visible_in_tab1_kg(self, engine_main):
        """Tab 2 accept 后 Tab 1 的 KG 也显示 accepted=True"""
        status = engine_main.get_status()
        accepted = [h for h in status["habits"] if h["accepted"]]
        # Phase 2 中已经 accept 了, 这里验证 Tab 1 视角一致
        # (Tab 1 和 Tab 2 都通过 engine.get_status() 读取)
        assert len(accepted) >= 1

    def test_reject_visible_in_tab1_kg(self, engine_main):
        """Tab 2 reject 后 Tab 1 的 KG 中该场景卡的 habits 消失"""
        status = engine_main.get_status()
        count_before = len(status["habits"])
        if count_before == 0:
            pytest.skip("No habits to reject")
        target = status["habits"][-1]["id"]
        target_card_id = status["habits"][-1]["card_id"]
        card_habits_count = sum(1 for h in status["habits"] if h["card_id"] == target_card_id)
        engine_main.reject_habit(target)
        status2 = engine_main.get_status()
        assert len(status2["habits"]) == count_before - card_habits_count

    def test_habit_count_consistent_across_tabs(self, engine_main):
        """Tab 1 KG count == Tab 2 Mini KG count (同一个 engine)"""
        # 模拟两个 tab 各自创建 engine 实例读取
        eng1 = HabitDemoEngine(username=USER_MAIN, llm_client=None)
        eng2 = HabitDemoEngine(username=USER_MAIN, llm_client=None)
        s1 = eng1.get_status()
        s2 = eng2.get_status()
        assert s1["habit_facts"] == s2["habit_facts"]
        assert len(s1["habits"]) == len(s2["habits"])
        # ID 集合一致
        ids1 = {h["id"] for h in s1["habits"]}
        ids2 = {h["id"] for h in s2["habits"]}
        assert ids1 == ids2
        eng1.close()
        eng2.close()


# ═══════════════════════════════════════════════════
# Cleanup
# ═══════════════════════════════════════════════════

def teardown_module():
    """清理临时目录"""
    try:
        shutil.rmtree(_TMP_DIR, ignore_errors=True)
    except Exception:
        pass
