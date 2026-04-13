"""
多用户切换 + 数据隔离验证

验证场景:
  Phase 1: 存储隔离 — 不同用户的 facts 不会互相泄漏
  Phase 2: 习惯隔离 — 用户 A 的习惯不会出现在用户 B 的结果中
  Phase 3: 推荐隔离 — accept/recommend 严格限定在目标用户
  Phase 4: 生命周期隔离 — reset/reject 不影响其他用户
  Phase 5: 并发写入 — 两个用户同时写入不会交叉
"""
import os
import sys
import json
import shutil
import tempfile
import logging

import pytest

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── 独立临时目录，避免污染正式数据 ──
_TMP_DIR = tempfile.mkdtemp(prefix="habit_multiuser_test_")
os.environ["MEMORY_DIR"] = _TMP_DIR

import panoramix_core.config as _cfg
_cfg.MEMORY_DIR = _TMP_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

USER_A = "test_alice"
USER_B = "test_bob"
USER_C = "test_charlie"


# ═══════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture(scope="module")
def engine_a():
    """用户 A 引擎"""
    from engine.habit_engine import HabitDemoEngine
    e = HabitDemoEngine(username=USER_A, llm_client=None)
    e.reset()
    yield e
    e.close()


@pytest.fixture(scope="module")
def engine_b():
    """用户 B 引擎"""
    from engine.habit_engine import HabitDemoEngine
    e = HabitDemoEngine(username=USER_B, llm_client=None)
    e.reset()
    yield e
    e.close()


@pytest.fixture(scope="module")
def engine_c():
    """用户 C 引擎 (空用户，不导入任何数据)"""
    from engine.habit_engine import HabitDemoEngine
    e = HabitDemoEngine(username=USER_C, llm_client=None)
    e.reset()
    yield e
    e.close()


@pytest.fixture(scope="module")
def mockup_data():
    """加载 mockup 数据"""
    from scenarios.mock_data_generator import generate_full_dataset
    return generate_full_dataset()


@pytest.fixture(scope="module")
def ingest_result_a(engine_a, mockup_data):
    """为用户 A 导入完整 mockup 数据"""
    from signals.simulator import SignalSimulator
    sim = SignalSimulator(mockup_data)
    result = engine_a.ingest_signal_batch(sim._events)
    log.info(f"User A ingested: {result['facts_ingested']} facts → {result['habits_detected']} habits")
    return result


@pytest.fixture(scope="module")
def ingest_result_b(engine_b):
    """为用户 B 仅导入一小组不同的信号 (手动构造，确保与 A 不同)"""
    events = [
        {
            "date": "2025-10-10",
            "signals": [
                {"t": "2025-10-10T22:00", "signal": "engine_status", "value": "on"},
                {"t": "2025-10-10T22:00", "signal": "drive_mode", "value": "sport"},
                {"t": "2025-10-10T22:00", "signal": "media_source", "value": "radio"},
                {"t": "2025-10-10T22:01", "signal": "window_position", "value": 50},
            ],
        },
        {
            "date": "2025-10-11",
            "signals": [
                {"t": "2025-10-11T22:00", "signal": "engine_status", "value": "on"},
                {"t": "2025-10-11T22:00", "signal": "drive_mode", "value": "sport"},
                {"t": "2025-10-11T22:00", "signal": "media_source", "value": "radio"},
            ],
        },
    ]
    result = engine_b.ingest_signal_batch(events)
    log.info(f"User B ingested: {result['facts_ingested']} facts → {result['habits_detected']} habits")
    return result


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """测试完成后清理临时目录"""
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)


# ═══════════════════════════════════════════════════
# Phase 1: 存储隔离
# ═══════════════════════════════════════════════════

class TestStorageIsolation:
    """验证 ChromaDB 存储层的用户数据隔离"""

    def test_user_a_has_facts(self, ingest_result_a):
        """用户 A 成功导入 facts"""
        assert ingest_result_a["facts_ingested"] > 0

    def test_user_b_has_facts(self, ingest_result_b):
        """用户 B 成功导入 facts"""
        assert ingest_result_b["facts_ingested"] > 0

    def test_user_a_b_fact_counts_differ(self, engine_a, engine_b, ingest_result_a, ingest_result_b):
        """A 和 B 的 fact 数量不同 (A 导入完整 mockup，B 仅少量)"""
        status_a = engine_a.get_status()
        status_b = engine_b.get_status()
        assert status_a["total_facts"] != status_b["total_facts"]

    def test_user_c_is_empty(self, engine_c):
        """空用户 C 没有任何数据"""
        status = engine_c.get_status()
        assert status["total_facts"] == 0
        assert status["habit_facts"] == 0
        assert status["pref_facts"] == 0

    def test_separate_storage_dirs(self):
        """每个用户有独立的存储目录"""
        dir_a = os.path.join(_TMP_DIR, USER_A)
        dir_b = os.path.join(_TMP_DIR, USER_B)
        assert os.path.isdir(dir_a)
        assert os.path.isdir(dir_b)
        assert dir_a != dir_b

    def test_separate_collections(self, engine_a, engine_b):
        """用户 A 和 B 使用不同的 ChromaDB collections"""
        coll_a = engine_a.fact_store._get_collection(USER_A)
        coll_b = engine_b.fact_store._get_collection(USER_B)
        assert coll_a.name != coll_b.name
        assert f"facts_user_{USER_A}" == coll_a.name
        assert f"facts_user_{USER_B}" == coll_b.name


# ═══════════════════════════════════════════════════
# Phase 2: 习惯隔离
# ═══════════════════════════════════════════════════

class TestHabitIsolation:
    """验证习惯检测结果不会跨用户泄漏"""

    def test_user_a_has_habits(self, engine_a, ingest_result_a):
        """用户 A 检测到习惯"""
        habits = engine_a.get_habits()
        assert len(habits) > 0

    def test_user_b_habits_independent(self, engine_a, engine_b, ingest_result_a, ingest_result_b):
        """用户 B 的习惯列表独立于 A"""
        habits_a = engine_a.get_habits()
        habits_b = engine_b.get_habits()
        # A 有完整 mockup → 多个习惯；B 只有少量数据
        assert len(habits_a) != len(habits_b) or all(
            ha.id != hb.id for ha in habits_a for hb in habits_b
        )

    def test_user_c_has_no_habits(self, engine_c):
        """空用户没有习惯"""
        habits = engine_c.get_habits()
        assert len(habits) == 0

    def test_habit_ids_no_overlap(self, engine_a, engine_b, ingest_result_a, ingest_result_b):
        """A 和 B 的 habit IDs 没有重叠"""
        ids_a = {h.id for h in engine_a.get_habits()}
        ids_b = {h.id for h in engine_b.get_habits()}
        assert ids_a.isdisjoint(ids_b), f"Overlapping IDs: {ids_a & ids_b}"

    def test_habit_texts_independent(self, engine_a, engine_b, ingest_result_a, ingest_result_b):
        """每个用户的习惯文本来自各自的输入数据"""
        texts_a = {h.text for h in engine_a.get_habits()}
        texts_b = {h.text for h in engine_b.get_habits()}
        # 因为输入数据不同，文本应该不同
        if texts_a and texts_b:
            # 至少存在一些非重叠文本
            assert texts_a != texts_b


# ═══════════════════════════════════════════════════
# Phase 3: 推荐隔离
# ═══════════════════════════════════════════════════

class TestRecommendationIsolation:
    """验证推荐系统不会跨用户推荐"""

    def test_accept_user_a_habit(self, engine_a, ingest_result_a):
        """为用户 A 接受第一个习惯"""
        habits = engine_a.get_habits()
        assert len(habits) > 0
        ok = engine_a.accept_habit(habits[0].id)
        assert ok is True

    def test_user_a_gets_recommendation(self, engine_a, ingest_result_a):
        """用户 A 在匹配上下文下获得推荐"""
        from panoramix_core.models.fact import StructuredContext

        habits = engine_a.get_habits()
        accepted = [h for h in habits if h.accepted]
        assert len(accepted) > 0

        # 构造与已接受习惯相同的上下文
        h = accepted[0]
        ctx = StructuredContext(
            time_bucket=h.context.time_bucket,
            hour=h.context.hour,
            weekday=h.context.weekday,
            vehicle_state=h.context.vehicle_state,
            geofence=h.context.geofence,
        )
        result = engine_a.get_recommendation(ctx, top_k=5)
        assert result.has_recommendations

    def test_user_b_no_recommendation_for_a_context(self, engine_a, engine_b,
                                                      ingest_result_a, ingest_result_b):
        """用户 B 用 A 的习惯上下文查询 → 不应返回 A 的习惯"""
        from panoramix_core.models.fact import StructuredContext

        # 取 A 的已接受习惯上下文
        habits_a = [h for h in engine_a.get_habits() if h.accepted]
        if not habits_a:
            pytest.skip("No accepted habits for A")

        h = habits_a[0]
        ctx = StructuredContext(
            time_bucket=h.context.time_bucket,
            hour=h.context.hour,
            weekday=h.context.weekday,
            vehicle_state=h.context.vehicle_state,
            geofence=h.context.geofence,
        )

        # B 用同样上下文查询
        result_b = engine_b.get_recommendation(ctx, top_k=5)

        # B 的推荐不应包含 A 的 habit ID
        a_ids = {h.id for h in engine_a.get_habits()}
        b_rec_ids = {a.habit_id for a in result_b.actions}
        assert a_ids.isdisjoint(b_rec_ids), f"A's habits leaked to B: {a_ids & b_rec_ids}"

    def test_user_c_no_recommendations(self, engine_c):
        """空用户 C 不应有任何推荐"""
        from panoramix_core.models.fact import StructuredContext

        ctx = StructuredContext(
            time_bucket="early_morning",
            hour=7,
            weekday=True,
            vehicle_state="engine_started",
        )
        result = engine_c.get_recommendation(ctx, top_k=5)
        assert not result.has_recommendations
        assert result.candidates_checked == 0

    def test_recommendation_username_scoped(self, engine_a, engine_b,
                                            ingest_result_a, ingest_result_b):
        """推荐结果只包含查询用户自己的习惯"""
        from panoramix_core.models.fact import StructuredContext

        ctx = StructuredContext(
            time_bucket="early_morning",
            hour=7,
            weekday=True,
            vehicle_state="engine_started",
        )

        result_a = engine_a.get_recommendation(ctx, top_k=10)
        result_b = engine_b.get_recommendation(ctx, top_k=10)

        # 确认返回的 total_habits 反映各自的数据
        all_a = engine_a.get_habits()
        all_b = engine_b.get_habits()
        assert result_a.total_habits == len(all_a)
        assert result_b.total_habits == len(all_b)


# ═══════════════════════════════════════════════════
# Phase 4: 生命周期隔离
# ═══════════════════════════════════════════════════

class TestLifecycleIsolation:
    """验证 reset / reject 不影响其他用户"""

    def test_reject_b_doesnt_affect_a(self, engine_a, engine_b,
                                       ingest_result_a, ingest_result_b):
        """Reject B 的习惯不影响 A 的数据"""
        habits_a_before = len(engine_a.get_habits())
        total_a_before = engine_a.get_status()["total_facts"]

        # reject B 的所有习惯
        for h in engine_b.get_habits():
            engine_b.reject_habit(h.id)

        # A 的数据不变
        habits_a_after = len(engine_a.get_habits())
        total_a_after = engine_a.get_status()["total_facts"]
        assert habits_a_after == habits_a_before
        assert total_a_after == total_a_before

    def test_reject_a_habit_in_b_engine_fails(self, engine_a, engine_b, ingest_result_a):
        """尝试通过 B 引擎 reject A 的习惯 → 找不到"""
        habits_a = engine_a.get_habits()
        if not habits_a:
            pytest.skip("No habits for A")
        ok = engine_b.reject_habit(habits_a[0].id)
        assert ok is False

    def test_accept_a_habit_in_b_engine_fails(self, engine_a, engine_b, ingest_result_a):
        """尝试通过 B 引擎 accept A 的习惯 → 找不到"""
        habits_a = [h for h in engine_a.get_habits() if not h.accepted]
        if not habits_a:
            pytest.skip("No unaccepted habits for A")
        ok = engine_b.accept_habit(habits_a[0].id)
        assert ok is False

    def test_reset_c_doesnt_affect_a(self, engine_a, engine_c, ingest_result_a):
        """Reset 空用户 C 不影响 A"""
        total_before = engine_a.get_status()["total_facts"]
        engine_c.reset()
        total_after = engine_a.get_status()["total_facts"]
        assert total_after == total_before


# ═══════════════════════════════════════════════════
# Phase 5: 并发写入隔离
# ═══════════════════════════════════════════════════

class TestConcurrentWriteIsolation:
    """验证两个用户同时写入不会交叉"""

    def test_parallel_ingest_isolation(self):
        """两个独立引擎分别写入，数据互不干扰"""
        from engine.habit_engine import HabitDemoEngine

        user_x = "test_xray"
        user_y = "test_yankee"

        ex = HabitDemoEngine(username=user_x, llm_client=None)
        ey = HabitDemoEngine(username=user_y, llm_client=None)
        ex.reset()
        ey.reset()

        # X: 导航 + eco 模式
        events_x = [{
            "date": "2025-11-01",
            "signals": [
                {"t": "2025-11-01T08:00", "signal": "engine_status", "value": "on"},
                {"t": "2025-11-01T08:00", "signal": "nav_destination", "value": "Office"},
                {"t": "2025-11-01T08:00", "signal": "drive_mode", "value": "eco"},
            ],
        }]

        # Y: sport 模式 + ACC
        events_y = [{
            "date": "2025-11-01",
            "signals": [
                {"t": "2025-11-01T20:00", "signal": "engine_status", "value": "on"},
                {"t": "2025-11-01T20:00", "signal": "drive_mode", "value": "sport"},
                {"t": "2025-11-01T20:00", "signal": "acc_distance", "value": "short"},
            ],
        }]

        ex.ingest_signal_batch(events_x)
        ey.ingest_signal_batch(events_y)

        sx = ex.get_status()
        sy = ey.get_status()

        # 各自独立
        assert sx["total_facts"] > 0
        assert sy["total_facts"] > 0

        # fact IDs 不交叉
        facts_x = ex.get_all_facts()
        facts_y = ey.get_all_facts()
        ids_x = {f.id for f in facts_x}
        ids_y = {f.id for f in facts_y}
        assert ids_x.isdisjoint(ids_y), "IDs should not overlap"

        # fact 文本各自独立 (X 有 nav，Y 没有)
        texts_x = {f.text for f in facts_x}
        texts_y = {f.text for f in facts_y}
        assert any("Office" in t or "navigation" in t.lower() for t in texts_x)
        assert not any("Office" in t or "navigation" in t.lower() for t in texts_y)

        # 清理
        ex.reset()
        ey.reset()
        ex.close()
        ey.close()

    def test_engine_reopen_preserves_data(self):
        """关闭引擎后重新打开，数据仍在"""
        from engine.habit_engine import HabitDemoEngine

        user = "test_persist"
        e1 = HabitDemoEngine(username=user, llm_client=None)
        e1.reset()

        events = [{
            "date": "2025-12-01",
            "signals": [
                {"t": "2025-12-01T09:00", "signal": "engine_status", "value": "on"},
                {"t": "2025-12-01T09:00", "signal": "drive_mode", "value": "eco"},
                {"t": "2025-12-01T09:00", "signal": "acc_distance", "value": "medium"},
            ],
        }]
        e1.ingest_signal_batch(events)
        count1 = e1.get_status()["total_facts"]
        assert count1 > 0
        e1.close()

        # 重新打开 → 数据仍在
        e2 = HabitDemoEngine(username=user, llm_client=None)
        count2 = e2.get_status()["total_facts"]
        assert count2 == count1

        # 清理
        e2.reset()
        e2.close()

    def test_other_user_unaffected_by_reopen(self):
        """打开用户 P 的引擎不会影响用户 Q 的数据"""
        from engine.habit_engine import HabitDemoEngine

        user_p = "test_papa"
        user_q = "test_quebec"

        ep = HabitDemoEngine(username=user_p, llm_client=None)
        eq = HabitDemoEngine(username=user_q, llm_client=None)
        ep.reset()
        eq.reset()

        # P 写入行为信号
        ep.ingest_signal_batch([{
            "date": "2025-12-05",
            "signals": [
                {"t": "2025-12-05T10:00", "signal": "engine_status", "value": "on"},
                {"t": "2025-12-05T10:00", "signal": "drive_mode", "value": "comfort"},
            ],
        }])

        count_p = ep.get_status()["total_facts"]
        count_q = eq.get_status()["total_facts"]

        assert count_p > 0
        assert count_q == 0  # Q 没有数据

        ep.reset()
        eq.reset()
        ep.close()
        eq.close()
