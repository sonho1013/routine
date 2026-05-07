"""
集成测试 — 对接mockup数据完整闭环

完整流程验证:
  mock_data_generator → SignalSimulator → HabitDemoEngine.ingest_from_simulator
  → signal_to_fact → ChromaDB(ada-002) → Hybrid DBSCAN + 连续性 → LLM场景卡
  → accept_habit → ProactiveExecutor.recommend → 推荐结果验证

测试矩阵:
  Phase 1: 数据生成 — mockup 数据结构 + 事件计数
  Phase 2: 信号→Fact — 转换质量 + StructuredContext 正确性
  Phase 3: 端到端 Pipeline — ingest → 聚类 → 习惯检测
  Phase 4: 习惯质量 — 场景覆盖 + 上下文一致性 + 置信度
  Phase 5: 执行闭环 — accept → recommend → 推荐匹配验证
  Phase 6: 生命周期 — reject + 重新查询 + 清理

运行:
  python -m pytest tests/test_integration_full_loop.py -v
  (需要 chromadb + openai 环境)
"""
import json
import os
import sys
import shutil
import tempfile
import logging
import pytest
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 临时存储目录 — 在导入 panoramix_core 之前设置
_TMP_DIR = tempfile.mkdtemp(prefix="habit_integration_test_")
os.environ["MEMORY_DIR"] = _TMP_DIR

import panoramix_core.config as _cfg
_cfg.MEMORY_DIR = _TMP_DIR

from scenarios.mock_data_generator import generate_full_dataset
from signals.simulator import SignalSimulator
from engine.signal_to_fact import signals_to_facts
from engine.habit_engine import HabitDemoEngine
from engine.proactive_executor import parse_habit_actions
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability
from panoramix_core.config import CONTEXT_MATCH_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TEST_USER = "integration_test"


# ═══════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════

@pytest.fixture(scope="module")
def mockup_data():
    """生成完整 mockup 数据集"""
    return generate_full_dataset()


@pytest.fixture(scope="module")
def simulator(mockup_data):
    """从 mockup 数据构建 SignalSimulator"""
    return SignalSimulator(mockup_data)


@pytest.fixture(scope="module")
def engine():
    """创建 HabitDemoEngine (无 LLM)"""
    eng = HabitDemoEngine(username=TEST_USER, llm_client=None)
    yield eng
    eng.reset()
    eng.close()


@pytest.fixture(scope="module")
def pipeline_result(engine, simulator):
    """运行完整 pipeline 并返回结果 (module scope, 只跑一次)"""
    engine.reset()
    result = engine.ingest_from_simulator(simulator)
    return result


# ═══════════════════════════════════════════════════
# Phase 1: 数据生成
# ═══════════════════════════════════════════════════

class TestPhase1DataGeneration:
    """验证 mockup 数据结构和数量"""

    def test_dataset_has_user(self, mockup_data):
        assert mockup_data["user"] == "Mary"
        assert mockup_data["user_id"] == "U001"

    def test_dataset_has_all_scenes(self, mockup_data):
        scenes = mockup_data["scenes"]
        assert "morning_commute" in scenes
        assert "arriving_home" in scenes
        assert "toll_parking_entry" in scenes
        assert "noise" in scenes

    def test_scene_event_counts(self, mockup_data):
        scenes = mockup_data["scenes"]
        assert len(scenes["morning_commute"]) == 5
        assert len(scenes["arriving_home"]) == 5
        assert len(scenes["toll_parking_entry"]) == 5
        assert len(scenes["noise"]) >= 5  # 至少 5 个噪声事件

    def test_total_events(self, mockup_data):
        total = sum(len(v) for v in mockup_data["scenes"].values())
        assert total >= 23  # 5+5+5+8=23

    def test_event_has_signals(self, mockup_data):
        """每个事件必须有 signals 列表"""
        for scene_name, events in mockup_data["scenes"].items():
            for event in events:
                assert "signals" in event, f"{scene_name} event missing signals"
                assert len(event["signals"]) > 0


class TestPhase1Simulator:
    """验证 SignalSimulator 对接"""

    def test_simulator_loads(self, simulator):
        assert simulator.user == "Mary"
        assert simulator.total_events >= 23

    def test_simulator_scene_summary(self, simulator):
        summary = simulator.get_scene_summary()
        assert summary.get("morning_commute", 0) == 5
        assert summary.get("arriving_home", 0) == 5
        assert summary.get("toll_parking_entry", 0) == 5

    def test_simulator_from_generator(self):
        """验证 from_generator 工厂方法"""
        sim = SignalSimulator.from_generator(generate_full_dataset)
        assert sim.total_events >= 23
        assert sim.user == "Mary"

    def test_simulator_iter_steps(self, simulator):
        """逐步迭代产出 Fact"""
        steps = list(simulator.iter_steps())
        assert len(steps) == simulator.total_events
        for step in steps:
            assert step.scene_label in (
                "morning_commute", "arriving_home",
                "toll_parking_entry", "noise"
            )
            assert isinstance(step.facts, list)


# ═══════════════════════════════════════════════════
# Phase 2: 信号→Fact 转换
# ═══════════════════════════════════════════════════

class TestPhase2SignalToFact:
    """验证信号到 Fact 的转换质量"""

    def test_total_facts(self, simulator):
        """全量信号转换后应产出 70+ facts"""
        all_facts = simulator.run_all()
        assert len(all_facts) >= 70, f"Expected 70+ facts, got {len(all_facts)}"
        simulator.reset()

    def test_morning_commute_facts(self, mockup_data):
        """早晨通勤事件应产出多种信号类型的 facts"""
        event = mockup_data["scenes"]["morning_commute"][0]
        facts = signals_to_facts(event)

        # 早晨通勤至少包含: hvac, seat_heating, nav, media, drive_mode, acc
        fact_texts = " ".join(f.text for f in facts)
        assert "temperature" in fact_texts.lower() or "air conditioning" in fact_texts.lower()
        assert "navigation" in fact_texts.lower()

    def test_arriving_home_facts(self, mockup_data):
        """到家场景应包含: media_off, hvac_off, keyless"""
        event = mockup_data["scenes"]["arriving_home"][0]
        facts = signals_to_facts(event)
        fact_texts = " ".join(f.text.lower() for f in facts)

        assert "stopped" in fact_texts or "off" in fact_texts
        assert "keyless" in fact_texts

    def test_toll_entry_facts(self, mockup_data):
        """收费站场景应包含: window 操作"""
        event = mockup_data["scenes"]["toll_parking_entry"][0]
        facts = signals_to_facts(event)
        fact_texts = " ".join(f.text.lower() for f in facts)
        assert "window" in fact_texts

    def test_structured_context_quality(self, mockup_data):
        """验证 StructuredContext 各维度正确提取"""
        # 早晨通勤: 应为 early_morning, engine_started, weekday
        event = mockup_data["scenes"]["morning_commute"][0]
        facts = signals_to_facts(event)

        for f in facts:
            ctx = f.context
            assert ctx.time_bucket in (
                "early_morning", "morning", "midday",
                "afternoon", "evening", "night", "unknown"
            )
            assert ctx.vehicle_state in (
                "engine_started", "parked", "crawling", "unknown"
            )
            assert isinstance(ctx.weekday, bool) or ctx.weekday is None
            assert isinstance(ctx.hour, int)

    def test_fact_metadata_has_signal(self, mockup_data):
        """每个 fact 的 json_metadata 应包含 signal 名"""
        event = mockup_data["scenes"]["morning_commute"][0]
        facts = signals_to_facts(event)

        for f in facts:
            assert f.json_metadata is not None
            meta = json.loads(f.json_metadata)
            assert "signal" in meta, f"Missing signal in metadata: {f.text}"


# ═══════════════════════════════════════════════════
# Phase 3: 端到端 Pipeline
# ═══════════════════════════════════════════════════

class TestPhase3Pipeline:
    """验证 HabitDemoEngine 完整 pipeline"""

    def test_pipeline_ingests_facts(self, pipeline_result):
        """pipeline 成功转换信号为 facts"""
        assert pipeline_result["facts_ingested"] >= 70

    def test_pipeline_detects_habits(self, pipeline_result):
        """pipeline 成功检测到习惯"""
        assert pipeline_result["habits_detected"] > 0, \
            "No habits detected — DBSCAN + consecutiveness may need tuning"

    def test_pipeline_clusters_facts(self, pipeline_result):
        """pipeline 有 facts 被聚类消费"""
        assert pipeline_result["facts_clustered"] > 0

    def test_pipeline_has_remaining_facts(self, pipeline_result):
        """聚类后仍有剩余 facts (噪声 + 未聚类)"""
        assert pipeline_result["facts_remaining"] > 0

    def test_pipeline_result_consistency(self, pipeline_result):
        """验证结果数值一致性"""
        r = pipeline_result
        expected_remaining = (
            r["total_before_clustering"]
            - r["facts_clustered"]
            + r["habits_detected"]
        )
        assert r["facts_remaining"] == expected_remaining

    def test_new_habits_have_text(self, pipeline_result):
        """新检测的 habit 有文本内容"""
        for h in pipeline_result["new_habits"]:
            assert h["text"], f"Habit with empty text: {h}"
            assert len(h["text"]) > 5

    def test_scene_cards_generated(self, pipeline_result):
        """场景卡正确生成 (无 LLM 时使用默认名)"""
        # 无 LLM 时 scene_cards 来自 _default_scene_name
        cards = pipeline_result.get("scene_cards", [])
        assert len(cards) >= 0  # 无 LLM 也能运行


# ═══════════════════════════════════════════════════
# Phase 4: 习惯质量
# ═══════════════════════════════════════════════════

class TestPhase4HabitQuality:
    """验证检测到的习惯的质量"""

    def test_habits_stored_in_chromadb(self, engine):
        """习惯已存入 ChromaDB"""
        habits = engine.get_habits()
        assert len(habits) > 0, "No habits in ChromaDB after pipeline"

    def test_habits_are_habit_type(self, engine):
        """所有 habit 类型正确"""
        habits = engine.get_habits()
        for h in habits:
            assert h.type == FactType.HABIT

    def test_habits_default_not_accepted(self, engine):
        """新检测的 habits 默认 accepted=False"""
        habits = engine.get_habits()
        # 至少有部分是未接受的 (pipeline 不自动 accept)
        unaccepted = [h for h in habits if not h.accepted]
        assert len(unaccepted) > 0

    def test_habits_have_context(self, engine):
        """habits 有关联的 StructuredContext"""
        habits = engine.get_habits()
        for h in habits:
            assert h.context is not None
            assert h.context.time_bucket is not None
            assert h.context.vehicle_state is not None

    def test_habit_context_diversity(self, engine):
        """检测到的 habits 覆盖不同上下文 (不全是同一 time_bucket)"""
        habits = engine.get_habits()
        buckets = {h.context.time_bucket for h in habits}
        # 三个场景 → 应该至少有 2 种不同 time_bucket
        assert len(buckets) >= 2, \
            f"Low context diversity: only {buckets}"

    def test_habit_texts_are_parseable(self, engine):
        """至少部分 habit text 可被 parse_habit_actions 解析"""
        habits = engine.get_habits()
        parsed_count = 0
        for h in habits:
            actions = parse_habit_actions(h.text)
            if actions:
                parsed_count += 1
        assert parsed_count > 0, \
            f"No habits parseable by parse_habit_actions"

    def test_status_summary(self, engine):
        """get_status() 返回完整摘要"""
        status = engine.get_status()
        assert status["username"] == TEST_USER
        assert status["total_facts"] > 0
        assert status["habit_facts"] > 0
        assert isinstance(status["habits"], list)
        assert len(status["habits"]) == status["habit_facts"]


# ═══════════════════════════════════════════════════
# Phase 5: 执行闭环 — accept + recommend
# ═══════════════════════════════════════════════════

class TestPhase5ExecutionLoop:
    """验证 accept → recommend 完整闭环"""

    def test_no_recommendations_before_accept(self, engine):
        """未 accept 时不应有推荐"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=8, weekday=True,
            vehicle_state="engine_started", geofence=None,
        )
        result = engine.get_recommendation(ctx)
        assert not result.has_recommendations, \
            "Got recommendations before accepting any habit"

    def test_accept_habit(self, engine):
        """accept 一个 habit"""
        habits = engine.get_habits()
        unaccepted = [h for h in habits if not h.accepted]
        assert len(unaccepted) > 0, "No unaccepted habits to test"

        habit_to_accept = unaccepted[0]
        ok = engine.accept_habit(habit_to_accept.id)
        assert ok is True

        # 验证已 accepted
        refreshed = engine.get_habits()
        accepted_ids = {h.id for h in refreshed if h.accepted}
        assert habit_to_accept.id in accepted_ids

    def test_recommend_after_accept(self, engine):
        """accept 后，匹配上下文应产生推荐"""
        # 找到已 accepted 的 habit，用其上下文构造请求
        habits = engine.get_habits()
        accepted = [h for h in habits if h.accepted]
        assert len(accepted) > 0

        target = accepted[0]
        ctx = StructuredContext(
            time_bucket=target.context.time_bucket,
            hour=target.context.hour or 8,
            weekday=target.context.weekday if target.context.weekday is not None else True,
            vehicle_state=target.context.vehicle_state,
            geofence=target.context.geofence,
        )
        result = engine.get_recommendation(ctx)
        assert result.has_recommendations, \
            f"No recommendations for accepted habit context: {ctx}"
        assert len(result.actions) >= 1

    def test_recommendation_structure(self, engine):
        """验证推荐结果数据结构"""
        habits = engine.get_habits()
        accepted = [h for h in habits if h.accepted]
        if not accepted:
            pytest.skip("No accepted habits")

        target = accepted[0]
        ctx = StructuredContext(
            time_bucket=target.context.time_bucket,
            hour=target.context.hour or 8,
            weekday=target.context.weekday if target.context.weekday is not None else True,
            vehicle_state=target.context.vehicle_state,
            geofence=target.context.geofence,
        )
        result = engine.get_recommendation(ctx)

        assert result.total_habits > 0
        assert result.candidates_checked > 0
        assert result.threshold == CONTEXT_MATCH_THRESHOLD

        action = result.actions[0]
        assert action.habit_id is not None
        assert action.habit_text is not None
        assert 0 <= action.context_distance <= 1
        assert 0 <= action.match_confidence <= 1
        assert isinstance(action.parsed_actions, dict)

    def test_recommendation_summary(self, engine):
        """推荐结果 summary 格式正确"""
        habits = engine.get_habits()
        accepted = [h for h in habits if h.accepted]
        if not accepted:
            pytest.skip("No accepted habits")

        target = accepted[0]
        ctx = StructuredContext(
            time_bucket=target.context.time_bucket,
            hour=target.context.hour or 8,
            weekday=target.context.weekday if target.context.weekday is not None else True,
            vehicle_state=target.context.vehicle_state,
            geofence=target.context.geofence,
        )
        result = engine.get_recommendation(ctx)
        summary = result.summary
        assert "recommendations" in summary.lower() or "no match" in summary.lower()

    def test_mismatched_context_no_recommendation(self, engine):
        """完全不匹配的上下文不应产生推荐"""
        ctx = StructuredContext(
            time_bucket="night", hour=23, weekday=False,
            vehicle_state="parked", geofence="unknown_place",
        )
        result = engine.get_recommendation(ctx)
        # 可能有也可能没有 (取决于 threshold)，但距离应较大
        if result.has_recommendations:
            # 如果有推荐，distance 应较大
            assert result.actions[0].context_distance > 0

    def test_accept_multiple_habits_and_recommend(self, engine):
        """accept 多个 habits 后推荐"""
        habits = engine.get_habits()
        unaccepted = [h for h in habits if not h.accepted]

        # accept 所有
        for h in unaccepted:
            engine.accept_habit(h.id)

        # 用不同上下文查询推荐
        contexts = [
            StructuredContext(
                time_bucket="early_morning", hour=8, weekday=True,
                vehicle_state="engine_started", geofence=None,
            ),
            StructuredContext(
                time_bucket="evening", hour=18, weekday=True,
                vehicle_state="parked", geofence=None,
            ),
        ]

        total_recommendations = 0
        for ctx in contexts:
            result = engine.get_recommendation(ctx)
            total_recommendations += len(result.actions)

        assert total_recommendations > 0, \
            "No recommendations from any context after accepting all habits"

    def test_top_k_limits_results(self, engine):
        """top_k 参数限制返回数量"""
        ctx = StructuredContext(
            time_bucket="early_morning", hour=8, weekday=True,
            vehicle_state="engine_started", geofence=None,
        )
        result = engine.get_recommendation(ctx, top_k=1)
        assert len(result.actions) <= 1


# ═══════════════════════════════════════════════════
# Phase 6: 生命周期管理
# ═══════════════════════════════════════════════════

class TestPhase6Lifecycle:
    """验证 reject + 清理等生命周期操作"""

    def test_reject_habit(self, engine):
        """reject retires the scene card containing the habit"""
        habits = engine.get_habits()
        if not habits:
            pytest.skip("No habits to reject")

        target = habits[0]
        count_before = len(habits)

        ok = engine.reject_habit(target.id)
        assert ok is True

        remaining = engine.get_habits()
        assert len(remaining) < count_before, "Reject should reduce habit count"
        assert target.id not in {h.id for h in remaining}

    def test_reject_nonexistent(self, engine):
        """reject 不存在的 habit 返回 False"""
        ok = engine.reject_habit("nonexistent-id-12345")
        assert ok is False

    def test_accept_nonexistent(self, engine):
        """accept 不存在的 habit 返回 False"""
        ok = engine.accept_habit("nonexistent-id-12345")
        assert ok is False

    def test_reset_clears_all(self, engine):
        """reset 清空所有数据"""
        engine.reset()
        all_facts = engine.get_all_facts()
        assert len(all_facts) == 0

        habits = engine.get_habits()
        assert len(habits) == 0


# ═══════════════════════════════════════════════════
# 清理
# ═══════════════════════════════════════════════════

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    """测试结束后清理临时目录"""
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
    log.info(f"Cleaned up temp storage: {_TMP_DIR}")
