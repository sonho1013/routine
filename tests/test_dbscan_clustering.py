"""
Hybrid DBSCAN 聚类单元测试 + 集成测试

覆盖:
  ── context_distance 距离函数 ──
  - context_distance(): 全维度加权计算
  - _time_bucket_distance(): 有序距离 + unknown中性
  - _categorical_distance(): 精确匹配 + unknown中性
  - _weekday_distance(): 布尔匹配 + None中性
  - 维度权重验证: 0.35 + 0.30 + 0.25 + 0.10 = 1.0

  ── Hybrid 距离矩阵 ──
  - _compute_hybrid_distance_matrix(): 文本cosine + 上下文 hybrid 融合
  - HYBRID_ALPHA 权重效果
  - 对角线为零 + 对称性

  ── DBSCAN 聚类 ──
  - _get_fact_clusters(): FactType过滤 + hybrid聚类
  - 同上下文同文本 → 聚成一簇
  - 同文本不同上下文 → 被上下文通道区分
  - 噪声点过滤(label=-1)
  - 聚类置信度附加 (cluster["confidence"])

  ── detect_habits 端到端 ──
  - 无LLM: 取首条text作为habit
  - Mock LLM: reword + confidence解析
  - cluster_filter集成 (连续性过滤)
  - 空输入 / 不足2条 / durability过滤
  - dominant_context写入habit fact
  - clustering_confidence写入json_metadata

  ── _dominant_context ──
  - 众数提取: time_bucket, vehicle_state, geofence, weekday
  - 平均hour计算
  - 空列表处理

  ── _get_habit_reword LLM解析 ──
  - 正常CSV解析
  - NO_HABIT拒绝
  - 低confidence过滤
  - 解析异常回退

  ── 真实 mockup 集成 ──
  - mockup数据 → 聚类数量 + 场景纯度验证
"""
import os
import sys
import json
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from panoramix_core.clustering.habits_detector import (
    HabitsDetector,
    context_distance,
    _time_bucket_distance,
    _categorical_distance,
    _weekday_distance,
    _dominant_context,
    TIME_BUCKETS,
    VEHICLE_STATES,
)
from panoramix_core.config import (
    HYBRID_ALPHA,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    HABIT_REWORD_CONFIDENCE_THRESHOLD,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _ctx(time_bucket="early_morning", vehicle_state="engine_started",
         geofence=None, weekday=True, hour=8,
         poi_type="home", wiper_state="off", temp_bucket="mild",
         window_state="closed", door_lock="locked", approach_unlock="enabled"):
    return StructuredContext(
        time_bucket=time_bucket, hour=hour, weekday=weekday,
        vehicle_state=vehicle_state, geofence=geofence,
        poi_type=poi_type, wiper_state=wiper_state, temp_bucket=temp_bucket,
        window_state=window_state, door_lock=door_lock,
        approach_unlock=approach_unlock,
    )


def _fact_with_emb(text, embedding, time_bucket="early_morning",
                   vehicle_state="engine_started", geofence=None,
                   weekday=True, hour=8, fact_type=FactType.PREF,
                   durability=FactDurability.LONG_TERM, day=0, **ctx_kwargs):
    """构造 {"fact": Fact, "embedding": list} 结构"""
    f = Fact(
        text=text,
        type=fact_type,
        durability=durability,
        time_stamp=datetime(2025, 10, 6 + day, hour, 15),
        source=FactSources.SIGNAL,
        context=_ctx(time_bucket, vehicle_state, geofence, weekday, hour,
                     **ctx_kwargs),
    )
    return {"fact": f, "embedding": embedding}


def _random_emb(dim=1536, seed=None):
    """生成随机归一化 embedding"""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _similar_emb(base, noise_scale=0.001, seed=None):
    """在 base 附近产生微小扰动"""
    rng = np.random.RandomState(seed)
    base_arr = np.array(base, dtype=np.float32)
    noise = rng.randn(len(base)).astype(np.float32) * noise_scale
    result = base_arr + noise
    return (result / np.linalg.norm(result)).tolist()


# ═══════════════════════════════════════════════════
# context_distance 子函数
# ═══════════════════════════════════════════════════

class TestTimeBucketDistance:
    def test_same(self):
        assert _time_bucket_distance("morning", "morning") == 0.0

    def test_adjacent(self):
        # early_morning(0) → morning(1), distance = 1/5 = 0.2
        assert _time_bucket_distance("early_morning", "morning") == pytest.approx(0.2)

    def test_extreme(self):
        # early_morning(0) → night(5), distance = 5/5 = 1.0
        assert _time_bucket_distance("early_morning", "night") == pytest.approx(1.0)

    def test_unknown_a(self):
        assert _time_bucket_distance("unknown", "morning") == 0.5

    def test_unknown_b(self):
        assert _time_bucket_distance("morning", "unknown") == 0.5

    def test_both_unknown(self):
        assert _time_bucket_distance("unknown", "unknown") == 0.5

    def test_invalid_bucket(self):
        assert _time_bucket_distance("noon", "morning") == 0.5

    @pytest.mark.parametrize("a,b,expected", [
        ("early_morning", "afternoon", 3/5),  # 0→3
        ("morning", "evening", 3/5),           # 1→4
        ("midday", "night", 3/5),              # 2→5
        ("afternoon", "evening", 1/5),         # 3→4
    ])
    def test_ordinal_distances(self, a, b, expected):
        assert _time_bucket_distance(a, b) == pytest.approx(expected, abs=0.01)


class TestCategoricalDistance:
    def test_same(self):
        assert _categorical_distance("parked", "parked") == 0.0

    def test_different(self):
        assert _categorical_distance("parked", "engine_started") == 1.0

    def test_none_a(self):
        assert _categorical_distance(None, "parked") == 0.5

    def test_unknown(self):
        assert _categorical_distance("unknown", "parked") == 0.5

    def test_both_none(self):
        assert _categorical_distance(None, None) == 0.5


class TestWeekdayDistance:
    def test_same_weekday(self):
        assert _weekday_distance(True, True) == 0.0

    def test_same_weekend(self):
        assert _weekday_distance(False, False) == 0.0

    def test_different(self):
        assert _weekday_distance(True, False) == 1.0

    def test_none_a(self):
        assert _weekday_distance(None, True) == 0.5

    def test_both_none(self):
        assert _weekday_distance(None, None) == 0.5


# ═══════════════════════════════════════════════════
# context_distance 综合
# ═══════════════════════════════════════════════════

class TestContextDistance:
    def test_identical_all_known(self):
        """所有维度已知 → 距离 = 0"""
        ctx = _ctx(geofence="home")
        assert context_distance(ctx, ctx) == 0.0

    def test_identical_with_unknown(self):
        """geofence=None → unknown 贡献 0.5 → 0.13*0.5 = 0.065"""
        ctx = _ctx()
        assert context_distance(ctx, ctx) == pytest.approx(0.065, abs=0.001)

    def test_completely_different(self):
        a = _ctx(time_bucket="early_morning", vehicle_state="engine_started",
                 geofence="home", weekday=True,
                 poi_type="home", wiper_state="off", temp_bucket="cold",
                 window_state="closed", door_lock="locked",
                 approach_unlock="enabled")
        b = _ctx(time_bucket="night", vehicle_state="parked",
                 geofence="workplace", weekday=False,
                 poi_type="park", wiper_state="max", temp_bucket="hot",
                 window_state="open", door_lock="unlocked",
                 approach_unlock="disabled")
        d = context_distance(a, b)
        # 所有 10 维全部 1.0，权重之和 = 1.0
        assert d == pytest.approx(1.0)

    def test_only_time_differs(self):
        a = _ctx(time_bucket="early_morning")
        b = _ctx(time_bucket="evening")
        d = context_distance(a, b)
        # time_bucket: 0.22 * (4/5) = 0.176
        # geofence: both None → 0.13 * 0.5 = 0.065
        # 其余均已知且相同 → 0
        expected = 0.22 * (4/5) + 0.13 * 0.5  # 0.176 + 0.065 = 0.241
        assert d == pytest.approx(expected, abs=0.01)

    def test_only_vehicle_differs(self):
        a = _ctx(vehicle_state="engine_started", geofence="home")
        b = _ctx(vehicle_state="parked", geofence="home")
        d = context_distance(a, b)
        # vehicle: 0.16 * 1.0 = 0.16, 其余同
        assert d == pytest.approx(0.16, abs=0.01)

    def test_weights_sum_to_one(self):
        """10 维权重之和 = 1.0（trigger list 2.xlsx 对齐后）"""
        weights = [0.22, 0.16, 0.13, 0.10, 0.09, 0.07, 0.06, 0.06, 0.06, 0.05]
        assert sum(weights) == pytest.approx(1.0)

    def test_range_zero_to_one(self):
        """context_distance 输出范围 [0, 1]"""
        for _ in range(50):
            a = _ctx(
                time_bucket=np.random.choice(TIME_BUCKETS),
                vehicle_state=np.random.choice(VEHICLE_STATES),
                geofence=np.random.choice(["home", "work", None]),
                weekday=np.random.choice([True, False, None]),
            )
            b = _ctx(
                time_bucket=np.random.choice(TIME_BUCKETS),
                vehicle_state=np.random.choice(VEHICLE_STATES),
                geofence=np.random.choice(["home", "work", None]),
                weekday=np.random.choice([True, False, None]),
            )
            d = context_distance(a, b)
            assert 0.0 <= d <= 1.0, f"context_distance out of range: {d}"

    def test_symmetry(self):
        a = _ctx(time_bucket="morning", vehicle_state="parked", geofence="home")
        b = _ctx(time_bucket="evening", vehicle_state="crawling", geofence=None)
        assert context_distance(a, b) == pytest.approx(context_distance(b, a))


# ═══════════════════════════════════════════════════
# Hybrid 距离矩阵
# ═══════════════════════════════════════════════════

class TestHybridDistanceMatrix:
    """测试 _compute_hybrid_distance_matrix"""

    def _make_detector(self):
        return HabitsDetector(llm_client=None)

    def test_diagonal_zero(self):
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", base, time_bucket="morning"),
            _fact_with_emb("set AC to 23", _similar_emb(base, seed=1), time_bucket="morning"),
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=2), time_bucket="morning"),
        ]
        det = self._make_detector()
        dist = det._compute_hybrid_distance_matrix(items)
        np.testing.assert_array_almost_equal(np.diag(dist), 0.0)

    def test_symmetry(self):
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", base, time_bucket="morning"),
            _fact_with_emb("close windows", _random_emb(seed=99), time_bucket="evening"),
        ]
        det = self._make_detector()
        dist = det._compute_hybrid_distance_matrix(items)
        assert dist[0, 1] == pytest.approx(dist[1, 0])

    def test_same_text_diff_context_nonzero(self):
        """相同文本不同上下文 → 距离 > 0 (20% context 通道起作用)"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", base, time_bucket="morning",
                           vehicle_state="engine_started"),
            _fact_with_emb("set AC to 22", base, time_bucket="evening",
                           vehicle_state="parked"),
        ]
        det = self._make_detector()
        dist = det._compute_hybrid_distance_matrix(items)
        # text_dist = 0 (identical embedding), ctx_dist > 0
        assert dist[0, 1] > 0

    def test_alpha_weight(self):
        """验证 HYBRID_ALPHA 权重比例"""
        # 不同text + 相同context
        base1 = _random_emb(seed=10)
        base2 = _random_emb(seed=20)
        items = [
            _fact_with_emb("set AC to 22", base1, time_bucket="morning",
                           vehicle_state="engine_started", geofence="home", weekday=True),
            _fact_with_emb("close windows", base2, time_bucket="morning",
                           vehicle_state="engine_started", geofence="home", weekday=True),
        ]
        det = self._make_detector()
        dist = det._compute_hybrid_distance_matrix(items)
        # ctx_dist should be 0 (same context, but geofence same → 0, weekday same → 0)
        # hybrid = ALPHA * text_dist + (1-ALPHA) * 0 = ALPHA * text_dist
        emb1 = np.array(base1)
        emb2 = np.array(base2)
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        expected_text_dist = 1.0 - cos_sim
        expected_hybrid = HYBRID_ALPHA * expected_text_dist
        assert dist[0, 1] == pytest.approx(expected_hybrid, abs=0.01)


# ═══════════════════════════════════════════════════
# _get_fact_clusters
# ═══════════════════════════════════════════════════

class TestGetFactClusters:
    def _make_detector(self):
        return HabitsDetector(llm_client=None)

    def test_same_context_clusters_together(self):
        """5 个相同上下文+相似文本 → 聚成一簇"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb(
                "set AC to 22", _similar_emb(base, seed=i),
                time_bucket="early_morning", vehicle_state="engine_started",
                day=i
            )
            for i in range(5)
        ]
        det = self._make_detector()
        clusters = det._get_fact_clusters(items, "PREF")
        assert len(clusters) >= 1
        # 至少一个 cluster 包含所有 5 个
        max_size = max(len(c["items"]) for c in clusters)
        assert max_size == 5

    def test_different_context_separates(self):
        """相同文本 + 不同上下文 → 被分开"""
        base = _random_emb(seed=42)
        morning_items = [
            _fact_with_emb(
                "set AC to 22", _similar_emb(base, seed=i),
                time_bucket="early_morning", vehicle_state="engine_started",
                geofence="home", weekday=True, day=i,
                poi_type="home", wiper_state="off",
            )
            for i in range(5)
        ]
        evening_items = [
            _fact_with_emb(
                "set AC to 22", _similar_emb(base, seed=i + 100),
                time_bucket="evening", vehicle_state="parked",
                geofence="workplace", weekday=True, hour=18, day=i,
                poi_type="work", wiper_state="high",
            )
            for i in range(5)
        ]
        items = morning_items + evening_items
        det = self._make_detector()
        clusters = det._get_fact_clusters(items, "PREF")
        # 应该至少有 2 个 cluster（上下文不同导致距离增加）
        assert len(clusters) >= 2

    def test_filters_by_fact_type(self):
        """只聚类指定 FactType"""
        base = _random_emb(seed=42)
        pref_items = [
            _fact_with_emb(
                "set AC to 22", _similar_emb(base, seed=i),
                fact_type=FactType.PREF, day=i
            )
            for i in range(5)
        ]
        media_items = [
            _fact_with_emb(
                "play music", _random_emb(seed=i + 200),
                fact_type=FactType.MEDIA, day=i
            )
            for i in range(5)
        ]
        items = pref_items + media_items
        det = self._make_detector()
        pref_clusters = det._get_fact_clusters(items, "PREF")
        media_clusters = det._get_fact_clusters(items, "MEDIA")
        # PREF 应有 cluster, MEDIA 的 text 各不同可能没有
        assert len(pref_clusters) >= 1
        # 验证 PREF cluster 中不含 MEDIA fact
        for c in pref_clusters:
            for item in c["items"]:
                assert item["fact"].type == FactType.PREF

    def test_less_than_two_returns_empty(self):
        """不足 2 条 → 空"""
        base = _random_emb(seed=42)
        items = [_fact_with_emb("set AC to 22", base)]
        det = self._make_detector()
        assert det._get_fact_clusters(items, "PREF") == []

    def test_null_embedding_excluded(self):
        """embedding 为 None 的 fact 被过滤"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        items.append({"fact": Fact(
            text="set AC to 22", type=FactType.PREF,
            durability=FactDurability.LONG_TERM,
            time_stamp=datetime.now(),
        ), "embedding": None})
        det = self._make_detector()
        clusters = det._get_fact_clusters(items, "PREF")
        # None embedding 的 fact 不应出现在任何 cluster
        for c in clusters:
            for item in c["items"]:
                assert item["embedding"] is not None

    def test_cluster_has_confidence(self):
        """每个 cluster 包含 confidence 对象"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        det = self._make_detector()
        clusters = det._get_fact_clusters(items, "PREF")
        for c in clusters:
            assert "confidence" in c
            cc = c["confidence"]
            assert 0.0 <= cc.confidence <= 1.0
            assert cc.cluster_size == len(c["items"])


# ═══════════════════════════════════════════════════
# detect_habits 端到端
# ═══════════════════════════════════════════════════

class TestDetectHabitsNoLLM:
    """无 LLM 模式: 取首条 fact text 作为 habit"""

    def test_basic_detection(self):
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        det = HabitsDetector(llm_client=None)
        habits, ids = det.detect_habits(items)
        assert len(habits) >= 1
        assert len(ids) >= 3  # at least min_samples
        for h in habits:
            assert h.type == FactType.HABIT
            assert h.durability == FactDurability.LONG_TERM
            assert h.source == FactSources.HABITS_DETECTOR

    def test_empty_input(self):
        det = HabitsDetector(llm_client=None)
        habits, ids = det.detect_habits([])
        assert habits == []
        assert ids == []

    def test_dominant_context_assigned(self):
        """habit fact 应继承聚类的主导上下文"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb(
                "set AC to 22", _similar_emb(base, seed=i),
                time_bucket="early_morning", vehicle_state="engine_started",
                weekday=True, day=i
            )
            for i in range(5)
        ]
        det = HabitsDetector(llm_client=None)
        habits, _ = det.detect_habits(items)
        assert len(habits) >= 1
        for h in habits:
            assert h.context.time_bucket == "early_morning"
            assert h.context.vehicle_state == "engine_started"
            assert h.context.weekday is True

    def test_clustering_confidence_in_metadata(self):
        """habit fact 的 json_metadata 包含 clustering_confidence"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        det = HabitsDetector(llm_client=None)
        habits, _ = det.detect_habits(items)
        for h in habits:
            assert h.json_metadata is not None
            meta = json.loads(h.json_metadata)
            assert "clustering_confidence" in meta
            assert 0.0 <= meta["clustering_confidence"] <= 1.0
            assert "clustering_detail" in meta

    def test_cluster_filter_integration(self):
        """cluster_filter 可以拒绝聚类"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        det = HabitsDetector(llm_client=None)

        # reject all clusters
        habits, ids = det.detect_habits(items, cluster_filter=lambda cf, af: False)
        assert len(habits) == 0
        assert len(ids) == 0

        # accept all clusters
        habits2, ids2 = det.detect_habits(items, cluster_filter=lambda cf, af: True)
        assert len(habits2) >= 1


class TestDetectHabitsWithMockLLM:
    """Mock LLM 模式"""

    def test_reword_success(self):
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "set AC around 22 degrees,0.9"

        det = HabitsDetector(llm_client=mock_llm)
        habits, ids = det.detect_habits(items)
        assert len(habits) >= 1
        assert habits[0].text == "set AC around 22 degrees"

    def test_reword_no_habit(self):
        """LLM 返回 NO_HABIT → 该聚类不生成 habit"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("random text", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "NO_HABIT,0.0"

        det = HabitsDetector(llm_client=mock_llm)
        habits, ids = det.detect_habits(items)
        assert len(habits) == 0

    def test_reword_low_confidence(self):
        """LLM confidence < threshold → 过滤"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        mock_llm = MagicMock()
        threshold = HABIT_REWORD_CONFIDENCE_THRESHOLD
        mock_llm.invoke.return_value = f"set AC,{threshold - 0.1}"

        det = HabitsDetector(llm_client=mock_llm)
        habits, _ = det.detect_habits(items)
        assert len(habits) == 0

    def test_reword_parse_error(self):
        """LLM 返回无法解析的输出 → 跳过"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "gibberish without comma"

        det = HabitsDetector(llm_client=mock_llm)
        habits, _ = det.detect_habits(items)
        assert len(habits) == 0

    def test_llm_exception(self):
        """LLM 调用抛异常 → 跳过该聚类"""
        base = _random_emb(seed=42)
        items = [
            _fact_with_emb("set AC to 22", _similar_emb(base, seed=i), day=i)
            for i in range(5)
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")

        det = HabitsDetector(llm_client=mock_llm)
        habits, _ = det.detect_habits(items)
        assert len(habits) == 0


# ═══════════════════════════════════════════════════
# _dominant_context
# ═══════════════════════════════════════════════════

class TestDominantContext:
    def test_uniform(self):
        contexts = [
            _ctx(time_bucket="morning", vehicle_state="parked", hour=9, weekday=True)
            for _ in range(5)
        ]
        dc = _dominant_context(contexts)
        assert dc.time_bucket == "morning"
        assert dc.vehicle_state == "parked"
        assert dc.hour == 9
        assert dc.weekday is True
        assert dc.geofence is None

    def test_majority_vote(self):
        contexts = [
            _ctx(time_bucket="morning", vehicle_state="engine_started"),
            _ctx(time_bucket="morning", vehicle_state="engine_started"),
            _ctx(time_bucket="afternoon", vehicle_state="parked"),
        ]
        dc = _dominant_context(contexts)
        assert dc.time_bucket == "morning"
        assert dc.vehicle_state == "engine_started"

    def test_average_hour(self):
        contexts = [
            _ctx(hour=7), _ctx(hour=8), _ctx(hour=9),
        ]
        dc = _dominant_context(contexts)
        assert dc.hour == 8  # round(24/3) = 8

    def test_geofence_majority(self):
        contexts = [
            _ctx(geofence="home"), _ctx(geofence="home"), _ctx(geofence="work"),
        ]
        dc = _dominant_context(contexts)
        assert dc.geofence == "home"

    def test_geofence_all_none(self):
        contexts = [_ctx(geofence=None) for _ in range(3)]
        dc = _dominant_context(contexts)
        assert dc.geofence is None

    def test_weekday_mixed(self):
        contexts = [
            _ctx(weekday=True), _ctx(weekday=True), _ctx(weekday=False),
        ]
        dc = _dominant_context(contexts)
        assert dc.weekday is True

    def test_weekday_all_none(self):
        contexts = [_ctx(weekday=None) for _ in range(3)]
        dc = _dominant_context(contexts)
        assert dc.weekday is None

    def test_empty_list(self):
        dc = _dominant_context([])
        assert dc.time_bucket == "unknown"
        assert dc.vehicle_state == "unknown"
        assert dc.hour == -1

    def test_negative_hour_excluded(self):
        """hour = -1 的值不参与平均"""
        contexts = [_ctx(hour=8), _ctx(hour=-1), _ctx(hour=10)]
        dc = _dominant_context(contexts)
        assert dc.hour == 9  # (8+10)/2


# ═══════════════════════════════════════════════════
# 配置常量验证
# ═══════════════════════════════════════════════════

class TestConfigConstants:
    def test_hybrid_alpha_range(self):
        assert 0.0 < HYBRID_ALPHA < 1.0

    def test_eps_positive(self):
        assert DBSCAN_EPS > 0

    def test_min_samples_ge_2(self):
        assert DBSCAN_MIN_SAMPLES >= 2

    def test_confidence_threshold_range(self):
        assert 0.0 < HABIT_REWORD_CONFIDENCE_THRESHOLD < 1.0

    def test_time_buckets_ordered(self):
        assert TIME_BUCKETS == [
            "early_morning", "morning", "midday",
            "afternoon", "evening", "night"
        ]


# ═══════════════════════════════════════════════════
# 真实 mockup 集成
# ═══════════════════════════════════════════════════

class TestMockupClustering:
    """使用真实 mockup 数据验证聚类结果"""

    @pytest.fixture
    def mockup_items(self):
        """从 mockup 数据构建 facts + fake embeddings (基于 text hash)"""
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

        # 为每个 fact 生成基于 text hash 的 embedding (相同 text → 相同 embedding)
        text_to_emb = {}
        items = []
        for f in all_facts:
            if f.text not in text_to_emb:
                seed = hash(f.text) % (2**31)
                text_to_emb[f.text] = _random_emb(seed=seed)
            emb = _similar_emb(text_to_emb[f.text], noise_scale=0.001,
                               seed=hash(f.id) % (2**31))
            items.append({"fact": f, "embedding": emb})
        return items

    def test_cluster_count(self, mockup_items):
        """83 facts 应产生多个聚类"""
        det = HabitsDetector(llm_client=None)
        habits, ids = det.detect_habits(mockup_items)
        assert len(habits) >= 5, f"Expected ≥5 habits, got {len(habits)}"
        assert len(ids) >= 20, f"Expected ≥20 clustered facts, got {len(ids)}"

    def test_all_habits_have_context(self, mockup_items):
        """每个 habit 都有非默认 context"""
        det = HabitsDetector(llm_client=None)
        habits, _ = det.detect_habits(mockup_items)
        for h in habits:
            assert h.context.time_bucket != "unknown"
            assert h.context.vehicle_state != "unknown"

    def test_all_habits_have_confidence(self, mockup_items):
        """每个 habit 都有 clustering_confidence"""
        det = HabitsDetector(llm_client=None)
        habits, _ = det.detect_habits(mockup_items)
        for h in habits:
            meta = json.loads(h.json_metadata)
            assert "clustering_confidence" in meta
            assert meta["clustering_confidence"] > 0

    def test_no_duplicate_ids_in_delete_list(self, mockup_items):
        """ids_to_delete 不应有重复"""
        det = HabitsDetector(llm_client=None)
        _, ids = det.detect_habits(mockup_items)
        assert len(ids) == len(set(ids))
