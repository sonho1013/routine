"""
聚类置信度单元测试

覆盖:
  - compute_cluster_confidence(): 紧凑/松散/单成员/全核心/部分核心
  - compute_all_cluster_confidences(): 多聚类批量计算
  - 各维度边界: cohesion、core_ratio、size_factor
  - HabitsDetector 集成: 置信度写入 habit json_metadata
  - 真实 mockup E2E: 通过管线验证分数分布
"""
import os
import sys
import json
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.cluster_confidence import (
    compute_cluster_confidence,
    compute_all_cluster_confidences,
    ClusterConfidence,
    W_COHESION,
    W_CORE_RATIO,
    W_SIZE_FACTOR,
)


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _make_tight_cluster(n=5, noise_dist=0.5, eps=0.10):
    """
    构造一个紧凑聚类 + 噪声点的距离矩阵。

    聚类成员间距离 = 0.02 (远小于 eps)
    噪声点到聚类距离 = noise_dist (远大于 eps)
    """
    total = n + 1  # n cluster members + 1 noise
    dist = np.zeros((total, total))

    # 聚类内距离
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = 0.02
            dist[j, i] = 0.02

    # 噪声点到聚类的距离
    for i in range(n):
        dist[i, n] = noise_dist
        dist[n, i] = noise_dist

    labels = np.array([0] * n + [-1])
    core_indices = np.arange(n)  # 所有聚类成员都是核心点
    return dist, labels, core_indices


def _make_loose_cluster(n=5, intra_dist=0.09, eps=0.10):
    """构造一个松散聚类（距离接近 eps 边界）"""
    total = n
    dist = np.zeros((total, total))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = intra_dist
            dist[j, i] = intra_dist

    labels = np.array([0] * n)
    core_indices = np.arange(n)
    return dist, labels, core_indices


# ═══════════════════════════════════════════════════
# compute_cluster_confidence 基本场景
# ═══════════════════════════════════════════════════

class TestTightCluster:
    """紧凑聚类 → 高置信度"""

    def test_high_confidence(self):
        dist, labels, cores = _make_tight_cluster(n=5, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)

        assert cc.confidence > 0.85
        assert cc.cohesion > 0.75  # 0.02 / 0.10 = 0.2, cohesion = 0.8
        assert cc.core_ratio == 1.0  # 全核心点
        assert cc.size_factor > 0.8  # 5 / (2*3) = 0.833
        assert cc.cluster_size == 5
        assert cc.mean_intra_dist == pytest.approx(0.02, abs=0.001)

    def test_detail_string(self):
        dist, labels, cores = _make_tight_cluster(n=5, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert "confidence=" in cc.detail
        assert "cohesion=" in cc.detail


class TestLooseCluster:
    """松散聚类（距离接近 eps）→ 低 cohesion"""

    def test_low_cohesion(self):
        dist, labels, cores = _make_loose_cluster(n=5, intra_dist=0.09, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)

        # cohesion = 1 - 0.09/0.10 = 0.1
        assert cc.cohesion == pytest.approx(0.1, abs=0.01)
        # confidence = 0.5*0.1 + 0.25*1.0 + 0.25*0.833 ≈ 0.508
        assert cc.confidence < 0.55  # 低 cohesion 拉低整体

    def test_at_eps_boundary(self):
        """距离 = eps → cohesion = 0"""
        dist, labels, cores = _make_loose_cluster(n=3, intra_dist=0.10, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert cc.cohesion == pytest.approx(0.0, abs=0.001)


class TestCoreRatio:
    """核心点占比影响"""

    def test_all_core(self):
        dist, labels, cores = _make_tight_cluster(n=5, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert cc.core_ratio == 1.0

    def test_partial_core(self):
        """5 成员中只有 3 个核心点"""
        dist, labels, _ = _make_tight_cluster(n=5, eps=0.10)
        partial_cores = np.array([0, 1, 2])  # 只有前 3 个是核心点
        cc = compute_cluster_confidence(dist, labels, partial_cores, 0, eps=0.10, min_samples=3)
        assert cc.core_ratio == pytest.approx(0.6, abs=0.01)

    def test_no_core_impossible_but_handled(self):
        """理论上不可能但防御性处理"""
        dist, labels, _ = _make_tight_cluster(n=3, eps=0.10)
        no_cores = np.array([])
        cc = compute_cluster_confidence(dist, labels, no_cores, 0, eps=0.10, min_samples=3)
        assert cc.core_ratio == 0.0


class TestSizeFactor:
    """证据充分度"""

    def test_minimal_cluster(self):
        """size = min_samples → factor = 0.5"""
        dist, labels, cores = _make_tight_cluster(n=3, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert cc.size_factor == pytest.approx(0.5, abs=0.01)

    def test_double_min_samples(self):
        """size = 2*min_samples → factor = 1.0"""
        dist, labels, cores = _make_tight_cluster(n=6, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert cc.size_factor == pytest.approx(1.0, abs=0.01)

    def test_large_cluster_capped(self):
        """size > 2*min_samples → factor capped at 1.0"""
        dist, labels, cores = _make_tight_cluster(n=10, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert cc.size_factor == 1.0


class TestEdgeCases:
    """边界情况"""

    def test_single_member(self):
        """单成员聚类 → confidence = 0"""
        dist = np.array([[0.0]])
        labels = np.array([0])
        cores = np.array([0])
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert cc.confidence == 0.0
        assert cc.cluster_size == 1

    def test_two_members(self):
        """两成员聚类"""
        dist = np.array([[0.0, 0.03], [0.03, 0.0]])
        labels = np.array([0, 0])
        cores = np.array([0, 1])
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=2)
        assert cc.cluster_size == 2
        assert cc.mean_intra_dist == pytest.approx(0.03, abs=0.001)
        assert cc.confidence > 0.0

    def test_zero_eps(self):
        """eps = 0 → cohesion = 0"""
        dist = np.array([[0.0, 0.01], [0.01, 0.0]])
        labels = np.array([0, 0])
        cores = np.array([0, 1])
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.0, min_samples=2)
        assert cc.cohesion == 0.0

    def test_confidence_bounded(self):
        """置信度始终在 [0, 1]"""
        dist, labels, cores = _make_tight_cluster(n=10, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert 0.0 <= cc.confidence <= 1.0


class TestWeightSum:
    """验证权重和为 1"""

    def test_weights_sum_to_one(self):
        assert W_COHESION + W_CORE_RATIO + W_SIZE_FACTOR == pytest.approx(1.0)


# ═══════════════════════════════════════════════════
# compute_all_cluster_confidences
# ═══════════════════════════════════════════════════

class TestComputeAll:
    def test_two_clusters_plus_noise(self):
        """两个聚类 + 噪声点"""
        # cluster 0: indices 0,1,2 (tight)
        # cluster 1: indices 3,4,5 (loose)
        # noise: index 6
        n = 7
        dist = np.ones((n, n)) * 0.5
        np.fill_diagonal(dist, 0)

        # cluster 0: tight
        for i in range(3):
            for j in range(i + 1, 3):
                dist[i, j] = 0.02
                dist[j, i] = 0.02

        # cluster 1: loose
        for i in range(3, 6):
            for j in range(i + 1, 6):
                dist[i, j] = 0.08
                dist[j, i] = 0.08

        labels = np.array([0, 0, 0, 1, 1, 1, -1])
        cores = np.array([0, 1, 2, 3, 4, 5])

        results = compute_all_cluster_confidences(dist, labels, cores, eps=0.10, min_samples=3)
        assert len(results) == 2
        assert results[0].cluster_id == 0
        assert results[1].cluster_id == 1
        # tight cluster should have higher confidence
        assert results[0].confidence > results[1].confidence

    def test_no_clusters(self):
        """全噪声 → 空列表"""
        dist = np.array([[0.0, 0.5], [0.5, 0.0]])
        labels = np.array([-1, -1])
        cores = np.array([])
        results = compute_all_cluster_confidences(dist, labels, cores, eps=0.10, min_samples=3)
        assert results == []


# ═══════════════════════════════════════════════════
# 置信度分布合理性
# ═══════════════════════════════════════════════════

class TestConfidenceDistribution:
    """验证不同紧凑度产生合理的置信度梯度"""

    @pytest.mark.parametrize("intra_dist,expected_min,expected_max", [
        (0.01, 0.85, 1.0),   # 非常紧凑
        (0.04, 0.65, 0.85),  # 中等紧凑
        (0.08, 0.45, 0.60),  # 松散 (core=1.0, size=0.83 still boost)
        (0.10, 0.40, 0.50),  # eps 边界 (cohesion=0, core+size still ~0.46)
    ])
    def test_gradient(self, intra_dist, expected_min, expected_max):
        dist, labels, cores = _make_loose_cluster(n=5, intra_dist=intra_dist, eps=0.10)
        cc = compute_cluster_confidence(dist, labels, cores, 0, eps=0.10, min_samples=3)
        assert expected_min <= cc.confidence <= expected_max, (
            f"intra_dist={intra_dist}: confidence={cc.confidence:.3f} "
            f"not in [{expected_min}, {expected_max}]"
        )


# ═══════════════════════════════════════════════════
# HabitsDetector 集成验证
# ═══════════════════════════════════════════════════

class TestHabitsDetectorIntegration:
    """验证 HabitsDetector 产出的 habit fact 包含 clustering_confidence"""

    def test_confidence_in_json_metadata(self):
        """使用 HabitsDetector（无 LLM）确认 confidence 写入"""
        from panoramix_core.clustering.habits_detector import HabitsDetector
        from panoramix_core.models.fact import Fact, StructuredContext
        from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources
        from datetime import datetime

        # 构造 5 个语义相同的 PREF facts + fake embeddings
        facts_with_emb = []
        base_emb = np.random.randn(1536).astype(np.float32)
        base_emb = base_emb / np.linalg.norm(base_emb)

        for i in range(5):
            f = Fact(
                text=f"set air conditioning temperature to 22 degrees",
                type=FactType.PREF,
                durability=FactDurability.LONG_TERM,
                time_stamp=datetime(2025, 10, 6 + i, 8, 15),
                source=FactSources.SIGNAL,
                context=StructuredContext(
                    time_bucket="early_morning",
                    hour=8,
                    weekday=True,
                    vehicle_state="engine_started",
                ),
            )
            # 微小扰动保持高相似度
            noise = np.random.randn(1536).astype(np.float32) * 0.001
            emb = (base_emb + noise).tolist()
            facts_with_emb.append({"fact": f, "embedding": emb})

        detector = HabitsDetector(llm_client=None)
        habits, ids_to_delete = detector.detect_habits(facts_with_emb)

        assert len(habits) >= 1
        for h in habits:
            assert h.json_metadata is not None
            meta = json.loads(h.json_metadata)
            assert "clustering_confidence" in meta
            assert 0.0 <= meta["clustering_confidence"] <= 1.0
            assert "clustering_detail" in meta
            detail = meta["clustering_detail"]
            assert "cohesion" in detail
            assert "core_ratio" in detail
            assert "size_factor" in detail

    def test_scene_preserves_clustering_confidence(self):
        """场景命名后 clustering_confidence 仍保留"""
        from engine.scene_card import _attach_scene_to_habit
        from panoramix_core.models.fact import Fact, StructuredContext
        from panoramix_core.models.fact_enums import FactType, FactDurability

        h = Fact(
            text="set AC to 22",
            type=FactType.HABIT,
            durability=FactDurability.LONG_TERM,
            context=StructuredContext(time_bucket="early_morning", vehicle_state="engine_started"),
            json_metadata=json.dumps({
                "clustering_confidence": 0.85,
                "clustering_detail": {"cohesion": 0.8, "core_ratio": 1.0, "size_factor": 0.83},
            }),
        )

        _attach_scene_to_habit(h, "Morning Commute", 0.95)
        meta = json.loads(h.json_metadata)

        # 场景信息已添加
        assert meta["scene_name"] == "Morning Commute"
        assert meta["scene_confidence"] == 0.95
        # 聚类置信度仍保留
        assert meta["clustering_confidence"] == 0.85
        assert meta["clustering_detail"]["cohesion"] == 0.8
