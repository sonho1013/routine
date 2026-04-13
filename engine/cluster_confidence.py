"""
聚类置信度 — 基于 DBSCAN 聚类密度/距离构造等效置信度分数

三个维度加权合成 [0, 1]:
  - cohesion (0.50): 聚类紧凑度 = 1 - mean_intra_dist / eps
    成员间平均距离越小越紧凑，趋近 eps 则退化为松散聚类
  - core_ratio (0.25): 核心点占比 = core_in_cluster / cluster_size
    全核心点 = 聚类稳定；多边界点 = 聚类脆弱
  - size_factor (0.25): 证据充分度 = min(1, cluster_size / (2 * min_samples))
    仅达 min_samples 门槛 = 勉强成聚；2× 以上 = 证据充足

用途:
  - 写入 habit fact 的 json_metadata["clustering_confidence"]
  - 提供不依赖 LLM 的客观聚类质量信号
  - 与 LLM 置信度、场景命名置信度形成多维可信度体系
"""
import logging
from dataclasses import dataclass
from typing import List, Set

import numpy as np

log = logging.getLogger(__name__)

# ── 维度权重 ──
W_COHESION = 0.50
W_CORE_RATIO = 0.25
W_SIZE_FACTOR = 0.25


@dataclass
class ClusterConfidence:
    """单个聚类的置信度评估结果"""
    cluster_id: int
    confidence: float       # 综合置信度 [0, 1]
    cohesion: float         # 紧凑度 [0, 1]
    core_ratio: float       # 核心点占比 [0, 1]
    size_factor: float      # 证据充分度 [0, 1]
    cluster_size: int
    mean_intra_dist: float  # 平均簇内距离
    max_intra_dist: float   # 最大簇内距离

    @property
    def detail(self) -> str:
        return (
            f"confidence={self.confidence:.3f} "
            f"(cohesion={self.cohesion:.3f}, core={self.core_ratio:.3f}, "
            f"size={self.size_factor:.3f}) "
            f"mean_d={self.mean_intra_dist:.4f}, max_d={self.max_intra_dist:.4f}, "
            f"n={self.cluster_size}"
        )


def compute_cluster_confidence(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    core_sample_indices: np.ndarray,
    cluster_id: int,
    eps: float,
    min_samples: int,
) -> ClusterConfidence:
    """
    计算单个聚类的置信度分数。

    Args:
        dist_matrix: N×N 预计算距离矩阵 (hybrid distance)
        labels: DBSCAN 标签数组 (长度 N)
        core_sample_indices: DBSCAN 核心样本的索引数组
        cluster_id: 要评估的聚类 ID (≥ 0)
        eps: DBSCAN eps 参数
        min_samples: DBSCAN min_samples 参数

    Returns:
        ClusterConfidence: 包含综合置信度及各维度分数
    """
    # 获取聚类成员索引
    member_indices = np.where(labels == cluster_id)[0]
    cluster_size = len(member_indices)

    if cluster_size < 2:
        return ClusterConfidence(
            cluster_id=cluster_id,
            confidence=0.0,
            cohesion=0.0,
            core_ratio=0.0,
            size_factor=0.0,
            cluster_size=cluster_size,
            mean_intra_dist=0.0,
            max_intra_dist=0.0,
        )

    # ── 1. Cohesion: 聚类紧凑度 ──
    # 提取聚类子矩阵的上三角距离
    sub_matrix = dist_matrix[np.ix_(member_indices, member_indices)]
    upper_tri = sub_matrix[np.triu_indices(cluster_size, k=1)]
    mean_intra_dist = float(np.mean(upper_tri))
    max_intra_dist = float(np.max(upper_tri))

    # 归一化: 距离 = 0 → cohesion = 1, 距离 = eps → cohesion = 0
    cohesion = max(0.0, 1.0 - mean_intra_dist / eps) if eps > 0 else 0.0

    # ── 2. Core Ratio: 核心点占比 ──
    core_set = set(core_sample_indices.tolist())
    core_in_cluster = sum(1 for idx in member_indices if idx in core_set)
    core_ratio = core_in_cluster / cluster_size

    # ── 3. Size Factor: 证据充分度 ──
    # min_samples 门槛 → 0.5, 2× min_samples → 1.0
    size_factor = min(1.0, cluster_size / (2.0 * min_samples))

    # ── 综合置信度 ──
    confidence = (
        W_COHESION * cohesion
        + W_CORE_RATIO * core_ratio
        + W_SIZE_FACTOR * size_factor
    )
    # 最终值夹到 [0, 1]
    confidence = max(0.0, min(1.0, confidence))

    result = ClusterConfidence(
        cluster_id=cluster_id,
        confidence=round(confidence, 4),
        cohesion=round(cohesion, 4),
        core_ratio=round(core_ratio, 4),
        size_factor=round(size_factor, 4),
        cluster_size=cluster_size,
        mean_intra_dist=round(mean_intra_dist, 6),
        max_intra_dist=round(max_intra_dist, 6),
    )
    log.debug(f"ClusterConfidence [{cluster_id}]: {result.detail}")
    return result


def compute_all_cluster_confidences(
    dist_matrix: np.ndarray,
    labels: np.ndarray,
    core_sample_indices: np.ndarray,
    eps: float,
    min_samples: int,
) -> List[ClusterConfidence]:
    """
    计算所有聚类的置信度分数。

    Returns:
        List[ClusterConfidence]: 按 cluster_id 排序
    """
    cluster_ids = sorted(set(labels) - {-1})
    results = []
    for cid in cluster_ids:
        cc = compute_cluster_confidence(
            dist_matrix, labels, core_sample_indices, cid, eps, min_samples
        )
        results.append(cc)
    return results
