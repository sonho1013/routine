"""
连续性验证 — DBSCAN 聚类后的时序连续性校验

DBSCAN 只看语义相似度，不看时间顺序。PRD 要求"连续 N 次相同行为"才算习惯。
本模块在聚类之后进行补充校验：验证聚类中的 Fact 是否来自最近连续的同类行为触发。

示例:
  Day1-5 每天早晨设置空调 22°C → DBSCAN 聚出 5 条 → 连续性通过 → 提升为习惯
  Day1,2 设 22°C, Day3 设 24°C, Day4,5 设 22°C → Day3 不在聚类中
    → 最近 5 条中仅 4 条在聚类 → 不通过 → 不提升

算法:
  1. 从聚类提取主导信号名（json_metadata.signal）
  2. 从聚类提取主导上下文模式（time_bucket + vehicle_state + geofence 众数）
  3. 在全量 facts 中找同信号 + 相似上下文的候选集
  4. 候选集按时间排序，取最近 N 条
  5. 检查最近 N 条是否全部在聚类中
"""
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from panoramix_core.models.fact import Fact

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 结果数据结构
# ═══════════════════════════════════════════════════

@dataclass
class SignalConsecutiveness:
    """单个信号维度的连续性校验结果"""
    signal_name: str
    is_consecutive: bool
    total_candidates: int       # 同信号+同上下文的总候选数
    required: int               # 要求连续次数
    recent_in_cluster: int      # 最近 N 次中在聚类内的数量
    detail: str = ""


@dataclass
class ClusterConsecutiveness:
    """聚类整体连续性校验结果"""
    is_consecutive: bool        # 主导信号维度通过
    cluster_size: int
    dominant_signal: str
    dominant_context: Dict
    signal_results: List[SignalConsecutiveness] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = []
        for r in self.signal_results:
            mark = "✓" if r.is_consecutive else "✗"
            parts.append(f"{mark} {r.signal_name}: {r.detail}")
        return "; ".join(parts)


# ═══════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════

def verify_cluster_consecutiveness(
    cluster_facts: List[Fact],
    all_facts: List[Fact],
    required_consecutive: int = 5,
) -> ClusterConsecutiveness:
    """
    验证 DBSCAN 聚类是否代表用户最近连续的行为模式。

    Args:
        cluster_facts: 聚类中的 Fact 列表
        all_facts: 全量 Fact 列表（含聚类内外所有 facts）
        required_consecutive: 要求的最近连续次数（PRD 默认 5）

    Returns:
        ClusterConsecutiveness: 包含是否通过 + 各信号维度详情
    """
    if not cluster_facts:
        return ClusterConsecutiveness(
            is_consecutive=False, cluster_size=0,
            dominant_signal="unknown", dominant_context={},
        )

    cluster_ids = {f.id for f in cluster_facts}

    # 1. 提取主导信号
    dominant_signal = _get_dominant_signal(cluster_facts)

    # 2. 提取主导上下文
    dominant_ctx = _get_dominant_context(cluster_facts)

    # 3. 在全量 facts 中找同信号+相似上下文的候选集
    candidates = _find_candidates(all_facts, dominant_signal, dominant_ctx)
    candidates.sort(key=lambda f: f.time_stamp)

    # 4. 校验最近 N 条
    result = _check_recent(candidates, cluster_ids, dominant_signal, required_consecutive)

    verdict = ClusterConsecutiveness(
        is_consecutive=result.is_consecutive,
        cluster_size=len(cluster_facts),
        dominant_signal=dominant_signal,
        dominant_context=dominant_ctx,
        signal_results=[result],
    )

    log.info(
        f"Consecutiveness [{dominant_signal}]: "
        f"{'PASS' if verdict.is_consecutive else 'FAIL'} — {result.detail}"
    )
    return verdict


def make_cluster_filter(required_consecutive: int = 5):
    """
    构造一个可传入 HabitsDetector.detect_habits(cluster_filter=...) 的过滤函数。

    用法:
        detector.detect_habits(items, cluster_filter=make_cluster_filter(5))
    """
    def _filter(cluster_facts: List[Fact], all_facts: List[Fact]) -> bool:
        result = verify_cluster_consecutiveness(
            cluster_facts, all_facts, required_consecutive
        )
        return result.is_consecutive
    return _filter


# ═══════════════════════════════════════════════════
# 内部实现
# ═══════════════════════════════════════════════════

def _get_dominant_signal(facts: List[Fact]) -> str:
    """提取聚类的主导信号名（众数）"""
    signals = [_extract_signal_name(f) for f in facts]
    counter = Counter(signals)
    return counter.most_common(1)[0][0] if counter else "unknown"


def _extract_signal_name(fact: Fact) -> str:
    """从 Fact.json_metadata 提取信号名"""
    if fact.json_metadata:
        try:
            return json.loads(fact.json_metadata).get("signal", "unknown")
        except (json.JSONDecodeError, TypeError):
            pass
    return "unknown"


def _get_dominant_context(facts: List[Fact]) -> Dict:
    """
    提取聚类的主导上下文模式（各维度众数 + 一致性比例）。

    一致性比例用于决定候选过滤是否启用该维度：
    若聚类内 <80% 的 facts 共享同一 time_bucket，说明该行为不绑定时段，
    候选过滤时跳过 time_bucket 维度（如收费站开窗发生在不同时段）。
    """
    n = len(facts)
    time_buckets = Counter(f.context.time_bucket for f in facts)
    vehicle_states = Counter(f.context.vehicle_state for f in facts)
    geofences = Counter(
        f.context.geofence for f in facts if f.context.geofence is not None
    )

    tb_top = time_buckets.most_common(1)[0] if time_buckets else ("unknown", 0)
    vs_top = vehicle_states.most_common(1)[0] if vehicle_states else ("unknown", 0)

    return {
        "time_bucket": tb_top[0],
        "time_bucket_agreement": tb_top[1] / n if n else 0,
        "vehicle_state": vs_top[0],
        "vehicle_state_agreement": vs_top[1] / n if n else 0,
        "geofence": geofences.most_common(1)[0][0] if geofences else None,
    }


_AGREEMENT_THRESHOLD = 0.8  # 聚类内 ≥80% 一致才启用该维度过滤

def _find_candidates(
    all_facts: List[Fact],
    signal_name: str,
    dominant_ctx: Dict,
) -> List[Fact]:
    """
    在全量 facts 中找同信号 + 相似上下文的候选集。

    匹配条件:
      - 信号名一致（必须）
      - vehicle_state 一致（必须）
      - time_bucket: 仅当聚类内一致性 ≥80% 时才过滤
        （收费站开窗可能发生在不同时段，不应按 time_bucket 过滤）
      - geofence: 若聚类有特定围栏则候选也需匹配
    """
    filter_time_bucket = dominant_ctx.get("time_bucket_agreement", 1.0) >= _AGREEMENT_THRESHOLD

    candidates = []
    for f in all_facts:
        if _extract_signal_name(f) != signal_name:
            continue
        if f.context.vehicle_state != dominant_ctx["vehicle_state"]:
            continue
        if filter_time_bucket:
            if f.context.time_bucket != dominant_ctx["time_bucket"]:
                continue
        if dominant_ctx["geofence"] is not None:
            if f.context.geofence != dominant_ctx["geofence"]:
                continue
        candidates.append(f)
    return candidates


def _check_recent(
    candidates: List[Fact],
    cluster_ids: set,
    signal_name: str,
    required: int,
) -> SignalConsecutiveness:
    """检查候选集的最近 N 条是否全部在聚类中"""
    total = len(candidates)

    if total < required:
        return SignalConsecutiveness(
            signal_name=signal_name,
            is_consecutive=False,
            total_candidates=total,
            required=required,
            recent_in_cluster=0,
            detail=f"insufficient data: {total}/{required}",
        )

    recent = candidates[-required:]
    in_cluster = sum(1 for f in recent if f.id in cluster_ids)

    return SignalConsecutiveness(
        signal_name=signal_name,
        is_consecutive=(in_cluster >= required),
        total_candidates=total,
        required=required,
        recent_in_cluster=in_cluster,
        detail=f"{in_cluster}/{required} recent in cluster",
    )
