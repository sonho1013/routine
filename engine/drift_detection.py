"""
drift_detection — 纯函数层

is_drifted / compute_raw_value_stats / cosine_distance 全部无副作用，
embedding_fallback 的网络调用通过依赖注入 embed_fn 隔离，方便单测。

§ 7.3 / § 7.4 of design spec.
"""
import json
import logging
import math
import statistics
from collections import Counter
from typing import Callable, List, Optional

log = logging.getLogger(__name__)

_EMBEDDING_DRIFT_THRESHOLD = 0.15


def is_drifted(
    signal: str,
    old_stats: dict,
    new_stats: dict,
    rule_engine,
    *,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> bool:
    """
    按信号的 drift_metric 判断是否漂移。

    - numeric_mean_diff: |new.mean - old.mean| > threshold
    - dominant_value_change: dominant_value 变化即漂移
    - embedding_fallback: cosine_distance(embed(old_text), embed(new_text)) > 0.15

    没有 rule 或 rule 无 drift_metric → 走 embedding_fallback。
    """
    rule = rule_engine.get_rule(signal) or {}
    metric = rule.get("drift_metric", "embedding_fallback")

    if metric == "numeric_mean_diff":
        threshold = rule["drift_threshold"]
        return abs(new_stats["mean"] - old_stats["mean"]) > threshold

    if metric == "dominant_value_change":
        return new_stats["dominant_value"] != old_stats["dominant_value"]

    if metric == "none":
        # yaml 里显式标 none 的信号跳过 drift 判定（如 nav_destination）
        return False

    if metric == "embedding_fallback":
        if embed_fn is None:
            from panoramix_core.embedder import Embedder
            embed_fn = Embedder().embed
        dist = cosine_distance(
            embed_fn(old_stats["habit_text"]),
            embed_fn(new_stats["habit_text"]),
        )
        return dist > _EMBEDDING_DRIFT_THRESHOLD

    log.warning(f"unknown drift_metric '{metric}' for signal '{signal}'")
    return False


def cosine_distance(a: List[float], b: List[float]) -> float:
    """
    1 - cosine_similarity.
    零向量返回 1.0（约定：无方向视为最大距离）。
    """
    if not a or not b:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def compute_raw_value_stats(pref_facts: list, signal_category: str) -> dict:
    """
    从一组 PREF fact 的 raw_value 聚合统计。

    pref_facts: Fact 对象列表（每个 fact.json_metadata 里有 raw_value 字段）
    signal_category: "numeric" | "categorical"
    返回符合 spec §3.4 的 raw_value_stats 字典。
    """
    raw_values = [
        json.loads(f.json_metadata)["raw_value"] for f in pref_facts
    ]

    if signal_category == "numeric":
        vals = [float(v) for v in raw_values if isinstance(v, (int, float))]
        return {
            "type": "numeric",
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
            "count": len(vals),
        }

    if signal_category == "categorical":
        counts = Counter(str(v) for v in raw_values)
        return {
            "type": "categorical",
            "dominant_value": counts.most_common(1)[0][0] if counts else "",
            "value_counts": dict(counts),
            "count": sum(counts.values()),
        }

    raise ValueError(f"unknown signal_category: {signal_category}")
