"""
HabitLifecycleManager — drift detection 核心 (Round 9)

本模块包含三组函数/类：
1. compute_structural_key() — 场景卡身份生成
2. classify_candidate()     — 候选场景卡分类（Reinforce/ModifyDrift/NewPending）
3. HabitLifecycleManager    — 协调器，串起 store 调用

§ 7 of design spec.
"""
import hashlib
import logging
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard

log = logging.getLogger(__name__)


def compute_structural_key(habits: List[Habit]) -> str:
    """
    计算场景卡身份锚点。

    Key source = hash(dominant_geofence | dominant_time_bucket |
                      dominant_vehicle_state | sorted(signal_type_set))

    weekday 故意不参与（Round 5）。
    平票时按字典序取第一个，保证确定性。

    返回 SHA-256 hex 的前 16 字符。
    """
    if not habits:
        raise ValueError("compute_structural_key requires at least one habit")

    dominant_geofence = _dominant_value(
        [h.context_geofence or "none" for h in habits]
    )
    dominant_time_bucket = _dominant_value(
        [h.context_time_bucket for h in habits]
    )
    dominant_vehicle_state = _dominant_value(
        [h.context_vehicle_state for h in habits]
    )
    signal_types = sorted({h.signal_name for h in habits})

    key_src = "|".join([
        dominant_geofence,
        dominant_time_bucket,
        dominant_vehicle_state,
        ",".join(signal_types),
    ])
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]


def _dominant_value(values: List[str]) -> str:
    """
    取出现次数最多的值；平票时按字典序取第一个。
    """
    counts = Counter(values)
    max_count = max(counts.values())
    tied = sorted(v for v, c in counts.items() if c == max_count)
    return tied[0]


# ── Classification results ───────────────────────────────────────


@dataclass(frozen=True)
class Reinforce:
    card_id: str


@dataclass(frozen=True)
class ModifyDrift:
    card_id: str
    drifted_signals: List[str]


@dataclass(frozen=True)
class NewPending:
    structural_key: str


ClassificationResult = object  # type alias for typing hints


# ── Helpers ────────────────────────────────────────────────────────


class MixedSignalClusterError(Exception):
    """cluster contains facts from multiple signal types — violates §3.7 assumption"""


def assert_cluster_pure_signal(cluster_facts: list) -> str:
    """
    §3.7: HabitsDetector 输出的一个 cluster 内所有 PREF facts 应来自同一 signal。
    若违反即抛异常，让假设违反变成显式可观测事件。
    """
    import json as _json
    signals = {
        _json.loads(f.json_metadata)["signal"] for f in cluster_facts
    }
    if len(signals) != 1:
        raise MixedSignalClusterError(
            f"cluster has {len(signals)} signal types: {signals}. "
            f"Current drift detection assumes single-signal habits."
        )
    return next(iter(signals))


# ── classify_candidate ────────────────────────────────────────────


def classify_candidate(
    candidate_habits: List[Habit],
    accepted_cards: List[SceneCard],
    signal_rule_engine,
    *,
    embed_fn=None,
):
    """
    候选场景卡分类 → Reinforce / ModifyDrift / NewPending

    §7.2 of spec.
    """
    from engine.drift_detection import is_drifted

    key = compute_structural_key(candidate_habits)

    # Step A: 严格主键匹配
    matched = [c for c in accepted_cards if c.structural_key == key]

    if not matched:
        return NewPending(structural_key=key)

    if len(matched) > 1:
        log.warning(
            f"structural_key {key} matches multiple accepted cards "
            f"({len(matched)}); treating as NewPending (DCM-3)"
        )
        return NewPending(structural_key=key)

    accepted = matched[0]
    frozen = accepted.frozen_content_snapshot or {}

    # Step B: 按信号判漂移
    drifted = _detect_drift_for_card(
        candidate_habits=candidate_habits,
        frozen_snapshot=frozen,
        rule_engine=signal_rule_engine,
        embed_fn=embed_fn,
    )

    if drifted:
        return ModifyDrift(card_id=accepted.card_id, drifted_signals=drifted)
    return Reinforce(card_id=accepted.card_id)


def _detect_drift_for_card(
    candidate_habits: List[Habit],
    frozen_snapshot: dict,
    rule_engine,
    embed_fn=None,
) -> List[str]:
    """
    对 candidate 和 frozen_snapshot 按 signal_name 求交集后逐信号判断。
    返回漂移的 signal 列表；空列表即 reinforce。
    """
    from engine.drift_detection import is_drifted

    old_by_signal = {
        h["signal"]: h["raw_value_stats"]
        for h in frozen_snapshot.get("habits", [])
    }
    # 给 raw_value_stats 塞一个 habit_text 字段，供 embedding_fallback 用
    for h in frozen_snapshot.get("habits", []):
        stats = old_by_signal[h["signal"]]
        stats.setdefault("habit_text", h.get("habit_text", ""))

    new_by_signal = {}
    for h in candidate_habits:
        stats = dict(h.raw_value_stats)
        stats.setdefault("habit_text", h.text)
        new_by_signal[h.signal_name] = stats

    drifted: List[str] = []
    for signal, new_stats in new_by_signal.items():
        if signal not in old_by_signal:
            # candidate 多出一个旧快照没有的信号
            # → structural_key 本应不同 → 防御性地当作漂移
            drifted.append(signal)
            continue
        if is_drifted(
            signal, old_by_signal[signal], new_stats, rule_engine,
            embed_fn=embed_fn,
        ):
            drifted.append(signal)

    return drifted


# ── HabitLifecycleManager ─────────────────────────────────────────


class HabitLifecycleManager:
    """
    协调器 — 把"候选场景卡分组、GPT naming、classify、store 写入"串起来。

    本类在 Wave 4 pipeline 重写中被 engine/habit_engine.py 调用。
    保持构造函数轻量：store 对象从外面注入。
    """

    def __init__(
        self,
        scene_card_store,
        habit_store,
        signal_rule_engine,
        *,
        embed_fn=None,
    ):
        self.scene_card_store = scene_card_store
        self.habit_store = habit_store
        self.signal_rule_engine = signal_rule_engine
        self.embed_fn = embed_fn

    def classify_all(
        self,
        candidates_by_key: dict,
        accepted_cards: List[SceneCard],
    ) -> list:
        """
        对每组候选 habits（按 structural_key 分组）做分类。
        返回 [(structural_key, candidate_habits, ClassificationResult), ...]
        """
        results = []
        for key, habits in candidates_by_key.items():
            r = classify_candidate(
                candidate_habits=habits,
                accepted_cards=accepted_cards,
                signal_rule_engine=self.signal_rule_engine,
                embed_fn=self.embed_fn,
            )
            results.append((key, habits, r))
        return results
