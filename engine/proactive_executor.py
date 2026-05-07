"""
主动执行器 — 基于 context_distance 匹配已接受习惯

当用户进入某个上下文场景（如早晨启动车辆）时：
  1. 获取所有已 accepted 的 HABIT facts
  2. 用 context_distance() 计算当前上下文与每个 habit 上下文的距离
  3. 距离 < CONTEXT_MATCH_THRESHOLD 的 habits 作为候选
  4. 按距离排序，返回推荐动作列表

不硬编码场景类型 — 匹配完全由连续距离度量驱动。

使用:
    executor = ProactiveExecutor(fact_store)
    result = executor.recommend(current_context)
    for action in result.actions:
        print(action.habit_text, action.confidence, action.parsed_actions)
"""
import json
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from panoramix_core.config import CONTEXT_MATCH_THRESHOLD
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType
from panoramix_core.clustering.habits_detector import context_distance

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════

@dataclass
class RecommendedAction:
    """单条推荐动作"""
    habit_id: str
    habit_text: str
    context_distance: float         # 当前上下文与 habit 上下文的距离 [0, 1]
    match_confidence: float         # 匹配置信度 = 1 - context_distance
    clustering_confidence: float    # 聚类置信度（来自 json_metadata）
    scene_name: Optional[str]       # 场景名（来自 json_metadata）
    parsed_actions: Dict            # 解析出的控制指令
    habit_context: Dict             # habit 的关联上下文

    @property
    def combined_confidence(self) -> float:
        """综合置信度 = 匹配置信度 × 聚类置信度"""
        return round(self.match_confidence * self.clustering_confidence, 4)


@dataclass
class RecommendationResult:
    """推荐结果"""
    current_context: Dict
    actions: List[RecommendedAction] = field(default_factory=list)
    total_habits: int = 0           # 已接受习惯总数
    candidates_checked: int = 0     # 参与匹配的候选数
    threshold: float = CONTEXT_MATCH_THRESHOLD

    @property
    def has_recommendations(self) -> bool:
        return len(self.actions) > 0

    @property
    def summary(self) -> str:
        if not self.actions:
            return f"No match (checked {self.candidates_checked} habits, threshold={self.threshold})"
        parts = []
        for a in self.actions:
            parts.append(
                f"[dist={a.context_distance:.3f}] {a.habit_text} "
                f"(scene={a.scene_name}, combined_conf={a.combined_confidence:.3f})"
            )
        return f"{len(self.actions)} recommendations:\n  " + "\n  ".join(parts)


# ═══════════════════════════════════════════════════
# 主动执行器
# ═══════════════════════════════════════════════════

class ProactiveExecutor:
    """
    基于上下文匹配的主动执行器。

    不硬编码场景类型 — 通过 context_distance() 连续距离度量
    找到与当前上下文最接近的已接受习惯。
    """

    def __init__(self, fact_store, threshold: float = CONTEXT_MATCH_THRESHOLD):
        """
        Args:
            fact_store: FactStore 实例（用于查询已存储的 habits）
            threshold: 上下文匹配阈值，距离 < threshold 视为匹配
        """
        self.fact_store = fact_store
        self.threshold = threshold

    def recommend(
        self,
        current_context: StructuredContext,
        username: str,
        top_k: int = 0,
        habits: Optional[List[Fact]] = None,
    ) -> RecommendationResult:
        """
        根据当前上下文推荐匹配的习惯动作。

        Args:
            current_context: 当前车辆上下文
            username: 用户标识
            top_k: 最多返回 top_k 个推荐（0 = 不限制）
            habits: 预加载的 habits 列表 (None 时从 Chroma 查询)

        Returns:
            RecommendationResult: 包含按距离排序的推荐动作列表
        """
        ctx_dict = _context_to_dict(current_context)

        # 1. 获取所有 HABIT 类型 facts
        if habits is not None:
            all_habits = habits
        else:
            all_habits = self.fact_store.get_facts(username, types=[FactType.HABIT])

        # 2. 筛选已接受的
        accepted = [h for h in all_habits if h.accepted]

        if not accepted:
            log.info(f"ProactiveExecutor: no accepted habits for user {username}")
            return RecommendationResult(
                current_context=ctx_dict,
                total_habits=len(all_habits),
                candidates_checked=0,
                threshold=self.threshold,
            )

        # 3. 逐个计算 context_distance
        candidates = []
        for habit in accepted:
            dist = context_distance(current_context, habit.context)
            if dist < self.threshold:
                meta = _parse_habit_meta(habit)
                parsed = parse_habit_actions(habit.text)
                candidates.append(RecommendedAction(
                    habit_id=habit.id,
                    habit_text=habit.text,
                    context_distance=round(dist, 4),
                    match_confidence=round(1.0 - dist, 4),
                    clustering_confidence=meta.get("clustering_confidence", 0.0),
                    scene_name=meta.get("scene_name"),
                    parsed_actions=parsed,
                    habit_context=_context_to_dict(habit.context),
                ))

        # 4. 按综合置信度降序排序 (Tab 2 展示顺序)
        candidates.sort(key=lambda c: c.combined_confidence, reverse=True)
        if top_k > 0:
            candidates = candidates[:top_k]

        result = RecommendationResult(
            current_context=ctx_dict,
            actions=candidates,
            total_habits=len(all_habits),
            candidates_checked=len(accepted),
            threshold=self.threshold,
        )

        log.info(f"ProactiveExecutor: {result.summary}")
        return result


# ═══════════════════════════════════════════════════
# 动作解析 — 从 habit text 提取可执行控制指令
# ═══════════════════════════════════════════════════

# 信号名 → 正则提取模式
_ACTION_PATTERNS = [
    # HVAC
    (r"(?:air conditioning|AC|cabin).+?(\d+(?:\.\d+)?)\s*(?:degrees|°)", "hvac_temp_target", float),
    (r"(?:AC|ac)\s+temperature.+?(\d+(?:\.\d+)?)\s*(?:degrees|°)", "hvac_temp_target", float),
    (r"turned (?:on|off) (?:cabin )?air conditioning", "hvac_power", lambda _: "toggle"),
    # 座椅加热
    (r"seat heating.+?level\s*(\d+)", "seat_heating", int),
    # 导航
    (r"navigation to\s+(.+?)(?:\s*$)", "nav_destination", str),
    (r"(fastest|shortest|eco)\s+route", "nav_route_pref", str),
    # 媒体
    (r"(?:listening to|playing)\s+(?:podcast|music)\s+(.+?)(?:\s*$)", "media_source", str),
    (r"media volume.+?(\d+)\s*percent", "media_volume", int),
    (r"stopped all media", "media_off", lambda _: True),
    (r"(?:tuned to|selected media content)\s+(.+?)(?:\s*$)", "media_content_id", str),
    # 车窗
    (r"(?:lowered|opened).+?window.+?(\d+)\s*percent", "window_position", int),
    (r"closed all.+?window", "window_position", lambda _: 0),
    # 驾驶模式
    (r"(?:switched to|selected)\s+(eco|sport|comfort|normal)\s+(?:driving )?mode", "drive_mode", str),
    # ACC
    (r"acc.+?distance.+?(short|medium|long)", "acc_distance", str),
    # 无钥匙
    (r"(?:enabled|disabled)\s+keyless", "keyless_entry", lambda _: "toggle"),
    # 引擎
    (r"shut down.+?engine", "engine_status", lambda _: "off"),
]


def parse_habit_actions(habit_text: str) -> Dict:
    """
    从习惯文本中解析可执行控制指令。

    基于正则模式匹配，将自然语言习惯描述转换为信号名+值。
    例: "set cabin air conditioning temperature to 22 degrees"
        → {"hvac_temp_target": 22}

    Returns:
        dict: {signal_name: value, ...}  匹配不到时返回空 dict
    """
    actions = {}
    text_lower = habit_text.lower()

    for pattern, signal, converter in _ACTION_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            try:
                raw = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                value = converter(raw)
                actions[signal] = value
            except (ValueError, IndexError):
                actions[signal] = raw if 'raw' in dir() else True
    return actions


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _parse_habit_meta(habit: Fact) -> Dict:
    """从 habit.json_metadata 提取关键字段"""
    if habit.json_metadata:
        try:
            return json.loads(habit.json_metadata)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _context_to_dict(ctx: StructuredContext) -> Dict:
    """StructuredContext → dict"""
    return {
        "time_bucket": ctx.time_bucket,
        "hour": ctx.hour,
        "weekday": ctx.weekday,
        "vehicle_state": ctx.vehicle_state,
        "geofence": ctx.geofence,
    }
