"""
engine/scene_card — 候选场景卡构造辅助

Wave 2 之后，SceneCard 的状态与持久化由 panoramix_core/store/scene_card_store.py
负责，本文件只保留以下两类辅助函数供 pipeline 使用：

  1. build_candidate_pending(): 从 habits 列表构造候选 SceneCard Pydantic 对象
     （阶段 1 慢动作内调用；不写 DB）
  2. _attach_scene_to_habit(): 将 scene_name 写入 Habit Fact 的 json_metadata
     （供 cluster_confidence 测试等内部调用）

老的 in-memory dataclass SceneCard 已删除。
"""
import json
import logging
from datetime import datetime
from typing import List

from panoramix_core.models.fact import Fact
from panoramix_core.models.habit import Habit
from panoramix_core.models.scene_card import SceneCard

log = logging.getLogger(__name__)


def build_candidate_pending(
    habits: List[Habit],
    structural_key: str,
    display_name: str,
    batch_id: int,
    username: str,
    content_snapshot: dict,
) -> SceneCard:
    """
    根据一组同 structural_key 的 habits 构造一张 pending SceneCard。
    **不写 DB**；调用方负责把它传给 SceneCardStore.insert_pending。
    """
    return SceneCard(
        username=username,
        status="pending",
        structural_key=structural_key,
        display_name=display_name,
        content_snapshot=content_snapshot,
        first_seen_batch_id=batch_id,
        last_reinforced_batch_id=batch_id,
        created_at=datetime.now(),
    )


def _attach_scene_to_habit(habit: Fact, scene_name: str, confidence: float):
    """将 scene_name 写入 habit 的 json_metadata（保留既有字段）"""
    meta = {}
    if habit.json_metadata:
        try:
            meta = json.loads(habit.json_metadata)
        except (json.JSONDecodeError, TypeError):
            pass
    meta["scene_name"] = scene_name
    meta["scene_confidence"] = confidence
    habit.json_metadata = json.dumps(meta)
