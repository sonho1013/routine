"""
场景卡命名 — 习惯合成后由 LLM 为上下文相似的习惯组生成场景名

流程:
  1. 按上下文相似性对习惯分组 (same time_bucket + vehicle_state + geofence → same group)
  2. 对每组调用 LLM 生成场景名 (scene_name, confidence)
  3. 将 scene_name 写入每个 habit 的 json_metadata

场景名在此阶段首次出现，之前全流程无场景标签。
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from panoramix_core.models.fact import Fact

log = logging.getLogger(__name__)

SCENE_PROMPT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "panoramix_core", "clustering", "prompts",
    "scene_naming_prompt.txt",
)

SCENE_CONFIDENCE_THRESHOLD = 0.6


# ═══════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════

@dataclass
class SceneCard:
    """场景卡：一组上下文相似习惯 + LLM 生成的场景名"""
    scene_name: str
    confidence: float
    context_key: str                    # 分组键 "time_bucket|vehicle_state|geofence"
    habits: List[Fact] = field(default_factory=list)

    @property
    def summary(self) -> Dict:
        return {
            "scene_name": self.scene_name,
            "confidence": self.confidence,
            "context_key": self.context_key,
            "habit_count": len(self.habits),
            "habits": [h.text for h in self.habits],
        }


# ═══════════════════════════════════════════════════
# 分组
# ═══════════════════════════════════════════════════

def _context_key(fact: Fact) -> str:
    """生成分组键: time_bucket|vehicle_state|geofence"""
    ctx = fact.context
    geo = ctx.geofence or "none"
    return f"{ctx.time_bucket}|{ctx.vehicle_state}|{geo}"


def group_habits_by_context(habits: List[Fact]) -> Dict[str, List[Fact]]:
    """按上下文相似性分组"""
    groups: Dict[str, List[Fact]] = {}
    for h in habits:
        key = _context_key(h)
        groups.setdefault(key, []).append(h)
    return groups


# ═══════════════════════════════════════════════════
# 上下文格式化
# ═══════════════════════════════════════════════════

_TIME_BUCKET_LABELS = {
    "early_morning": "early morning",
    "morning": "morning",
    "midday": "midday",
    "afternoon": "afternoon",
    "evening": "evening",
    "night": "night",
    "unknown": "unknown time",
}

_VEHICLE_STATE_LABELS = {
    "engine_started": "engine started",
    "parked": "parked",
    "crawling": "crawling",
    "driving": "driving",
    "idle": "idle",
    "unknown": "unknown state",
}


def _format_context(fact: Fact) -> str:
    """将 StructuredContext 格式化为 prompt 中的自然语言描述"""
    ctx = fact.context
    time_label = _TIME_BUCKET_LABELS.get(ctx.time_bucket, ctx.time_bucket)
    if ctx.weekday is True:
        time_label += " (weekday)"
    elif ctx.weekday is False:
        time_label += " (weekend)"

    vehicle_label = _VEHICLE_STATE_LABELS.get(ctx.vehicle_state, ctx.vehicle_state)
    geo_label = ctx.geofence if ctx.geofence else "none"

    return f"Time: {time_label}, Vehicle: {vehicle_label}, Location: {geo_label}"


# ═══════════════════════════════════════════════════
# LLM 调用
# ═══════════════════════════════════════════════════

def _load_prompt_template() -> str:
    with open(SCENE_PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def _call_llm_scene_name(
    llm_client,
    prompt_template: str,
    context_str: str,
    habits_str: str,
) -> Tuple[str, float]:
    """
    调用 LLM 生成场景名。

    Returns:
        (scene_name, confidence)
        失败时返回 ("General Driving", 0.5)
    """
    prompt = prompt_template.format(context=context_str, habits=habits_str)

    try:
        raw = llm_client.invoke(prompt)
        log.debug(f"Scene naming raw: {raw}")
    except Exception as e:
        log.error(f"Scene naming LLM call failed: {e}")
        return "General Driving", 0.5

    try:
        name, conf_str = raw.rsplit(",", 1)
        confidence = float(conf_str.strip())
        name = name.strip().strip('"').strip("'")

        if not name:
            return "General Driving", 0.5
        if confidence < SCENE_CONFIDENCE_THRESHOLD:
            log.warning(f"Scene name low confidence {confidence}: {name}")
            return name, confidence

        return name, confidence
    except Exception as e:
        log.error(f"Could not parse scene naming output: {raw}. Error: {e}")
        return "General Driving", 0.5


# ═══════════════════════════════════════════════════
# 写入 json_metadata
# ═══════════════════════════════════════════════════

def _attach_scene_to_habit(habit: Fact, scene_name: str, confidence: float):
    """将 scene_name 写入 habit 的 json_metadata"""
    meta = {}
    if habit.json_metadata:
        try:
            meta = json.loads(habit.json_metadata)
        except (json.JSONDecodeError, TypeError):
            pass
    meta["scene_name"] = scene_name
    meta["scene_confidence"] = confidence
    habit.json_metadata = json.dumps(meta)


# ═══════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════

def generate_scene_cards(
    habits: List[Fact],
    llm_client=None,
) -> List[SceneCard]:
    """
    为一组习惯生成场景卡。

    流程:
      1. 按上下文相似性分组
      2. 对每组调用 LLM 生成场景名
      3. 将 scene_name 写入每个 habit 的 json_metadata

    Args:
        habits: HABIT 类型的 Fact 列表（应已有 dominant context）
        llm_client: 实现 .invoke(prompt) -> str 的 LLM 客户端
                    None 时使用默认场景名

    Returns:
        List[SceneCard]: 场景卡列表
    """
    if not habits:
        return []

    groups = group_habits_by_context(habits)
    log.info(f"Scene card: {len(habits)} habits → {len(groups)} context groups")

    scene_cards = []
    prompt_template = None
    if llm_client is not None:
        try:
            prompt_template = _load_prompt_template()
        except Exception as e:
            log.error(f"Failed to load scene prompt: {e}")

    for ctx_key, group_habits in groups.items():
        # 用组内第一个 habit 的 context 做代表（同组 context 相同）
        context_str = _format_context(group_habits[0])
        habits_str = "\n".join(f"- {h.text}" for h in group_habits)

        if llm_client is not None and prompt_template is not None:
            scene_name, confidence = _call_llm_scene_name(
                llm_client, prompt_template, context_str, habits_str
            )
        else:
            # 无 LLM 时用上下文键做默认场景名
            scene_name = _default_scene_name(ctx_key)
            confidence = 0.5

        # 写入每个 habit 的 json_metadata
        for h in group_habits:
            _attach_scene_to_habit(h, scene_name, confidence)

        card = SceneCard(
            scene_name=scene_name,
            confidence=confidence,
            context_key=ctx_key,
            habits=group_habits,
        )
        scene_cards.append(card)
        log.info(
            f"Scene card: [{ctx_key}] → \"{scene_name}\" "
            f"(confidence={confidence:.2f}, {len(group_habits)} habits)"
        )

    return scene_cards


def _default_scene_name(ctx_key: str) -> str:
    """无 LLM 时根据 context_key 生成默认场景名"""
    parts = ctx_key.split("|")
    time_bucket = parts[0] if len(parts) > 0 else "unknown"
    vehicle_state = parts[1] if len(parts) > 1 else "unknown"

    time_label = _TIME_BUCKET_LABELS.get(time_bucket, time_bucket).title()
    vehicle_label = _VEHICLE_STATE_LABELS.get(vehicle_state, vehicle_state).title()

    return f"{time_label} {vehicle_label}"
