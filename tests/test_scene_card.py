"""
场景卡命名单元测试

覆盖:
  - group_habits_by_context(): 分组逻辑
  - _format_context(): 上下文格式化
  - _attach_scene_to_habit(): json_metadata 写入
  - _default_scene_name(): 无 LLM 时的默认命名
  - generate_scene_cards(): 无 LLM / 有 LLM mock / 解析错误
  - SceneCard.summary: 数据结构
  - HabitDemoEngine 集成: scene_cards 出现在结果中
"""
import os
import sys
import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.scene_card import (
    SceneCard,
    generate_scene_cards,
    group_habits_by_context,
    _context_key,
    _format_context,
    _attach_scene_to_habit,
    _default_scene_name,
    _call_llm_scene_name,
    _load_prompt_template,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _habit(text, time_bucket="early_morning", vehicle_state="engine_started",
           geofence=None, weekday=True, json_metadata=None):
    """构造 HABIT 类型 Fact"""
    return Fact(
        text=text,
        type=FactType.HABIT,
        durability=FactDurability.LONG_TERM,
        time_stamp=datetime(2025, 10, 10, 8, 0),
        source=FactSources.HABITS_DETECTOR,
        context=StructuredContext(
            time_bucket=time_bucket,
            hour=8,
            weekday=weekday,
            vehicle_state=vehicle_state,
            geofence=geofence,
        ),
        json_metadata=json_metadata,
    )


# ═══════════════════════════════════════════════════
# 分组
# ═══════════════════════════════════════════════════

class TestContextKey:
    def test_basic(self):
        h = _habit("test", time_bucket="morning", vehicle_state="parked", geofence="home")
        assert _context_key(h) == "morning|parked|home"

    def test_no_geofence(self):
        h = _habit("test", geofence=None)
        assert _context_key(h) == "early_morning|engine_started|none"


class TestGroupHabits:
    def test_single_group(self):
        habits = [
            _habit("set AC to 22"),
            _habit("turn on seat heating"),
            _habit("start navigation to work"),
        ]
        groups = group_habits_by_context(habits)
        assert len(groups) == 1
        key = "early_morning|engine_started|none"
        assert len(groups[key]) == 3

    def test_multiple_groups(self):
        habits = [
            _habit("set AC to 22", time_bucket="early_morning", vehicle_state="engine_started"),
            _habit("turn off AC", time_bucket="evening", vehicle_state="parked", geofence="home"),
            _habit("close windows", time_bucket="evening", vehicle_state="parked", geofence="home"),
            _habit("lower window", time_bucket="afternoon", vehicle_state="crawling"),
        ]
        groups = group_habits_by_context(habits)
        assert len(groups) == 3
        assert len(groups["evening|parked|home"]) == 2
        assert len(groups["early_morning|engine_started|none"]) == 1
        assert len(groups["afternoon|crawling|none"]) == 1

    def test_empty(self):
        assert group_habits_by_context([]) == {}


# ═══════════════════════════════════════════════════
# 上下文格式化
# ═══════════════════════════════════════════════════

class TestFormatContext:
    def test_weekday_morning(self):
        h = _habit("test", time_bucket="early_morning", vehicle_state="engine_started", weekday=True)
        ctx = _format_context(h)
        assert "early morning" in ctx
        assert "weekday" in ctx
        assert "engine started" in ctx

    def test_weekend(self):
        h = _habit("test", weekday=False)
        ctx = _format_context(h)
        assert "weekend" in ctx

    def test_with_geofence(self):
        h = _habit("test", geofence="home")
        ctx = _format_context(h)
        assert "home" in ctx

    def test_no_geofence(self):
        h = _habit("test", geofence=None)
        ctx = _format_context(h)
        assert "none" in ctx

    def test_no_weekday_info(self):
        h = _habit("test", weekday=None)
        ctx = _format_context(h)
        assert "weekday" not in ctx
        assert "weekend" not in ctx


# ═══════════════════════════════════════════════════
# json_metadata 写入
# ═══════════════════════════════════════════════════

class TestAttachScene:
    def test_empty_metadata(self):
        h = _habit("test")
        _attach_scene_to_habit(h, "Morning Commute", 0.95)
        meta = json.loads(h.json_metadata)
        assert meta["scene_name"] == "Morning Commute"
        assert meta["scene_confidence"] == 0.95

    def test_existing_metadata(self):
        h = _habit("test", json_metadata='{"signal": "hvac_temp_target"}')
        _attach_scene_to_habit(h, "Morning Commute", 0.90)
        meta = json.loads(h.json_metadata)
        assert meta["scene_name"] == "Morning Commute"
        assert meta["signal"] == "hvac_temp_target"

    def test_overwrite_scene(self):
        h = _habit("test")
        _attach_scene_to_habit(h, "Old Scene", 0.8)
        _attach_scene_to_habit(h, "New Scene", 0.9)
        meta = json.loads(h.json_metadata)
        assert meta["scene_name"] == "New Scene"


# ═══════════════════════════════════════════════════
# 默认场景名
# ═══════════════════════════════════════════════════

class TestDefaultSceneName:
    def test_morning_started(self):
        name = _default_scene_name("early_morning|engine_started|none")
        assert "Early Morning" in name
        assert "Engine Started" in name

    def test_evening_parked(self):
        name = _default_scene_name("evening|parked|home")
        assert "Evening" in name
        assert "Parked" in name


# ═══════════════════════════════════════════════════
# LLM 调用解析
# ═══════════════════════════════════════════════════

class TestCallLLMSceneName:
    def test_valid_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Weekday Morning Commute,0.95"
        name, conf = _call_llm_scene_name(mock_llm, "{context}{habits}", "ctx", "habits")
        assert name == "Weekday Morning Commute"
        assert conf == 0.95

    def test_quoted_response(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = '"Arriving Home Shutdown",0.92'
        name, conf = _call_llm_scene_name(mock_llm, "{context}{habits}", "ctx", "habits")
        assert name == "Arriving Home Shutdown"
        assert conf == 0.92

    def test_low_confidence(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Vague Scene,0.3"
        name, conf = _call_llm_scene_name(mock_llm, "{context}{habits}", "ctx", "habits")
        assert name == "Vague Scene"
        assert conf == 0.3

    def test_parse_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "invalid output without comma"
        name, conf = _call_llm_scene_name(mock_llm, "{context}{habits}", "ctx", "habits")
        assert name == "General Driving"
        assert conf == 0.5

    def test_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API timeout")
        name, conf = _call_llm_scene_name(mock_llm, "{context}{habits}", "ctx", "habits")
        assert name == "General Driving"
        assert conf == 0.5


# ═══════════════════════════════════════════════════
# generate_scene_cards (无 LLM)
# ═══════════════════════════════════════════════════

class TestGenerateSceneCardsNoLLM:
    def test_empty_habits(self):
        cards = generate_scene_cards([], llm_client=None)
        assert cards == []

    def test_single_group(self):
        habits = [
            _habit("set AC to 22"),
            _habit("start navigation to work"),
        ]
        cards = generate_scene_cards(habits, llm_client=None)
        assert len(cards) == 1
        assert cards[0].confidence == 0.5
        assert len(cards[0].habits) == 2
        # 每个 habit 应有 scene_name 写入 json_metadata
        for h in habits:
            meta = json.loads(h.json_metadata)
            assert "scene_name" in meta
            assert meta["scene_confidence"] == 0.5

    def test_multiple_groups(self):
        habits = [
            _habit("set AC to 22", time_bucket="early_morning", vehicle_state="engine_started"),
            _habit("close windows", time_bucket="evening", vehicle_state="parked", geofence="home"),
        ]
        cards = generate_scene_cards(habits, llm_client=None)
        assert len(cards) == 2


# ═══════════════════════════════════════════════════
# generate_scene_cards (有 LLM mock)
# ═══════════════════════════════════════════════════

class TestGenerateSceneCardsWithLLM:
    def test_llm_scene_naming(self):
        habits = [
            _habit("set AC to 22", time_bucket="early_morning", vehicle_state="engine_started"),
            _habit("start navigation", time_bucket="early_morning", vehicle_state="engine_started"),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "Weekday Morning Commute,0.95"

        cards = generate_scene_cards(habits, llm_client=mock_llm)
        assert len(cards) == 1
        assert cards[0].scene_name == "Weekday Morning Commute"
        assert cards[0].confidence == 0.95

        for h in habits:
            meta = json.loads(h.json_metadata)
            assert meta["scene_name"] == "Weekday Morning Commute"

    def test_multiple_groups_multiple_calls(self):
        habits = [
            _habit("set AC to 22", time_bucket="early_morning", vehicle_state="engine_started"),
            _habit("close windows", time_bucket="evening", vehicle_state="parked", geofence="home"),
        ]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            "Morning Commute,0.90",
            "Arriving Home Shutdown,0.92",
        ]

        cards = generate_scene_cards(habits, llm_client=mock_llm)
        assert len(cards) == 2
        assert mock_llm.invoke.call_count == 2

        names = {c.scene_name for c in cards}
        assert "Morning Commute" in names
        assert "Arriving Home Shutdown" in names

    def test_llm_failure_fallback(self):
        habits = [_habit("set AC to 22")]
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")

        cards = generate_scene_cards(habits, llm_client=mock_llm)
        assert len(cards) == 1
        assert cards[0].scene_name == "General Driving"
        assert cards[0].confidence == 0.5


# ═══════════════════════════════════════════════════
# SceneCard 数据结构
# ═══════════════════════════════════════════════════

class TestSceneCardSummary:
    def test_summary(self):
        habits = [_habit("set AC to 22"), _habit("start nav")]
        card = SceneCard(
            scene_name="Morning Commute",
            confidence=0.95,
            context_key="early_morning|engine_started|none",
            habits=habits,
        )
        s = card.summary
        assert s["scene_name"] == "Morning Commute"
        assert s["confidence"] == 0.95
        assert s["habit_count"] == 2
        assert "set AC to 22" in s["habits"]


# ═══════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════

class TestPromptTemplate:
    def test_load(self):
        template = _load_prompt_template()
        assert "{context}" in template
        assert "{habits}" in template
        assert "scene_name,confidence_score" in template
