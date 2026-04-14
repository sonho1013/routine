"""
engine/scene_card 构造辅助函数测试 — 迁移自旧 test_scene_card.py

覆盖 Wave 5 重写后保留的两个辅助函数：
  - _attach_scene_to_habit(): json_metadata 写入
  - build_candidate_pending(): 候选 SceneCard 构造
"""
import json
import os
import sys
from datetime import datetime

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.scene_card import _attach_scene_to_habit, build_candidate_pending
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


def _habit(text, json_metadata=None):
    return Fact(
        text=text,
        type=FactType.HABIT,
        durability=FactDurability.LONG_TERM,
        time_stamp=datetime(2025, 10, 10, 8, 0),
        source=FactSources.HABITS_DETECTOR,
        context=StructuredContext(
            time_bucket="early_morning",
            hour=8,
            weekday=True,
            vehicle_state="engine_started",
        ),
        json_metadata=json_metadata,
    )


# ── _attach_scene_to_habit ──

class TestAttachScene:
    def test_empty_metadata(self):
        h = _habit("test")
        _attach_scene_to_habit(h, "Morning Commute", 0.95)
        meta = json.loads(h.json_metadata)
        assert meta["scene_name"] == "Morning Commute"
        assert meta["scene_confidence"] == 0.95

    def test_existing_metadata_preserved(self):
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
        assert meta["scene_confidence"] == 0.9


# ── build_candidate_pending ──

class TestBuildCandidatePending:
    def test_returns_pydantic_scene_card(self):
        from panoramix_core.models.scene_card import SceneCard
        from panoramix_core.models.habit import Habit

        habits = [
            Habit(
                username="tester",
                batch_id=1,
                text="set AC to 22",
                signal_category="numeric",
                signal_name="hvac_temp_target",
                structural_key="sk_abc",
                context_time_bucket="early_morning",
                context_vehicle_state="engine_started",
                raw_value_stats={"habit_text": "set AC to 22"},
                member_fact_ids=[],
            )
        ]
        card = build_candidate_pending(
            habits=habits,
            structural_key="sk_abc",
            display_name="Morning Commute",
            batch_id=1,
            username="tester",
            content_snapshot={"habits": [{"habit_text": "set AC to 22"}]},
        )

        assert isinstance(card, SceneCard)
        assert card.status == "pending"
        assert card.structural_key == "sk_abc"
        assert card.display_name == "Morning Commute"
        assert card.username == "tester"
        assert card.first_seen_batch_id == 1
        assert card.last_reinforced_batch_id == 1
