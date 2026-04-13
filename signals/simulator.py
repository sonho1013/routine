"""
JSON 剧本模拟器 — 对接外部 mockup 数据格式

职责:
  加载 mockup JSON → 按时间线回放事件 → 每个事件喂给 signal_to_fact → 产出 Fact 列表
  支持批量模式（一次性处理全部）和单步模式（逐事件推进，用于 Streamlit）

mockup 数据格式 (由外部同事提供):
  {
    "user": "Mary",
    "user_id": "U001",
    "base_date": "2025-10-06",
    "scenes": {
      "morning_commute": [ {event}, {event}, ... ],
      "arriving_home":   [ ... ],
      "noise":           [ ... ]
    }
  }

  每个 event:
  {
    "scene": "morning_commute",   # 场景标签（仅用于验证，不喂给管线）
    "day": 1,
    "date": "2025-10-06",
    "signals": [
      {"t": "2025-10-06T08:15:00", "signal": "engine_status", "value": "on"},
      ...
    ]
  }
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple

from signals.signal_event import SignalEvent, parse_event_signals
from engine.signal_to_fact import signals_to_facts
from panoramix_core.models.fact import Fact

log = logging.getLogger(__name__)


class SignalSimulator:
    """
    JSON 剧本回放模拟器。

    用法:
        sim = SignalSimulator.from_file("tests/step1_mockup_data.json")

        # 批量模式 — 一次性处理全部事件，返回所有 Fact
        all_facts = sim.run_all()

        # 单步模式 — 逐事件迭代（用于 Streamlit Tab 1 的 "Analysis to model"）
        for step in sim.iter_steps():
            print(step.day, step.scene_label, len(step.facts))
    """

    def __init__(self, mockup_data: Dict):
        self.user = mockup_data.get("user", "unknown")
        self.user_id = mockup_data.get("user_id", "U000")
        self.base_date = mockup_data.get("base_date", "")
        self._raw = mockup_data

        # 展平所有事件并按时间排序
        self._events = self._flatten_and_sort(mockup_data)
        self._cursor = 0

        log.info(
            f"Simulator loaded: user={self.user}, "
            f"{len(self._events)} events, base_date={self.base_date}"
        )

    # ── 构造方法 ──

    @classmethod
    def from_file(cls, path: str) -> "SignalSimulator":
        """从 JSON 文件加载 mockup 数据"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    @classmethod
    def from_generator(cls, generator_func, **kwargs) -> "SignalSimulator":
        """从 mock_data_generator.generate_full_dataset() 加载"""
        data = generator_func(**kwargs)
        return cls(data)

    # ── 批量模式 ──

    def run_all(self) -> List[Fact]:
        """一次性处理全部事件，返回所有生成的 Fact"""
        all_facts = []
        for event in self._events:
            facts = signals_to_facts(event)
            all_facts.extend(facts)
        log.info(f"run_all: {len(self._events)} events → {len(all_facts)} facts")
        return all_facts

    # ── 单步模式 ──

    def iter_steps(self) -> Iterator["SimulatorStep"]:
        """逐事件迭代，每步返回 SimulatorStep"""
        for i, event in enumerate(self._events):
            facts = signals_to_facts(event)
            signal_events = parse_event_signals(event)
            yield SimulatorStep(
                index=i,
                total=len(self._events),
                event_data=event,
                signal_events=signal_events,
                facts=facts,
                scene_label=event.get("scene", "unknown"),
                day=event.get("day", 0),
                date=event.get("date", ""),
            )

    def step(self) -> Optional["SimulatorStep"]:
        """推进一步，返回 SimulatorStep 或 None（已结束）"""
        if self._cursor >= len(self._events):
            return None
        event = self._events[self._cursor]
        facts = signals_to_facts(event)
        signal_events = parse_event_signals(event)
        result = SimulatorStep(
            index=self._cursor,
            total=len(self._events),
            event_data=event,
            signal_events=signal_events,
            facts=facts,
            scene_label=event.get("scene", "unknown"),
            day=event.get("day", 0),
            date=event.get("date", ""),
        )
        self._cursor += 1
        return result

    def reset(self):
        """重置游标到起点"""
        self._cursor = 0

    # ── 查询 ──

    @property
    def total_events(self) -> int:
        return len(self._events)

    @property
    def remaining(self) -> int:
        return max(0, len(self._events) - self._cursor)

    @property
    def progress(self) -> float:
        """当前进度 0.0 ~ 1.0"""
        if not self._events:
            return 1.0
        return self._cursor / len(self._events)

    def get_events_by_scene(self, scene_label: str) -> List[Dict]:
        """获取指定场景的事件列表"""
        return [e for e in self._events if e.get("scene") == scene_label]

    def get_scene_summary(self) -> Dict[str, int]:
        """各场景事件计数"""
        summary = {}
        for e in self._events:
            scene = e.get("scene", "unknown")
            summary[scene] = summary.get(scene, 0) + 1
        return summary

    # ── 内部 ──

    @staticmethod
    def _flatten_and_sort(mockup_data: Dict) -> List[Dict]:
        """将按场景分组的事件展平为按时间排序的列表"""
        events = []
        scenes = mockup_data.get("scenes", {})
        for scene_name, scene_events in scenes.items():
            for event in scene_events:
                events.append(event)

        # 按首条信号的时间戳排序
        def _sort_key(event):
            signals = event.get("signals", [])
            if signals:
                return signals[0].get("t", "")
            return event.get("date", "")

        events.sort(key=_sort_key)
        return events


class SimulatorStep:
    """单步回放结果"""

    def __init__(
        self,
        index: int,
        total: int,
        event_data: Dict,
        signal_events: List[SignalEvent],
        facts: List[Fact],
        scene_label: str,
        day: int,
        date: str,
    ):
        self.index = index
        self.total = total
        self.event_data = event_data
        self.signal_events = signal_events
        self.facts = facts
        self.scene_label = scene_label  # 仅用于验证/展示，不参与管线
        self.day = day
        self.date = date

    @property
    def progress_pct(self) -> float:
        """进度百分比"""
        return (self.index + 1) / self.total * 100 if self.total else 100

    def __repr__(self):
        return (
            f"Step({self.index+1}/{self.total} "
            f"day={self.day} scene={self.scene_label} "
            f"signals={len(self.signal_events)} facts={len(self.facts)})"
        )
