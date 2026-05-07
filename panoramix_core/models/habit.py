"""Habit Pydantic 模型 — 对应 habits SQLite 表"""
import json
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Habit(BaseModel):
    habit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    batch_id: int

    text: str
    signal_category: str  # "numeric" | "categorical"
    signal_name: str
    structural_key: str

    context_time_bucket: str
    context_vehicle_state: str
    context_geofence: Optional[str] = None
    context_weekday: Optional[int] = None  # None / 0 / 1
    # 与 trigger list 2.xlsx 对齐的补充前置条件维度
    context_poi_type: Optional[str] = None
    context_wiper_state: str = "unknown"
    context_temp_bucket: str = "unknown"
    context_window_state: str = "unknown"
    context_door_lock: Optional[str] = None
    context_approach_unlock: Optional[str] = None

    raw_value_stats: dict  # §3.4 中 raw_value_stats 片段
    member_fact_ids: list[str]

    scene_card_id: Optional[str] = None

    created_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_row(cls, row) -> "Habit":
        def ts(k):
            v = row[k]
            return datetime.fromisoformat(v) if v else None

        return cls(
            habit_id=row["habit_id"],
            username=row["username"],
            batch_id=row["batch_id"],
            text=row["text"],
            signal_category=row["signal_category"],
            signal_name=row["signal_name"],
            structural_key=row["structural_key"],
            context_time_bucket=row["context_time_bucket"],
            context_vehicle_state=row["context_vehicle_state"],
            context_geofence=row["context_geofence"],
            context_weekday=row["context_weekday"],
            context_poi_type=_row_get(row, "context_poi_type"),
            context_wiper_state=_row_get(row, "context_wiper_state") or "unknown",
            context_temp_bucket=_row_get(row, "context_temp_bucket") or "unknown",
            context_window_state=_row_get(row, "context_window_state") or "unknown",
            context_door_lock=_row_get(row, "context_door_lock"),
            context_approach_unlock=_row_get(row, "context_approach_unlock"),
            raw_value_stats=json.loads(row["raw_value_stats_json"]),
            member_fact_ids=json.loads(row["member_fact_ids_json"]),
            scene_card_id=row["scene_card_id"],
            created_at=ts("created_at"),
        )

    def to_row_dict(self) -> dict:
        return {
            "habit_id": self.habit_id,
            "username": self.username,
            "batch_id": self.batch_id,
            "text": self.text,
            "signal_category": self.signal_category,
            "signal_name": self.signal_name,
            "structural_key": self.structural_key,
            "context_time_bucket": self.context_time_bucket,
            "context_vehicle_state": self.context_vehicle_state,
            "context_geofence": self.context_geofence,
            "context_weekday": self.context_weekday,
            "context_poi_type": self.context_poi_type,
            "context_wiper_state": self.context_wiper_state,
            "context_temp_bucket": self.context_temp_bucket,
            "context_window_state": self.context_window_state,
            "context_door_lock": self.context_door_lock,
            "context_approach_unlock": self.context_approach_unlock,
            "raw_value_stats_json": json.dumps(self.raw_value_stats),
            "member_fact_ids_json": json.dumps(self.member_fact_ids),
            "scene_card_id": self.scene_card_id,
            "created_at": self.created_at.isoformat(),
        }


def _row_get(row, key):
    """sqlite3.Row 对未 ALTER 前的 DB 缺列时会 IndexError，这里防御性取值。"""
    try:
        return row[key]
    except (IndexError, KeyError):
        return None
