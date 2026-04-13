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
            "raw_value_stats_json": json.dumps(self.raw_value_stats),
            "member_fact_ids_json": json.dumps(self.member_fact_ids),
            "scene_card_id": self.scene_card_id,
            "created_at": self.created_at.isoformat(),
        }
