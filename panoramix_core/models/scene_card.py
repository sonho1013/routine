"""SceneCard Pydantic 模型 — 对应 scene_cards SQLite 表"""
import json
import uuid
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


SceneCardStatus = Literal['pending', 'accepted', 'recommendation', 'retired']


class SceneCard(BaseModel):
    card_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    status: SceneCardStatus
    structural_key: str
    display_name: str

    # 仅 accepted 卡使用（冻结快照）
    frozen_content_snapshot: Optional[dict] = None

    # 仅 pending/recommendation 卡使用
    content_snapshot: Optional[dict] = None

    # 仅 recommendation 卡使用
    supersede_candidate_for: Optional[list[str]] = None

    first_seen_batch_id: int
    last_reinforced_batch_id: int

    created_at: datetime
    accepted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None

    # ── SQLite row <-> model ──

    @classmethod
    def from_row(cls, row) -> "SceneCard":
        """row: sqlite3.Row 或 dict-like"""
        def j(k):
            v = row[k]
            return json.loads(v) if v else None

        def ts(k):
            v = row[k]
            return datetime.fromisoformat(v) if v else None

        return cls(
            card_id=row["card_id"],
            username=row["username"],
            status=row["status"],
            structural_key=row["structural_key"],
            display_name=row["display_name"],
            frozen_content_snapshot=j("frozen_content_snapshot_json"),
            content_snapshot=j("content_snapshot_json"),
            supersede_candidate_for=j("supersede_candidate_for_json"),
            first_seen_batch_id=row["first_seen_batch_id"],
            last_reinforced_batch_id=row["last_reinforced_batch_id"],
            created_at=ts("created_at"),
            accepted_at=ts("accepted_at"),
            retired_at=ts("retired_at"),
        )

    def to_row_dict(self) -> dict:
        return {
            "card_id": self.card_id,
            "username": self.username,
            "status": self.status,
            "structural_key": self.structural_key,
            "display_name": self.display_name,
            "frozen_content_snapshot_json":
                json.dumps(self.frozen_content_snapshot) if self.frozen_content_snapshot else None,
            "content_snapshot_json":
                json.dumps(self.content_snapshot) if self.content_snapshot else None,
            "supersede_candidate_for_json":
                json.dumps(self.supersede_candidate_for) if self.supersede_candidate_for else None,
            "first_seen_batch_id": self.first_seen_batch_id,
            "last_reinforced_batch_id": self.last_reinforced_batch_id,
            "created_at": self.created_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
        }
