"""
SceneCardStore — 场景卡状态机的 SQLite 薄封装

所有 accepted 卡的不变式（不可变 display_name/frozen/structural_key）
在此层强制。违反即抛 SceneCardImmutableError。
"""
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional

from panoramix_core.models.scene_card import SceneCard
from panoramix_core.store.sqlite_connection import get_connection

log = logging.getLogger(__name__)


class SceneCardImmutableError(Exception):
    """尝试修改 accepted 卡的不可变字段，或对错状态做状态转换"""


class SceneCardStore:
    def __init__(self, username: str, db_path: Optional[str] = None):
        self.username = username
        self.db_path = db_path
        self._conn = get_connection(db_path)

    # ── 事务 ──

    @contextmanager
    def transaction(self):
        """BEGIN IMMEDIATE; yield conn; COMMIT or ROLLBACK"""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── 写入 ──

    def insert_pending(
        self, card: SceneCard, batch_id: int, *, conn: sqlite3.Connection,
    ) -> None:
        assert card.status == "pending"
        self._insert(card, conn)

    def insert_recommendation(
        self, card: SceneCard, batch_id: int,
        supersede_candidate_for: list[str], *, conn: sqlite3.Connection,
    ) -> None:
        assert card.status == "recommendation"
        card.supersede_candidate_for = supersede_candidate_for
        self._insert(card, conn)

    def _insert(self, card: SceneCard, conn: sqlite3.Connection) -> None:
        row = card.to_row_dict()
        conn.execute(
            """
            INSERT INTO scene_cards (
                card_id, username, status, structural_key, display_name,
                frozen_content_snapshot_json, content_snapshot_json,
                supersede_candidate_for_json,
                first_seen_batch_id, last_reinforced_batch_id,
                created_at, accepted_at, retired_at
            ) VALUES (
                :card_id, :username, :status, :structural_key, :display_name,
                :frozen_content_snapshot_json, :content_snapshot_json,
                :supersede_candidate_for_json,
                :first_seen_batch_id, :last_reinforced_batch_id,
                :created_at, :accepted_at, :retired_at
            )
            """,
            row,
        )

    def upsert_pending_by_structural_key(
        self, structural_key: str, card_data: dict, batch_id: int,
        *, conn: sqlite3.Connection,
    ) -> str:
        """
        Q8.3: 按 structural_key 在 pending 中 upsert。
        accepted / retired / recommendation 的同 key 卡不受影响。
        """
        row = conn.execute(
            """
            SELECT card_id, first_seen_batch_id FROM scene_cards
             WHERE username = ? AND status = 'pending'
               AND structural_key = ?
             LIMIT 1
            """,
            (self.username, structural_key),
        ).fetchone()

        if row is not None:
            conn.execute(
                """
                UPDATE scene_cards
                   SET display_name = ?,
                       content_snapshot_json = ?,
                       last_reinforced_batch_id = ?
                 WHERE card_id = ?
                """,
                (
                    card_data["display_name"],
                    json.dumps(card_data.get("content_snapshot")),
                    batch_id,
                    row["card_id"],
                ),
            )
            return row["card_id"]

        new_card = SceneCard(
            username=self.username,
            status="pending",
            structural_key=structural_key,
            display_name=card_data["display_name"],
            content_snapshot=card_data.get("content_snapshot"),
            first_seen_batch_id=batch_id,
            last_reinforced_batch_id=batch_id,
            created_at=datetime.now(),
        )
        self._insert(new_card, conn)
        return new_card.card_id

    def upsert_recommendation_by_target(
        self, target_accepted_id: str, card_data: dict, batch_id: int,
        *, conn: sqlite3.Connection,
    ) -> str:
        """
        Q8.2: 查找 "supersede_candidate_for 包含 target_accepted_id" 的
        现存 recommendation；命中 upsert，未命中 insert。
        """
        rows = conn.execute(
            """
            SELECT card_id, supersede_candidate_for_json, first_seen_batch_id
              FROM scene_cards
             WHERE username = ? AND status = 'recommendation'
            """,
            (self.username,),
        ).fetchall()

        existing_id: Optional[str] = None
        for row in rows:
            targets = json.loads(row["supersede_candidate_for_json"] or "[]")
            if target_accepted_id in targets:
                existing_id = row["card_id"]
                break

        if existing_id is not None:
            conn.execute(
                """
                UPDATE scene_cards
                   SET display_name = ?,
                       content_snapshot_json = ?,
                       structural_key = ?,
                       last_reinforced_batch_id = ?
                 WHERE card_id = ?
                """,
                (
                    card_data["display_name"],
                    json.dumps(card_data.get("content_snapshot")),
                    card_data["structural_key"],
                    batch_id,
                    existing_id,
                ),
            )
            return existing_id

        new_card = SceneCard(
            username=self.username,
            status="recommendation",
            structural_key=card_data["structural_key"],
            display_name=card_data["display_name"],
            content_snapshot=card_data.get("content_snapshot"),
            supersede_candidate_for=[target_accepted_id],
            first_seen_batch_id=batch_id,
            last_reinforced_batch_id=batch_id,
            created_at=datetime.now(),
        )
        self._insert(new_card, conn)
        return new_card.card_id

    # ── 状态转换 ──

    def accept(self, card_id: str) -> None:
        """
        pending / recommendation → accepted
        冻结 display_name + structural_key + content→frozen。
        若原为 recommendation：级联 retire 其 supersede_candidate_for 指向的 accepted 卡。
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")

            if row["status"] not in ("pending", "recommendation"):
                raise SceneCardImmutableError(
                    f"accept() requires pending/recommendation; got {row['status']}"
                )

            # 冻结：content_snapshot → frozen_content_snapshot
            content_json = row["content_snapshot_json"] or "{}"
            now_iso = datetime.now().isoformat()

            conn.execute(
                """
                UPDATE scene_cards
                   SET status = 'accepted',
                       frozen_content_snapshot_json = ?,
                       content_snapshot_json = NULL,
                       accepted_at = ?
                 WHERE card_id = ?
                """,
                (content_json, now_iso, card_id),
            )

            # 级联 retire 旧 accepted 卡
            if row["status"] == "recommendation":
                targets = json.loads(row["supersede_candidate_for_json"] or "[]")
                for target_id in targets:
                    conn.execute(
                        """
                        UPDATE scene_cards
                           SET status = 'retired',
                               retired_at = ?
                         WHERE card_id = ? AND username = ?
                               AND status = 'accepted'
                        """,
                        (now_iso, target_id, self.username),
                    )

    def retire(self, card_id: str) -> None:
        """accepted → retired（仅允许由用户主动调用或 accept() 内部级联）"""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")
            if row["status"] != "accepted":
                raise SceneCardImmutableError(
                    f"retire() requires accepted; got {row['status']}"
                )
            conn.execute(
                "UPDATE scene_cards SET status='retired', retired_at=? WHERE card_id=?",
                (datetime.now().isoformat(), card_id),
            )

    def reject_recommendation(self, card_id: str) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")
            if row["status"] != "recommendation":
                raise SceneCardImmutableError(
                    f"reject_recommendation() requires recommendation; got {row['status']}"
                )
            conn.execute(
                "DELETE FROM scene_cards WHERE card_id = ?", (card_id,),
            )

    def dismiss_pending(self, card_id: str) -> None:
        """用户主动忽略 pending 卡"""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
                (card_id, self.username),
            ).fetchone()
            if row is None:
                raise ValueError(f"card not found: {card_id}")
            if row["status"] != "pending":
                raise SceneCardImmutableError(
                    f"dismiss_pending() requires pending; got {row['status']}"
                )
            conn.execute(
                "DELETE FROM scene_cards WHERE card_id = ?", (card_id,),
            )

    # ── 清理 ──

    def delete_stale_pending(
        self, current_batch_id: int, *, conn: sqlite3.Connection,
    ) -> int:
        cur = conn.execute(
            """
            DELETE FROM scene_cards
             WHERE username = ? AND status = 'pending'
               AND last_reinforced_batch_id < ?
            """,
            (self.username, current_batch_id),
        )
        return cur.rowcount

    def delete_stale_recommendation(
        self, current_batch_id: int, *, conn: sqlite3.Connection,
    ) -> int:
        cur = conn.execute(
            """
            DELETE FROM scene_cards
             WHERE username = ? AND status = 'recommendation'
               AND last_reinforced_batch_id < ?
            """,
            (self.username, current_batch_id),
        )
        return cur.rowcount

    def update_last_reinforced(
        self, card_id: str, batch_id: int, *, conn: sqlite3.Connection,
    ) -> None:
        """
        只允许 reinforce accepted 卡（其它状态的 reinforce 走 upsert_* 路径）。
        不修改 display_name / frozen / structural_key。
        """
        row = conn.execute(
            "SELECT status FROM scene_cards WHERE card_id = ? AND username = ?",
            (card_id, self.username),
        ).fetchone()
        if row is None:
            raise ValueError(f"card not found: {card_id}")
        if row["status"] != "accepted":
            raise SceneCardImmutableError(
                f"update_last_reinforced() targets accepted only; got {row['status']}"
            )
        conn.execute(
            "UPDATE scene_cards SET last_reinforced_batch_id = ? WHERE card_id = ?",
            (batch_id, card_id),
        )

    # ── 查询 ──

    def get_by_status(self, status: str) -> List[SceneCard]:
        rows = self._conn.execute(
            """
            SELECT * FROM scene_cards
             WHERE username = ? AND status = ?
             ORDER BY created_at
            """,
            (self.username, status),
        ).fetchall()
        return [SceneCard.from_row(r) for r in rows]

    def get_by_structural_key(
        self, structural_key: str, status: Optional[str] = None,
    ) -> List[SceneCard]:
        if status is None:
            rows = self._conn.execute(
                """
                SELECT * FROM scene_cards
                 WHERE username = ? AND structural_key = ?
                 ORDER BY created_at
                """,
                (self.username, structural_key),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT * FROM scene_cards
                 WHERE username = ? AND structural_key = ? AND status = ?
                 ORDER BY created_at
                """,
                (self.username, structural_key, status),
            ).fetchall()
        return [SceneCard.from_row(r) for r in rows]

    def get_all_accepted(self) -> List[SceneCard]:
        return self.get_by_status("accepted")

    def find_recommendation_by_target(
        self, accepted_card_id: str,
    ) -> Optional[SceneCard]:
        rows = self._conn.execute(
            """
            SELECT * FROM scene_cards
             WHERE username = ? AND status = 'recommendation'
            """,
            (self.username,),
        ).fetchall()
        for row in rows:
            targets = json.loads(row["supersede_candidate_for_json"] or "[]")
            if accepted_card_id in targets:
                return SceneCard.from_row(row)
        return None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
