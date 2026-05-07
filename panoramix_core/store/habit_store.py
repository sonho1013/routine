"""
HabitStore — habits 表的 SQLite 薄封装

职责：
  - batch_id 原子分配（事务内 MAX + 1）
  - insert_many / get_by_batch
  - 按 retain_n 保留最近 N 批的滑动窗口清理
"""
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

from panoramix_core.models.habit import Habit
from panoramix_core.store.sqlite_connection import get_connection


class HabitStore:
    def __init__(self, username: str, db_path: Optional[str] = None):
        self.username = username
        self.db_path = db_path
        self._conn = get_connection(db_path)

    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ── batch_id ──

    def get_latest_batch_id(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(batch_id), 0) AS max_bid FROM habits WHERE username = ?",
            (self.username,),
        ).fetchone()
        return int(row["max_bid"])

    def peek_next_batch_id(self) -> int:
        """读-only 窥视。真正分配用 allocate_next_batch_id 在事务内做。"""
        return self.get_latest_batch_id() + 1

    def allocate_next_batch_id(self, *, conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(batch_id), 0) AS max_bid FROM habits WHERE username = ?",
            (self.username,),
        ).fetchone()
        return int(row["max_bid"]) + 1

    # ── 写入 ──

    def insert_many(
        self, habits: List[Habit], batch_id: int, *, conn: sqlite3.Connection,
    ) -> None:
        if not habits:
            return
        rows = []
        for h in habits:
            # 确保 batch_id 一致（防御性）
            if h.batch_id != batch_id:
                h = h.model_copy(update={"batch_id": batch_id})
            rows.append(h.to_row_dict())
        conn.executemany(
            """
            INSERT INTO habits (
                habit_id, username, batch_id, text,
                signal_category, signal_name, structural_key,
                context_time_bucket, context_vehicle_state,
                context_geofence, context_weekday,
                context_poi_type, context_wiper_state, context_temp_bucket,
                context_window_state, context_door_lock, context_approach_unlock,
                raw_value_stats_json, member_fact_ids_json,
                scene_card_id, created_at
            ) VALUES (
                :habit_id, :username, :batch_id, :text,
                :signal_category, :signal_name, :structural_key,
                :context_time_bucket, :context_vehicle_state,
                :context_geofence, :context_weekday,
                :context_poi_type, :context_wiper_state, :context_temp_bucket,
                :context_window_state, :context_door_lock, :context_approach_unlock,
                :raw_value_stats_json, :member_fact_ids_json,
                :scene_card_id, :created_at
            )
            """,
            rows,
        )

    def update_scene_card_id(
        self, habit_ids: List[str], scene_card_id: str, *, conn: sqlite3.Connection,
    ) -> None:
        """批处理完成后回填每个 habit 归属的 scene_card_id"""
        conn.executemany(
            "UPDATE habits SET scene_card_id = ? WHERE habit_id = ?",
            [(scene_card_id, h) for h in habit_ids],
        )

    # ── 清理 ──

    def delete_old_batches(
        self, current_batch_id: int, retain_n: int = 3,
        *, conn: sqlite3.Connection,
    ) -> int:
        """
        保留最近 retain_n 批，删除更早的。
        retain_n=3 且 current_batch_id=7 → 保留 5,6,7 → 删除 < 5
        """
        cutoff = current_batch_id - retain_n + 1
        cur = conn.execute(
            "DELETE FROM habits WHERE username = ? AND batch_id < ?",
            (self.username, cutoff),
        )
        return cur.rowcount

    # ── 查询 ──

    def get_by_batch(self, batch_id: int) -> List[Habit]:
        rows = self._conn.execute(
            """
            SELECT * FROM habits
             WHERE username = ? AND batch_id = ?
             ORDER BY created_at
            """,
            (self.username, batch_id),
        ).fetchall()
        return [Habit.from_row(r) for r in rows]

    def get_latest_batch_habits(self) -> List[Habit]:
        latest = self.get_latest_batch_id()
        if latest == 0:
            return []
        return self.get_by_batch(latest)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
