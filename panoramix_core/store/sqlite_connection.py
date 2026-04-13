"""
SQLite 连接工厂 — 共享 `storage/habit_memory.db` 的单文件

SceneCardStore + HabitStore 都通过 `get_connection(db_path)` 拿连接。
首次调用会自动执行 sqlite_schema.sql 初始化表结构。
"""
import logging
import os
import sqlite3
import threading
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "storage/habit_memory.db"

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "sqlite_schema.sql")
_initialized_paths: set[str] = set()
_init_lock = threading.Lock()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    返回新的 SQLite 连接。自动初始化 schema（每个 db_path 仅一次）。

    连接特性：
      - row_factory = sqlite3.Row
      - foreign_keys = ON
      - isolation_level = None（手动事务控制）
    """
    path = db_path or DEFAULT_DB_PATH
    _ensure_schema(path)

    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_schema(db_path: str) -> None:
    with _init_lock:
        if db_path in _initialized_paths:
            return

        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)

        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            ddl = f.read()

        init_conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            init_conn.executescript(ddl)
            log.info(f"SQLite schema initialized at {db_path}")
        finally:
            init_conn.close()

        _initialized_paths.add(db_path)


def reset_schema_cache() -> None:
    """测试用：清空 schema 初始化缓存"""
    with _init_lock:
        _initialized_paths.clear()
