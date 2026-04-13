"""HabitStore 单元测试 — CRUD + batch 保留策略"""
import os
import tempfile
from datetime import datetime

import pytest

from panoramix_core.models.habit import Habit
from panoramix_core.store.habit_store import HabitStore
from panoramix_core.store.sqlite_connection import reset_schema_cache


@pytest.fixture
def tmp_db():
    reset_schema_cache()
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def store(tmp_db):
    return HabitStore(username="tester", db_path=tmp_db)


def _make_habit(batch_id: int, signal_name: str = "hvac_temp_target") -> Habit:
    return Habit(
        username="tester",
        batch_id=batch_id,
        text=f"set temperature to 22 (batch {batch_id})",
        signal_category="numeric",
        signal_name=signal_name,
        structural_key="sk_home_morning",
        context_time_bucket="early_morning",
        context_vehicle_state="engine_started",
        context_geofence="home",
        context_weekday=1,
        raw_value_stats={"type": "numeric", "mean": 22.0, "count": 10},
        member_fact_ids=["pref-1", "pref-2"],
        created_at=datetime.now(),
    )


def test_allocate_next_batch_id_on_empty(store):
    with store.transaction() as conn:
        bid = store.allocate_next_batch_id(conn=conn)
    assert bid == 1


def test_allocate_next_batch_id_increments(store):
    with store.transaction() as conn:
        store.insert_many([_make_habit(5)], batch_id=5, conn=conn)

    with store.transaction() as conn:
        bid = store.allocate_next_batch_id(conn=conn)
    assert bid == 6


def test_insert_many_and_get_by_batch(store):
    habits = [_make_habit(1), _make_habit(1, signal_name="media_source")]
    with store.transaction() as conn:
        store.insert_many(habits, batch_id=1, conn=conn)

    got = store.get_by_batch(1)
    assert len(got) == 2
    assert {h.signal_name for h in got} == {"hvac_temp_target", "media_source"}


def test_delete_old_batches_retain_3(store):
    for bid in range(1, 8):  # batches 1..7
        with store.transaction() as conn:
            store.insert_many([_make_habit(bid)], batch_id=bid, conn=conn)

    with store.transaction() as conn:
        deleted = store.delete_old_batches(current_batch_id=7, retain_n=3, conn=conn)

    # 保留 batch 5, 6, 7 → 删除 1, 2, 3, 4
    assert deleted == 4
    remaining_batches = {h.batch_id for h in store.get_by_batch(5)}
    assert store.get_by_batch(1) == []
    assert store.get_by_batch(4) == []
    assert len(store.get_by_batch(5)) == 1
    assert len(store.get_by_batch(7)) == 1


def test_get_latest_batch_id(store):
    with store.transaction() as conn:
        store.insert_many([_make_habit(1)], batch_id=1, conn=conn)
    with store.transaction() as conn:
        store.insert_many([_make_habit(5)], batch_id=5, conn=conn)

    assert store.get_latest_batch_id() == 5


def test_get_latest_batch_id_empty(store):
    assert store.get_latest_batch_id() == 0
