"""
端到端多批次 drift 测试 — spec §10.2 关键场景

覆盖用户主权不变式：
1. Drift → recommendation 非替换（accepted 卡 frozen 字段不变）
2. Recommendation 去重（连续多批次同 drift 产生的 rec ID 稳定）
3. Accept recommendation 级联 retire 旧 accepted
4. Accepted 卡 dormant 不退役（多批次未匹配仍保留）
5. Pending 陈腐清理（单批次未重现即删除）
"""
import os
import tempfile

import pytest

from engine.habit_engine import HabitDemoEngine
from panoramix_core.store.scene_card_store import SceneCardStore
from panoramix_core.store.sqlite_connection import reset_schema_cache


@pytest.fixture
def isolated_engine(monkeypatch, tmp_path):
    """
    为每个测试起一个全新的 engine，隔离 SQLite 和 Chroma。

    隔离策略：
      - SQLite: monkeypatch DEFAULT_DB_PATH 到临时文件
      - Chroma: 每个测试用一个独立的 username，在 fixture 结束时调 engine.reset()
      - 使用 llm_client=None 让 reword_cluster 走降级路径（返回 fact_texts[0]），
        e2e 不依赖真实 GPT 调用
    """
    import uuid as _uuid
    reset_schema_cache()

    db_path = str(tmp_path / "habit_memory.db")
    monkeypatch.setattr(
        "panoramix_core.store.sqlite_connection.DEFAULT_DB_PATH",
        db_path,
    )

    unique_user = f"e2e_{_uuid.uuid4().hex[:8]}"
    engine = HabitDemoEngine(
        username=unique_user,
        llm_client=None,
        required_consecutive=3,  # 降低门槛方便测试
    )
    try:
        yield engine
    finally:
        try:
            engine.reset()
        except Exception:
            pass
        try:
            engine.close()
        except Exception:
            pass


def _events_morning_home_temp(temp_value: float, day_offset: int = 0):
    """生成一批"早上在家 hvac 温度 = temp_value"的 mockup events。

    返回一个 events list，每个 event 含 signals 数组（符合现有
    signals_to_facts() 的 event schema）。
    """
    from datetime import datetime, timedelta
    base = datetime(2026, 4, 1, 7, 0, 0) + timedelta(days=day_offset)
    events = []
    for i in range(6):
        ts = (base + timedelta(minutes=i * 30)).isoformat()
        events.append({
            "timestamp": ts,
            "user": "e2e",
            "signals": [
                {"signal": "hvac_temp_target", "value": temp_value},
                {"signal": "gps_latitude", "value": 48.8566},
                {"signal": "gps_longitude", "value": 2.3522},
                {"signal": "engine_status", "value": "on"},
                {"signal": "gear_position", "value": "P"},
                {"signal": "vehicle_speed", "value": 0},
            ],
        })
    return events


def _batch(temp_value: float, start_day: int, end_day: int):
    """组装多天的 events 成一个 batch"""
    out = []
    for d in range(start_day, end_day):
        out.extend(_events_morning_home_temp(temp_value, day_offset=d))
    return out


def _media_events(day_offset: int):
    """生成一批"晚上开车听音乐"的 mockup events"""
    from datetime import datetime, timedelta
    base = datetime(2026, 5, 1, 18, 0, 0) + timedelta(days=day_offset)
    events = []
    for i in range(4):
        ts = (base + timedelta(minutes=i * 15)).isoformat()
        events.append({
            "timestamp": ts,
            "user": "e2e",
            "signals": [
                {"signal": "media_source", "value": "music", "content_id": "jazz"},
                {"signal": "gps_latitude", "value": 0.0},
                {"signal": "gps_longitude", "value": 0.0},
                {"signal": "engine_status", "value": "on"},
                {"signal": "gear_position", "value": "D"},
                {"signal": "vehicle_speed", "value": 60},
            ],
        })
    return events


def _media_batch(start_day: int, end_day: int):
    out = []
    for d in range(start_day, end_day):
        out.extend(_media_events(d))
    return out


# ── Scenario 1: Drift → recommendation 非替换 ──

def test_drift_generates_recommendation_keeps_accepted_frozen(isolated_engine):
    engine = isolated_engine

    # Batch 1: 温度 22°C，用户接受产出的 pending 卡
    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))

    pendings = engine.scene_card_store.get_by_status("pending")
    assert len(pendings) >= 1
    morning_card = pendings[0]
    morning_card_id = morning_card.card_id
    frozen_structural_key_before = morning_card.structural_key

    engine.scene_card_store.accept(morning_card_id)

    accepted_before = engine.scene_card_store.get_by_status("accepted")
    assert len(accepted_before) == 1
    frozen_name_before = accepted_before[0].display_name
    frozen_snapshot_before = accepted_before[0].frozen_content_snapshot

    # Batch 2: 温度 25°C (+3 > 1.5 阈值 → drift)
    engine.ingest_signal_batch(_batch(25.0, start_day=5, end_day=10))

    # Accepted 卡未变
    accepted_after = engine.scene_card_store.get_by_status("accepted")
    assert len(accepted_after) == 1
    assert accepted_after[0].card_id == morning_card_id
    assert accepted_after[0].display_name == frozen_name_before
    assert accepted_after[0].frozen_content_snapshot == frozen_snapshot_before
    assert accepted_after[0].structural_key == frozen_structural_key_before

    # 新产生了 recommendation 卡
    recs = engine.scene_card_store.get_by_status("recommendation")
    assert len(recs) == 1
    assert morning_card_id in recs[0].supersede_candidate_for


# ── Scenario 2: recommendation 去重 ──

def test_recommendation_deduped_across_batches(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pending = engine.scene_card_store.get_by_status("pending")[0]
    engine.scene_card_store.accept(pending.card_id)

    engine.ingest_signal_batch(_batch(25.0, start_day=5, end_day=10))
    recs1 = engine.scene_card_store.get_by_status("recommendation")
    assert len(recs1) == 1
    rec1_id = recs1[0].card_id
    rec1_reinforced = recs1[0].last_reinforced_batch_id

    engine.ingest_signal_batch(_batch(25.5, start_day=10, end_day=15))
    recs2 = engine.scene_card_store.get_by_status("recommendation")
    assert len(recs2) == 1
    assert recs2[0].card_id == rec1_id
    assert recs2[0].last_reinforced_batch_id > rec1_reinforced


# ── Scenario 3: Accept recommendation 级联 retire ──

def test_accept_recommendation_retires_old_accepted(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pending = engine.scene_card_store.get_by_status("pending")[0]
    old_accepted_id = pending.card_id
    engine.scene_card_store.accept(old_accepted_id)

    engine.ingest_signal_batch(_batch(25.0, start_day=5, end_day=10))
    rec = engine.scene_card_store.get_by_status("recommendation")[0]

    engine.scene_card_store.accept(rec.card_id)

    still_accepted = engine.scene_card_store.get_by_status("accepted")
    assert len(still_accepted) == 1
    assert still_accepted[0].card_id == rec.card_id

    retired = engine.scene_card_store.get_by_status("retired")
    assert any(r.card_id == old_accepted_id for r in retired)


# ── Scenario 4: Accepted 卡 dormant 不退役 ──

def test_dormant_accepted_card_not_retired(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pending = engine.scene_card_store.get_by_status("pending")[0]
    accepted_id = pending.card_id
    engine.scene_card_store.accept(accepted_id)

    # 后续 5 批完全无关的信号（media_source）
    for d in range(5):
        engine.ingest_signal_batch(_media_events(day_offset=d))

    still = engine.scene_card_store.get_by_status("accepted")
    assert any(c.card_id == accepted_id for c in still)


# ── Scenario 5: Pending 陈腐清理 ──

def test_stale_pending_cleaned_up(isolated_engine):
    engine = isolated_engine

    engine.ingest_signal_batch(_batch(22.0, start_day=0, end_day=5))
    pendings1 = engine.scene_card_store.get_by_status("pending")
    assert len(pendings1) >= 1
    old_temp_ids = {p.card_id for p in pendings1}

    # Batch 2: 完全不同的 signal → 旧 pending 不会被 reinforce
    engine.ingest_signal_batch(_media_batch(start_day=0, end_day=5))

    pendings2 = engine.scene_card_store.get_by_status("pending")
    new_ids = {p.card_id for p in pendings2}
    # 原来的 temperature pending 应已被清理；可能存在新的 media pending
    assert old_temp_ids.isdisjoint(new_ids)
