"""SceneCardStore 单元测试 — 覆盖所有 API + accepted 不变式"""
import os
import tempfile
from datetime import datetime

import pytest

from panoramix_core.models.scene_card import SceneCard
from panoramix_core.store.scene_card_store import (
    SceneCardStore, SceneCardImmutableError,
)
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
    return SceneCardStore(username="tester", db_path=tmp_db)


def _make_pending_card(
    structural_key="sk_abc", display_name="Morning Home Routine",
    batch_id=1,
) -> SceneCard:
    return SceneCard(
        username="tester",
        status="pending",
        structural_key=structural_key,
        display_name=display_name,
        content_snapshot={"habits": []},
        first_seen_batch_id=batch_id,
        last_reinforced_batch_id=batch_id,
        created_at=datetime.now(),
    )


# ── 基本 CRUD ──

def test_insert_pending_and_get(store):
    card = _make_pending_card()
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)
    got = store.get_by_status("pending")
    assert len(got) == 1
    assert got[0].card_id == card.card_id


def test_upsert_pending_by_structural_key_hit(store):
    """同一 structural_key 的第二次 upsert 应更新 last_reinforced_batch_id"""
    card = _make_pending_card(structural_key="sk_x", batch_id=1)
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)

    with store.transaction() as conn:
        new_id = store.upsert_pending_by_structural_key(
            structural_key="sk_x",
            card_data={
                "display_name": "Updated Name",
                "content_snapshot": {"habits": ["h1"]},
            },
            batch_id=2,
            conn=conn,
        )

    assert new_id == card.card_id  # 同一 ID
    got = store.get_by_structural_key("sk_x", status="pending")
    assert len(got) == 1
    assert got[0].display_name == "Updated Name"
    assert got[0].last_reinforced_batch_id == 2
    assert got[0].first_seen_batch_id == 1


def test_upsert_pending_by_structural_key_miss(store):
    """无匹配 structural_key 时应 insert 新卡"""
    with store.transaction() as conn:
        new_id = store.upsert_pending_by_structural_key(
            structural_key="sk_new",
            card_data={
                "display_name": "Brand New",
                "content_snapshot": {"habits": []},
            },
            batch_id=1,
            conn=conn,
        )
    got = store.get_by_status("pending")
    assert len(got) == 1
    assert got[0].card_id == new_id
    assert got[0].first_seen_batch_id == 1


# ── accept 不变式 ──

def test_accept_pending_freezes_fields(store):
    card = _make_pending_card(display_name="Original")
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)

    store.accept(card.card_id)

    accepted = store.get_by_status("accepted")
    assert len(accepted) == 1
    a = accepted[0]
    assert a.display_name == "Original"
    assert a.frozen_content_snapshot is not None
    assert a.accepted_at is not None
    assert a.structural_key == card.structural_key


def test_accept_rejects_already_accepted(store):
    card = _make_pending_card()
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)
    store.accept(card.card_id)

    with pytest.raises(SceneCardImmutableError):
        store.accept(card.card_id)


def test_cannot_upsert_pending_into_accepted(store):
    """accepted 卡的 structural_key 不应被 upsert_pending 污染"""
    card = _make_pending_card(structural_key="sk_locked")
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)
    store.accept(card.card_id)

    with store.transaction() as conn:
        new_id = store.upsert_pending_by_structural_key(
            structural_key="sk_locked",
            card_data={"display_name": "X", "content_snapshot": {}},
            batch_id=2,
            conn=conn,
        )
    # 新 pending 卡应该是全新一张，不影响已 accepted 的
    assert new_id != card.card_id
    pending = store.get_by_status("pending")
    assert len(pending) == 1
    accepted = store.get_by_status("accepted")
    assert len(accepted) == 1
    assert accepted[0].card_id == card.card_id
    assert accepted[0].display_name != "X"


# ── recommendation 去重 (Q8.2) ──

def test_upsert_recommendation_by_target_dedup(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    # batch 2: 发现 drift → 插入一张 recommendation
    with store.transaction() as conn:
        rec1_id = store.upsert_recommendation_by_target(
            target_accepted_id=accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "Morning (updated temp)",
                "content_snapshot": {"drifted_signals": ["hvac_temp_target"]},
            },
            batch_id=2,
            conn=conn,
        )

    # batch 3: 又发现 drift → 应该 upsert 同一张 recommendation
    with store.transaction() as conn:
        rec2_id = store.upsert_recommendation_by_target(
            target_accepted_id=accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "Morning (updated temp v2)",
                "content_snapshot": {"drifted_signals": ["hvac_temp_target"]},
            },
            batch_id=3,
            conn=conn,
        )

    assert rec1_id == rec2_id
    recs = store.get_by_status("recommendation")
    assert len(recs) == 1
    assert recs[0].display_name == "Morning (updated temp v2)"
    assert recs[0].last_reinforced_batch_id == 3


# ── accept recommendation 级联 retire (Q8.1 决策 A) ──

def test_accept_recommendation_retires_target_accepted(store):
    old_accepted = _make_pending_card(structural_key="sk_a", display_name="Old")
    with store.transaction() as conn:
        store.insert_pending(old_accepted, batch_id=1, conn=conn)
    store.accept(old_accepted.card_id)

    with store.transaction() as conn:
        rec_id = store.upsert_recommendation_by_target(
            target_accepted_id=old_accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "New",
                "content_snapshot": {"drifted_signals": ["hvac_temp_target"]},
            },
            batch_id=2,
            conn=conn,
        )

    store.accept(rec_id)

    accepted = store.get_by_status("accepted")
    assert len(accepted) == 1
    assert accepted[0].card_id == rec_id
    assert accepted[0].display_name == "New"

    retired = store.get_by_status("retired")
    assert len(retired) == 1
    assert retired[0].card_id == old_accepted.card_id
    assert retired[0].retired_at is not None


# ── 陈腐清理 (Step 7, Step 8) ──

def test_delete_stale_pending(store):
    old = _make_pending_card(structural_key="sk_old")
    with store.transaction() as conn:
        store.insert_pending(old, batch_id=1, conn=conn)

    with store.transaction() as conn:
        count = store.delete_stale_pending(current_batch_id=2, conn=conn)

    assert count == 1
    assert store.get_by_status("pending") == []


def test_delete_stale_recommendation(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    with store.transaction() as conn:
        store.upsert_recommendation_by_target(
            target_accepted_id=accepted.card_id,
            card_data={
                "structural_key": "sk_a",
                "display_name": "Old rec",
                "content_snapshot": {},
            },
            batch_id=2,
            conn=conn,
        )

    with store.transaction() as conn:
        count = store.delete_stale_recommendation(current_batch_id=3, conn=conn)

    assert count == 1
    assert store.get_by_status("recommendation") == []


# ── accepted 永远不被陈腐清理 ──

def test_accepted_not_affected_by_stale_cleanup(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    # 多轮清理
    for batch in range(2, 10):
        with store.transaction() as conn:
            store.delete_stale_pending(current_batch_id=batch, conn=conn)
            store.delete_stale_recommendation(current_batch_id=batch, conn=conn)

    still_there = store.get_by_status("accepted")
    assert len(still_there) == 1
    assert still_there[0].card_id == accepted.card_id


# ── reinforce ──

def test_update_last_reinforced(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    with store.transaction() as conn:
        store.update_last_reinforced(accepted.card_id, batch_id=5, conn=conn)

    got = store.get_by_status("accepted")[0]
    assert got.last_reinforced_batch_id == 5
    # 但 display_name / frozen 不变
    assert got.display_name == "Morning Home Routine"


# ── 用户主动 retire ──

def test_retire_accepted_by_user(store):
    accepted = _make_pending_card(structural_key="sk_a")
    with store.transaction() as conn:
        store.insert_pending(accepted, batch_id=1, conn=conn)
    store.accept(accepted.card_id)

    store.retire(accepted.card_id)

    assert store.get_by_status("accepted") == []
    retired = store.get_by_status("retired")
    assert len(retired) == 1


def test_retire_non_accepted_raises(store):
    card = _make_pending_card()
    with store.transaction() as conn:
        store.insert_pending(card, batch_id=1, conn=conn)

    with pytest.raises(SceneCardImmutableError):
        store.retire(card.card_id)


# ── 多用户隔离 ──

def test_multi_user_isolation(tmp_db):
    reset_schema_cache()
    alice = SceneCardStore(username="alice", db_path=tmp_db)
    bob = SceneCardStore(username="bob", db_path=tmp_db)

    with alice.transaction() as conn:
        alice.insert_pending(
            SceneCard(
                username="alice", status="pending",
                structural_key="sk", display_name="Alice Card",
                content_snapshot={},
                first_seen_batch_id=1, last_reinforced_batch_id=1,
                created_at=datetime.now(),
            ),
            batch_id=1, conn=conn,
        )

    assert len(alice.get_by_status("pending")) == 1
    assert len(bob.get_by_status("pending")) == 0
