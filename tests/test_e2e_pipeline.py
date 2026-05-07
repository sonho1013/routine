"""
端到端管线测试：mockup data → Fact → ChromaDB → Hybrid DBSCAN → 习惯检测

测试步骤：
1. 从 step3_hybrid_dbscan_result.json 加载已验证的 mockup 数据
2. 构建 Fact 对象 + StructuredContext
3. 通过 FactStoreChroma 存入 ChromaDB (自动生成 embedding)
4. 从 ChromaDB 读回 Facts，验证数据完整性
5. 获取 facts_with_embeddings，运行 HabitsDetector hybrid DBSCAN
6. 输出聚类结果，与已验证的 cluster 标签对比
"""
import json
import os
import sys
import shutil
import logging
import tempfile
from datetime import datetime, timedelta

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 在 import panoramix_core 模块之前设置临时存储目录，避免写入正式路径
_TMP_DIR = tempfile.mkdtemp(prefix="habit_e2e_test_")
os.environ["MEMORY_DIR"] = _TMP_DIR

import panoramix_core.config as _cfg
_cfg.MEMORY_DIR = _TMP_DIR

from panoramix_core.config import (
    HYBRID_ALPHA, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, HABIT_CANDIDATE_FACT_TYPES,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources
from panoramix_core.store.fact_store_chroma import FactStoreChroma
from panoramix_core.clustering.habits_detector import HabitsDetector, context_distance

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

import pytest

TEST_USER = "test_e2e"
DATA_FILE = os.path.join(os.path.dirname(__file__), "step3_hybrid_dbscan_result.json")


def load_mockup_facts():
    """从验证结果 JSON 构建 Fact 对象列表，保留 _scene_label 和 expected cluster"""
    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    facts = []
    expected_clusters = []
    for i, item in enumerate(data["facts"]):
        ctx = StructuredContext(
            time_bucket=item["context"]["time_bucket"],
            hour=item["context"]["hour"],
            weekday=item["context"]["weekday"],
            vehicle_state=item["context"]["vehicle_state"],
            geofence=item["context"].get("geofence"),
        )
        fact = Fact(
            text=item["text"],
            type=FactType.PREF,  # 所有 mockup facts 用 PREF 类型
            durability=FactDurability.LONG_TERM,
            time_stamp=datetime(2025, 10, 6) + timedelta(days=i % 5),
            source=FactSources.SIGNAL,
            context=ctx,
        )
        facts.append(fact)
        expected_clusters.append({
            "fact_id": fact.id,
            "text": item["text"],
            "_scene_label": item.get("_scene_label", "unknown"),
            "expected_cluster": item.get("cluster", -1),
        })

    return facts, expected_clusters


# ── Pytest fixtures ──

@pytest.fixture(scope="module")
def e2e_facts():
    """加载 mockup facts 和 expected clusters"""
    return load_mockup_facts()


@pytest.fixture(scope="module")
def e2e_store(e2e_facts):
    """创建 FactStoreChroma，存入 facts，返回 (store, facts, expected_clusters)"""
    facts, expected_clusters = e2e_facts
    store = FactStoreChroma(TEST_USER)
    store.store_facts(TEST_USER, facts)
    yield store, facts, expected_clusters
    store.close()


# ── Tests ──

def test_step1_store_facts(e2e_store):
    """Step 1: 将 Facts 存入 ChromaDB"""
    store, facts, _ = e2e_store
    stored = store.get_facts(TEST_USER)
    assert len(stored) == len(facts), f"Count mismatch: stored {len(facts)}, got {len(stored)}"


def test_step2_verify_metadata(e2e_store, e2e_facts):
    """Step 2: 验证 metadata 往返完整性 (text, type, context fields)"""
    store, _, _ = e2e_store
    original_facts, _ = e2e_facts
    stored_facts = store.get_facts(TEST_USER)

    original_by_id = {f.id: f for f in original_facts}
    errors = []

    for sf in stored_facts:
        orig = original_by_id.get(sf.id)
        if orig is None:
            errors.append(f"  Unknown fact ID: {sf.id}")
            continue

        if sf.text != orig.text:
            errors.append(f"  Text mismatch for {sf.id}: '{sf.text}' vs '{orig.text}'")

        if sf.type != orig.type:
            errors.append(f"  Type mismatch for {sf.id}: {sf.type} vs {orig.type}")

        for field in ["time_bucket", "vehicle_state", "geofence", "weekday", "hour"]:
            stored_val = getattr(sf.context, field)
            orig_val = getattr(orig.context, field)
            if stored_val != orig_val:
                errors.append(
                    f"  Context.{field} mismatch for {sf.id}: {stored_val} vs {orig_val}"
                )

    if errors:
        raise AssertionError(f"Metadata roundtrip failed with {len(errors)} errors:\n" + "\n".join(errors[:10]))


def test_step3_get_facts_with_embeddings(e2e_store):
    """Step 3: 获取 facts + embeddings，验证 embedding 维度"""
    store, _, _ = e2e_store
    items = store.get_facts_with_embeddings(TEST_USER)

    assert len(items) > 0, "No facts with embeddings returned"

    dims = set()
    for item in items:
        emb = item["embedding"]
        assert emb is not None, f"Null embedding for fact {item['fact'].id}"
        dims.add(len(emb))

    assert len(dims) == 1, f"Inconsistent embedding dimensions: {dims}"
    dim = dims.pop()
    assert dim > 0, "Zero-dimensional embeddings"


def test_step4_hybrid_dbscan(e2e_store):
    """Step 4: 运行 HabitsDetector hybrid DBSCAN，对比预期聚类"""
    store, _, expected_clusters = e2e_store
    items = store.get_facts_with_embeddings(TEST_USER)

    detector = HabitsDetector(llm_client=None)
    new_habits, ids_to_delete = detector.detect_habits(items)

    assert len(new_habits) > 0, "No habits detected"

    clustered_facts = [item for item in items if item["fact"].id in ids_to_delete]
    assert len(clustered_facts) > 0, "No facts were clustered"


def test_step5_delete_and_store_habits(e2e_store):
    """Step 5: 删除原始 facts，存入合成 habits — 完整 lifecycle"""
    store, _, _ = e2e_store
    items = store.get_facts_with_embeddings(TEST_USER)

    detector = HabitsDetector(llm_client=None)
    new_habits, ids_to_delete = detector.detect_habits(items)

    count_before = len(store.get_facts(TEST_USER))

    if ids_to_delete:
        store.delete_facts(TEST_USER, ids_to_delete)
    if new_habits:
        store.store_facts(TEST_USER, new_habits)

    remaining = store.get_facts(TEST_USER)
    expected_count = count_before - len(ids_to_delete) + len(new_habits)
    assert len(remaining) == expected_count, (
        f"Count mismatch: expected {expected_count}, got {len(remaining)}"
    )

    habit_facts = [f for f in remaining if f.type == FactType.HABIT]
    assert len(habit_facts) > 0, "No HABIT type facts in store after lifecycle"


def main():
    log.info("╔══════════════════════════════════════════════╗")
    log.info("║  E2E Pipeline Test: Fact → Chroma → DBSCAN  ║")
    log.info("╚══════════════════════════════════════════════╝")

    tmp_dir = _TMP_DIR
    log.info(f"Using temp storage: {tmp_dir}")

    try:
        # 加载数据
        facts, expected_clusters = load_mockup_facts()
        log.info(f"Loaded {len(facts)} facts from {DATA_FILE}")

        # 初始化 store
        store = FactStoreChroma(TEST_USER)

        # 执行测试步骤
        stored_facts = test_step1_store_facts(store, facts)
        test_step2_verify_metadata(stored_facts, facts)
        items = test_step3_get_facts_with_embeddings(store)
        new_habits, ids_to_delete = test_step4_hybrid_dbscan(items, expected_clusters)
        test_step5_delete_and_store_habits(store, new_habits, ids_to_delete)

        store.close()

        log.info("")
        log.info("════════════════════════════════════════")
        log.info("  ALL 5 STEPS PASSED — Pipeline works!")
        log.info("════════════════════════════════════════")

    except Exception as e:
        log.error(f"\n  TEST FAILED: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info(f"Cleaned up temp storage: {tmp_dir}")


if __name__ == "__main__":
    main()
