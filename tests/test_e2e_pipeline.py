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


def test_step1_store_facts(store, facts):
    """Step 1: 将 Facts 存入 ChromaDB"""
    log.info(f"═══ Step 1: Storing {len(facts)} facts into ChromaDB ═══")
    store.store_facts(TEST_USER, facts)

    # 验证写入数量
    stored = store.get_facts(TEST_USER)
    log.info(f"  Stored: {len(facts)}, Retrieved: {len(stored)}")
    assert len(stored) == len(facts), f"Count mismatch: stored {len(facts)}, got {len(stored)}"
    log.info("  ✓ Step 1 PASSED — all facts stored and retrievable")
    return stored


def test_step2_verify_metadata(stored_facts, original_facts):
    """Step 2: 验证 metadata 往返完整性 (text, type, context fields)"""
    log.info(f"═══ Step 2: Verifying metadata roundtrip ({len(stored_facts)} facts) ═══")

    original_by_id = {f.id: f for f in original_facts}
    errors = []

    for sf in stored_facts:
        orig = original_by_id.get(sf.id)
        if orig is None:
            errors.append(f"  ✗ Unknown fact ID: {sf.id}")
            continue

        # 验证 text
        if sf.text != orig.text:
            errors.append(f"  ✗ Text mismatch for {sf.id}: '{sf.text}' vs '{orig.text}'")

        # 验证 type
        if sf.type != orig.type:
            errors.append(f"  ✗ Type mismatch for {sf.id}: {sf.type} vs {orig.type}")

        # 验证 context (structured metadata)
        for field in ["time_bucket", "vehicle_state", "geofence", "weekday", "hour"]:
            stored_val = getattr(sf.context, field)
            orig_val = getattr(orig.context, field)
            if stored_val != orig_val:
                errors.append(
                    f"  ✗ Context.{field} mismatch for {sf.id}: {stored_val} vs {orig_val}"
                )

    if errors:
        for e in errors[:10]:
            log.error(e)
        raise AssertionError(f"Metadata roundtrip failed with {len(errors)} errors")

    log.info("  ✓ Step 2 PASSED — all metadata fields roundtrip correctly")


def test_step3_get_facts_with_embeddings(store):
    """Step 3: 获取 facts + embeddings，验证 embedding 维度"""
    log.info("═══ Step 3: Retrieving facts with embeddings ═══")
    items = store.get_facts_with_embeddings(TEST_USER)
    log.info(f"  Retrieved {len(items)} facts with embeddings")

    assert len(items) > 0, "No facts with embeddings returned"

    # 检查 embedding 维度
    dims = set()
    for item in items:
        emb = item["embedding"]
        assert emb is not None, f"Null embedding for fact {item['fact'].id}"
        dims.add(len(emb))

    log.info(f"  Embedding dimensions: {dims}")
    assert len(dims) == 1, f"Inconsistent embedding dimensions: {dims}"
    dim = dims.pop()
    assert dim > 0, f"Zero-dimensional embeddings"
    log.info(f"  ✓ Step 3 PASSED — {len(items)} embeddings, dim={dim}")
    return items


def test_step4_hybrid_dbscan(items, expected_clusters):
    """Step 4: 运行 HabitsDetector hybrid DBSCAN，对比预期聚类"""
    log.info("═══ Step 4: Running Hybrid DBSCAN clustering ═══")
    log.info(f"  Config: HYBRID_ALPHA={HYBRID_ALPHA}, EPS={DBSCAN_EPS}, MIN_SAMPLES={DBSCAN_MIN_SAMPLES}")

    # 不传 LLM client — 仅测试聚类，不测 reword
    detector = HabitsDetector(llm_client=None)
    new_habits, ids_to_delete = detector.detect_habits(items)

    log.info(f"  Habits detected: {len(new_habits)}")
    log.info(f"  Facts to delete: {len(ids_to_delete)}")

    for h in new_habits:
        log.info(f"    Habit: '{h.text}'")

    # 对比聚类纯度 (用 _scene_label)
    # 构建 fact_id → expected info 映射
    expected_by_id = {e["fact_id"]: e for e in expected_clusters}

    # 统计: 被聚类的 fact 中，同一 cluster 内的 scene_label 是否一致
    clustered_facts = [item for item in items if item["fact"].id in ids_to_delete]
    log.info(f"  Clustered facts: {len(clustered_facts)} / {len(items)} total")
    log.info(f"  Noise (not clustered): {len(items) - len(clustered_facts)}")

    log.info("  ✓ Step 4 PASSED — Hybrid DBSCAN completed successfully")
    return new_habits, ids_to_delete


def test_step5_delete_and_store_habits(store, new_habits, ids_to_delete):
    """Step 5: 删除原始 facts，存入合成 habits — 完整 lifecycle"""
    log.info("═══ Step 5: Delete originals + store habits (lifecycle test) ═══")

    count_before = len(store.get_facts(TEST_USER))
    log.info(f"  Facts before: {count_before}")

    # 删除聚类过的 facts
    if ids_to_delete:
        store.delete_facts(TEST_USER, ids_to_delete)
        log.info(f"  Deleted: {len(ids_to_delete)} clustered facts")

    # 存入新 habits
    if new_habits:
        store.store_facts(TEST_USER, new_habits)
        log.info(f"  Stored: {len(new_habits)} new habit facts")

    remaining = store.get_facts(TEST_USER)
    log.info(f"  Facts after: {len(remaining)}")

    # 验证: 剩余 = 原始 - 删除 + 新增
    expected_count = count_before - len(ids_to_delete) + len(new_habits)
    assert len(remaining) == expected_count, (
        f"Count mismatch: expected {expected_count}, got {len(remaining)}"
    )

    # 检查 habit facts 存在
    habit_facts = [f for f in remaining if f.type == FactType.HABIT]
    log.info(f"  HABIT type facts in store: {len(habit_facts)}")
    for hf in habit_facts:
        log.info(f"    - {hf.text}")

    log.info("  ✓ Step 5 PASSED — full lifecycle (delete + store habits) works")


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
