"""
关键验证：Embedding 距离测试
=================================================
验证目标：
1. 同场景 Fact 的 Embedding 余弦距离 < eps (0.2)  → DBSCAN 能聚成簇
2. 跨场景 Fact 的 Embedding 余弦距离 > eps (0.2)  → 不会跨场景污染
3. 噪声 Fact 与场景 Fact 距离 > eps              → 噪声不会混入习惯

使用 OpenAI ada-002 通过环境变量中的 API Key 调用。
"""
import os
import sys
import json
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from scenarios.mock_data_generator import generate_full_dataset
from engine.signal_to_fact import scene_signals_to_facts, noise_signals_to_facts


# ──────────────────────────────────────────────
# Embedding 工具
# ──────────────────────────────────────────────
def _fix_socks_proxy():
    """httpx 要求 socks5:// 而非 socks://，修正环境变量"""
    import os
    for var in ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY",
                "all_proxy", "https_proxy", "http_proxy"):
        val = os.environ.get(var, "")
        if val.startswith("socks://"):
            os.environ[var] = val.replace("socks://", "socks5://", 1)


def get_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> np.ndarray:
    """调用 OpenAI ada-002 获取 Embedding 向量"""
    _fix_socks_proxy()
    client = OpenAI()  # 从环境变量读取 OPENAI_API_KEY / OPENAI_API_BASE
    response = client.embeddings.create(input=texts, model=model)
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """余弦距离 = 1 - 余弦相似度"""
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - sim


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """计算 N×N 余弦距离矩阵"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    sim_matrix = normalized @ normalized.T
    return 1.0 - sim_matrix


# ──────────────────────────────────────────────
# Step 1: 生成 Mock 数据 → Fact 文本
# ──────────────────────────────────────────────
def generate_all_facts() -> Dict[str, List[Dict]]:
    """生成全部 Fact 并按场景分组"""
    dataset = generate_full_dataset()

    all_facts = {
        "morning_commute": [],
        "arriving_home": [],
        "toll_parking_entry": [],
        "noise": [],
    }

    for scene_type in ["morning_commute", "arriving_home", "toll_parking_entry"]:
        for event in dataset["scenes"][scene_type]:
            facts = scene_signals_to_facts(scene_type, event)
            for f in facts:
                f["_day"] = event.get("day")
            all_facts[scene_type].extend(facts)

    for event in dataset["scenes"]["noise"]:
        facts = noise_signals_to_facts(event)
        all_facts["noise"].extend(facts)

    return all_facts


# ──────────────────────────────────────────────
# Step 2: 验证逻辑
# ──────────────────────────────────────────────
def run_validation():
    print("=" * 70)
    print("  Embedding 关键验证 — Signal → Fact → ada-002 → 余弦距离")
    print("=" * 70)

    EPS = 0.2  # DBSCAN 目标 eps
    all_facts = generate_all_facts()

    # 统计
    for scene, facts in all_facts.items():
        print(f"\n  {scene}: {len(facts)} facts")
        for f in facts[:3]:
            print(f"    → \"{f['text'][:80]}...\"" if len(f['text']) > 80 else f"    → \"{f['text']}\"")
        if len(facts) > 3:
            print(f"    ... ({len(facts) - 3} more)")

    # ──── 收集所有 Fact 文本 ────
    labeled_facts = []
    for scene, facts in all_facts.items():
        for f in facts:
            labeled_facts.append({
                "text": f["text"],
                "scene": scene,
                "type": f.get("type", "PREF"),
                "signal": f.get("metadata", {}).get("signal", ""),
                "day": f.get("_day", "?"),
            })

    texts = [f["text"] for f in labeled_facts]
    print(f"\n  Total Fact texts to embed: {len(texts)}")

    # ──── 调用 ada-002 ────
    print("\n  Calling ada-002 for embeddings...")
    embeddings = get_embeddings(texts)
    print(f"  Got {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    # ──── 计算距离矩阵 ────
    dist_matrix = cosine_distance_matrix(embeddings)

    # ──── 验证 1: 同场景内距离 ────
    print("\n" + "=" * 70)
    print("  验证 1: 同场景 Fact 余弦距离 (应 < eps={:.2f})".format(EPS))
    print("=" * 70)

    for scene in ["morning_commute", "arriving_home", "toll_parking_entry"]:
        indices = [i for i, f in enumerate(labeled_facts) if f["scene"] == scene]
        if len(indices) < 2:
            continue

        # 按 signal 类型分组，同类 Fact 应该更近
        by_signal = {}
        for idx in indices:
            sig = labeled_facts[idx]["signal"]
            by_signal.setdefault(sig, []).append(idx)

        print(f"\n  [{scene}] — {len(indices)} facts, {len(by_signal)} signal types")

        for sig, sig_indices in sorted(by_signal.items()):
            if len(sig_indices) < 2:
                continue
            dists = []
            for i, j in combinations(sig_indices, 2):
                dists.append(dist_matrix[i][j])
            avg_d = np.mean(dists)
            max_d = np.max(dists)
            min_d = np.min(dists)
            status = "✓ PASS" if max_d < EPS else "✗ FAIL"

            # 展示文本样例
            sample1 = labeled_facts[sig_indices[0]]["text"][:60]
            sample2 = labeled_facts[sig_indices[1]]["text"][:60]

            print(f"    {sig:<25} n={len(sig_indices):>2}  "
                  f"avg={avg_d:.4f}  max={max_d:.4f}  min={min_d:.4f}  {status}")
            print(f"      e.g. \"{sample1}\"")
            print(f"      vs.  \"{sample2}\"")

    # ──── 验证 2: 跨场景距离 ────
    print("\n" + "=" * 70)
    print("  验证 2: 跨场景 Fact 余弦距离 (应 > eps={:.2f})".format(EPS))
    print("=" * 70)

    scenes = ["morning_commute", "arriving_home", "toll_parking_entry"]
    for s1, s2 in combinations(scenes, 2):
        idx1 = [i for i, f in enumerate(labeled_facts) if f["scene"] == s1]
        idx2 = [i for i, f in enumerate(labeled_facts) if f["scene"] == s2]

        cross_dists = []
        for i in idx1:
            for j in idx2:
                cross_dists.append(dist_matrix[i][j])

        avg_d = np.mean(cross_dists)
        min_d = np.min(cross_dists)
        max_d = np.max(cross_dists)
        status = "✓ PASS" if min_d > EPS else "⚠ WARN" if avg_d > EPS else "✗ FAIL"

        print(f"\n  {s1:<25} ↔ {s2:<25}")
        print(f"    avg={avg_d:.4f}  min={min_d:.4f}  max={max_d:.4f}  {status}")

        # 找最近的一对
        min_i, min_j = None, None
        min_val = float('inf')
        for i in idx1:
            for j in idx2:
                if dist_matrix[i][j] < min_val:
                    min_val = dist_matrix[i][j]
                    min_i, min_j = i, j
        if min_i is not None:
            print(f"    closest pair (d={min_val:.4f}):")
            print(f"      [{s1}] \"{labeled_facts[min_i]['text'][:70]}\"")
            print(f"      [{s2}] \"{labeled_facts[min_j]['text'][:70]}\"")

    # ──── 验证 3: 噪声与场景的距离 ────
    print("\n" + "=" * 70)
    print("  验证 3: 噪声 Fact 与场景 Fact 距离 (应 > eps={:.2f})".format(EPS))
    print("=" * 70)

    noise_indices = [i for i, f in enumerate(labeled_facts) if f["scene"] == "noise"]

    for scene in scenes:
        scene_indices = [i for i, f in enumerate(labeled_facts) if f["scene"] == scene]
        cross_dists = []
        for i in noise_indices:
            for j in scene_indices:
                cross_dists.append(dist_matrix[i][j])

        if not cross_dists:
            continue

        avg_d = np.mean(cross_dists)
        min_d = np.min(cross_dists)
        status = "✓ PASS" if min_d > EPS else "⚠ WARN"

        print(f"\n  noise ↔ {scene:<25}")
        print(f"    avg={avg_d:.4f}  min={min_d:.4f}  {status}")

        # 最危险的一对
        min_i, min_j = None, None
        min_val = float('inf')
        for i in noise_indices:
            for j in scene_indices:
                if dist_matrix[i][j] < min_val:
                    min_val = dist_matrix[i][j]
                    min_i, min_j = i, j
        if min_i is not None:
            print(f"    closest pair (d={min_val:.4f}):")
            print(f"      [noise] \"{labeled_facts[min_i]['text'][:70]}\"")
            print(f"      [{scene}] \"{labeled_facts[min_j]['text'][:70]}\"")

    # ──── 验证 4: DBSCAN 模拟 ────
    print("\n" + "=" * 70)
    print("  验证 4: DBSCAN 聚类模拟 (eps={:.2f}, min_samples=5)".format(EPS))
    print("=" * 70)

    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=EPS, min_samples=3, metric='cosine')
    labels = clustering.fit_predict(embeddings)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"\n  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")

    # 展示每个聚类的组成
    for cluster_id in sorted(set(labels)):
        members = [labeled_facts[i] for i in range(len(labels)) if labels[i] == cluster_id]
        scene_counts = {}
        for m in members:
            scene_counts[m["scene"]] = scene_counts.get(m["scene"], 0) + 1

        label_str = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise (-1)"
        purity = max(scene_counts.values()) / len(members) if members else 0
        purity_status = "✓ Pure" if purity >= 0.9 else "⚠ Mixed"

        print(f"\n  {label_str}: {len(members)} members — {purity_status}")
        for scene, cnt in sorted(scene_counts.items(), key=lambda x: -x[1]):
            print(f"    {scene}: {cnt}")
        # 样例
        for m in members[:2]:
            print(f"    → \"{m['text'][:70]}\"")

    # ──── 总结 ────
    print("\n" + "=" * 70)
    print("  验证总结")
    print("=" * 70)
    print(f"  eps = {EPS}")
    print(f"  Embedding dim = {embeddings.shape[1]}")
    print(f"  Total facts = {len(labeled_facts)}")
    print(f"  DBSCAN clusters = {n_clusters}, noise = {n_noise}")
    print()

    # 保存结果
    result = {
        "eps": EPS,
        "embedding_dim": int(embeddings.shape[1]),
        "total_facts": len(labeled_facts),
        "dbscan_clusters": n_clusters,
        "dbscan_noise": int(n_noise),
        "facts": [
            {
                "text": f["text"],
                "scene": f["scene"],
                "signal": f["signal"],
                "cluster": int(labels[i]),
            }
            for i, f in enumerate(labeled_facts)
        ],
    }

    out_path = os.path.join(os.path.dirname(__file__), "embedding_validation_result.json")
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
    print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    run_validation()
