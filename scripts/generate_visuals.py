"""
生成汇报可视化材料 — 证明 ChromaDB + ada-002 + DBSCAN 管线可用

输出 3 张图到 output/ 目录:
  1. fig1_tsne_clusters.png  — t-SNE 散点图（按 DBSCAN 聚类着色）
  2. fig2_distance_heatmap.png — Hybrid 距离矩阵热力图
  3. fig3_cluster_summary.png — 聚类汇总（成员数 + 置信度）

用法:
    conda run -n rwkv-fla python scripts/generate_visuals.py
"""
import os
import sys
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.signal_to_fact import signals_to_facts
from panoramix_core.store.fact_store_chroma import FactStoreChroma
from panoramix_core.clustering.habits_detector import HabitsDetector
from panoramix_core.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, HYBRID_ALPHA
from engine.cluster_confidence import compute_all_cluster_confidences

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 场景标签 → 显示名 + 颜色 ──
SCENE_COLORS = {
    "morning_commute": "#2196F3",     # 蓝
    "arriving_home": "#4CAF50",       # 绿
    "toll_parking_entry": "#FF9800",  # 橙
    "noise": "#9E9E9E",              # 灰
}
SCENE_NAMES_CN = {
    "morning_commute": "Morning Commute",
    "arriving_home": "Arriving Home",
    "toll_parking_entry": "Toll/Parking",
    "noise": "Noise/Random",
}

# ── DBSCAN 聚类颜色 (用 tab20) ──
CLUSTER_CMAP = plt.cm.tab20


def load_mockup_data():
    """加载 mockup 数据 → facts"""
    data_file = os.path.join(PROJECT_ROOT, "tests", "step1_mockup_data.json")
    with open(data_file, encoding="utf-8") as f:
        data = json.load(f)

    all_facts = []
    scene_labels = []  # 保留 ground truth scene 用于着色

    for scene_name, events in data["scenes"].items():
        for event in events:
            facts = signals_to_facts(event)
            all_facts.extend(facts)
            scene_labels.extend([scene_name] * len(facts))

    return all_facts, scene_labels


def run_pipeline(facts):
    """运行完整管线: ChromaDB存储 → ada-002 embedding → 取回"""
    username = "visual-demo"
    store = FactStoreChroma(username)

    # 清空旧数据
    try:
        old = store.get_facts(username)
        if old:
            store.delete_facts(username, [f.id for f in old])
    except Exception:
        pass

    # 存入 ChromaDB (触发 ada-002 embedding)
    log.info(f"Storing {len(facts)} facts to ChromaDB (ada-002 embedding)...")
    store.store_facts(username, facts)

    # 取回 facts + embeddings
    items = store.get_facts_with_embeddings(username)
    log.info(f"Retrieved {len(items)} facts with embeddings")

    # 清理
    store.delete_facts(username, [it["fact"].id for it in items])
    store.close()

    return items


def compute_hybrid_distance(items):
    """计算 hybrid 距离矩阵"""
    detector = HabitsDetector(llm_client=None)
    return detector._compute_hybrid_distance_matrix(items)


def match_scene_labels(items, original_facts, scene_labels):
    """将 ChromaDB 返回的 items 匹配回原始 scene labels"""
    # 基于 text + timestamp 匹配
    fact_to_scene = {}
    for f, s in zip(original_facts, scene_labels):
        key = (f.text, str(f.time_stamp))
        fact_to_scene[key] = s

    matched = []
    for item in items:
        f = item["fact"]
        key = (f.text, str(f.time_stamp))
        matched.append(fact_to_scene.get(key, "noise"))
    return matched


# ═══════════════════════════════════════════════════
# 图 1: t-SNE 聚类散点图
# ═══════════════════════════════════════════════════

def plot_tsne(embeddings, labels, scene_labels_matched, output_path):
    """t-SNE 2D 散点图，按 DBSCAN 聚类着色，形状区分场景"""
    n = len(embeddings)
    tsne = TSNE(n_components=2, perplexity=min(15, n - 1),
                random_state=42, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l >= 0])

    # 场景 → marker
    scene_markers = {
        "morning_commute": "o",
        "arriving_home": "s",
        "toll_parking_entry": "^",
        "noise": "x",
    }

    for i in range(n):
        cid = labels[i]
        scene = scene_labels_matched[i]
        marker = scene_markers.get(scene, "o")

        if cid == -1:
            color = "#CCCCCC"
            alpha = 0.4
            size = 30
        else:
            color = CLUSTER_CMAP(cid % 20)
            alpha = 0.85
            size = 60

        ax.scatter(coords[i, 0], coords[i, 1],
                   c=[color], marker=marker, s=size, alpha=alpha,
                   edgecolors="white", linewidths=0.5)

    # 图例 — 场景形状
    scene_handles = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="#555",
                   markersize=8, label=SCENE_NAMES_CN[s])
        for s, m in scene_markers.items()
    ]
    legend1 = ax.legend(handles=scene_handles, loc="upper left",
                        title="Ground Truth Scene", fontsize=9, title_fontsize=10)
    ax.add_artist(legend1)

    # 图例 — 聚类颜色 (只显示 top 聚类)
    cluster_handles = []
    for cid in sorted(set(labels)):
        if cid == -1:
            cluster_handles.append(Patch(facecolor="#CCCCCC", label="noise (-1)"))
        elif cid <= 14:
            cluster_handles.append(Patch(facecolor=CLUSTER_CMAP(cid % 20),
                                         label=f"cluster {cid}"))
    ax.legend(handles=cluster_handles, loc="upper right",
              title="DBSCAN Cluster", fontsize=7, title_fontsize=9,
              ncol=2)
    ax.add_artist(legend1)

    ax.set_title(f"Hybrid DBSCAN Clustering (t-SNE)\n"
                 f"83 facts → {n_clusters} clusters | "
                 f"α={HYBRID_ALPHA}, ε={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# 图 2: 距离矩阵热力图
# ═══════════════════════════════════════════════════

def plot_heatmap(dist_matrix, labels, scene_labels_matched, output_path):
    """距离矩阵热力图，按聚类排序"""
    # 按 cluster → scene 排序
    n = len(labels)
    order = sorted(range(n), key=lambda i: (labels[i] if labels[i] >= 0 else 999,
                                             scene_labels_matched[i]))
    sorted_dist = dist_matrix[np.ix_(order, order)]
    sorted_labels = [labels[i] for i in order]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    im = ax.imshow(sorted_dist, cmap="YlOrRd_r", aspect="auto",
                   vmin=0, vmax=min(0.3, np.max(sorted_dist)))

    # 聚类边界线
    prev = sorted_labels[0]
    for i, l in enumerate(sorted_labels):
        if l != prev:
            ax.axhline(y=i - 0.5, color="blue", linewidth=0.5, alpha=0.5)
            ax.axvline(x=i - 0.5, color="blue", linewidth=0.5, alpha=0.5)
            prev = l

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Hybrid Distance (α·text_cosine + (1-α)·context)", fontsize=10)

    ax.set_title(f"Hybrid Distance Matrix (sorted by cluster)\n"
                 f"α={HYBRID_ALPHA} | Blue lines = cluster boundaries | "
                 f"ε={DBSCAN_EPS} threshold",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Fact index (sorted)")
    ax.set_ylabel("Fact index (sorted)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# 图 3: 聚类汇总
# ═══════════════════════════════════════════════════

def plot_cluster_summary(dist_matrix, labels, core_indices, scene_labels_matched,
                         output_path):
    """聚类汇总: 成员数 + 置信度 + 场景分布"""
    confidences = compute_all_cluster_confidences(
        dist_matrix, labels, core_indices, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
    )

    cluster_ids = [cc.cluster_id for cc in confidences]
    sizes = [cc.cluster_size for cc in confidences]
    confs = [cc.confidence for cc in confidences]
    cohesions = [cc.cohesion for cc in confidences]

    # 每个聚类的主要场景
    cluster_scenes = []
    for cid in cluster_ids:
        members = [scene_labels_matched[i] for i in range(len(labels)) if labels[i] == cid]
        from collections import Counter
        top = Counter(members).most_common(1)[0][0]
        cluster_scenes.append(top)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[1, 1])

    # 上图: 成员数柱状图
    bar_colors = [SCENE_COLORS.get(s, "#999") for s in cluster_scenes]
    x = np.arange(len(cluster_ids))
    bars = ax1.bar(x, sizes, color=bar_colors, edgecolor="white", linewidth=0.5)

    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(size), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax1.set_ylabel("Cluster Size", fontsize=11)
    ax1.set_title(f"DBSCAN Clustering Results — {len(cluster_ids)} clusters from 83 facts\n"
                  f"(colored by ground truth scene)",
                  fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=8)

    # 场景图例
    handles = [Patch(facecolor=c, label=SCENE_NAMES_CN[s])
               for s, c in SCENE_COLORS.items()]
    ax1.legend(handles=handles, loc="upper right", fontsize=9)

    noise_count = sum(1 for l in labels if l == -1)
    ax1.text(0.01, 0.95, f"Noise: {noise_count} facts",
             transform=ax1.transAxes, fontsize=9, color="#666",
             verticalalignment="top")

    # 下图: 置信度三维度堆叠柱状图
    w = 0.6
    bottom1 = np.zeros(len(cluster_ids))
    bottom2 = np.array([cc.cohesion * 0.50 for cc in confidences])
    bottom3 = bottom2 + np.array([cc.core_ratio * 0.25 for cc in confidences])

    ax2.bar(x, [cc.cohesion * 0.50 for cc in confidences], w,
            label=f"cohesion (×{0.50})", color="#42A5F5")
    ax2.bar(x, [cc.core_ratio * 0.25 for cc in confidences], w,
            bottom=bottom2, label=f"core_ratio (×{0.25})", color="#66BB6A")
    ax2.bar(x, [cc.size_factor * 0.25 for cc in confidences], w,
            bottom=bottom3, label=f"size_factor (×{0.25})", color="#FFA726")

    # 总分标注
    for i, conf in enumerate(confs):
        ax2.text(x[i], conf + 0.01, f"{conf:.2f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax2.set_ylabel("Clustering Confidence", fontsize=11)
    ax2.set_xlabel("Cluster ID", fontsize=11)
    ax2.set_title("Cluster Confidence Score Breakdown\n"
                  "confidence = 0.50·cohesion + 0.25·core_ratio + 0.25·size_factor",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{c}" for c in cluster_ids], fontsize=8)
    ax2.set_ylim(0, 1.08)
    ax2.axhline(y=0.7, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.text(len(cluster_ids) - 0.5, 0.71, "threshold=0.7",
             fontsize=8, color="red", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Saved: {output_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Paranomix Memory — 可视化材料生成")
    log.info("=" * 60)

    # 1. 加载数据
    log.info("\n[1/5] Loading mockup data...")
    facts, scene_labels = load_mockup_data()
    log.info(f"  {len(facts)} facts, {len(set(scene_labels))} scene types")

    # 2. 运行管线 (ChromaDB + ada-002)
    log.info("\n[2/5] Running pipeline (ChromaDB + ada-002 embedding)...")
    items = run_pipeline(facts)

    # 匹配场景标签
    scene_labels_matched = match_scene_labels(items, facts, scene_labels)

    # 3. 计算 hybrid 距离矩阵
    log.info("\n[3/5] Computing hybrid distance matrix...")
    dist_matrix = compute_hybrid_distance(items)
    embeddings = np.array([it["embedding"] for it in items])

    # 4. 运行 DBSCAN
    log.info("\n[4/5] Running DBSCAN clustering...")
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES,
                        metric="precomputed")
    labels = clustering.fit_predict(dist_matrix)
    core_indices = clustering.core_sample_indices_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = sum(1 for l in labels if l == -1)
    log.info(f"  {n_clusters} clusters, {noise_count} noise points")

    # 5. 生成图表
    log.info("\n[5/5] Generating visualizations...")

    plot_tsne(embeddings, labels, scene_labels_matched,
              os.path.join(OUTPUT_DIR, "fig1_tsne_clusters.png"))

    plot_heatmap(dist_matrix, labels, scene_labels_matched,
                 os.path.join(OUTPUT_DIR, "fig2_distance_heatmap.png"))

    plot_cluster_summary(dist_matrix, labels, core_indices, scene_labels_matched,
                         os.path.join(OUTPUT_DIR, "fig3_cluster_summary.png"))

    log.info("\n" + "=" * 60)
    log.info(f"Done! 3 figures saved to {OUTPUT_DIR}/")
    log.info("  fig1_tsne_clusters.png   — t-SNE 聚类散点图")
    log.info("  fig2_distance_heatmap.png — Hybrid 距离矩阵")
    log.info("  fig3_cluster_summary.png  — 聚类汇总+置信度")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
