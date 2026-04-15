"""
Cluster Visualization — t-SNE / PCA 向量空间可视化

在 Streamlit 中交互式展示:
  1. t-SNE 2D 散点图 (DBSCAN 聚类着色 + 场景标记)
  2. PCA  2D 散点图 (同上，更稳定的线性投影)
  3. Hybrid 距离矩阵热力图 (按聚类排序)
  4. 聚类置信度分解柱状图

数据来源: ChromaDB 中的 fact embeddings + Hybrid DBSCAN 聚类结果
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

log = logging.getLogger(__name__)

# ── 配色 (对齐 style.css dark theme) ──
_BG = "#0A1628"
_PANEL_BG = "#132035"
_CARD_BG = "#1A2A42"
_BORDER = "#1E3A5F"
_TEAL = "#00BFC8"
_GREEN = "#00C896"
_TEXT_PRI = "#E8EDF3"
_TEXT_SEC = "#8899AA"

# DBSCAN 聚类配色 (最多 12 个聚类 + noise)
CLUSTER_COLORS = [
    "#3B82F6",  # blue
    "#10B981",  # green
    "#F59E0B",  # amber
    "#EF4444",  # red
    "#A855F7",  # purple
    "#06B6D4",  # cyan
    "#F97316",  # orange
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#8B5CF6",  # violet
    "#84CC16",  # lime
    "#E11D48",  # rose
]
NOISE_COLOR = "#4B5563"  # gray


def render_cluster_visualization(username: str):
    """
    主入口: 从 ChromaDB 取 embeddings → DBSCAN 聚类 → 交互式图表。

    在 Tab 1 右侧或独立面板中调用。
    """
    try:
        data = _load_embedding_data(username)
    except Exception as e:
        st.caption("No embedding data available. Run 'Load Mockup Data' or 'Analysis to model' first.")
        log.debug(f"Cluster viz load: {e}")
        return

    if data is None:
        st.caption("No embedding data available. Run 'Load Mockup Data' or 'Analysis to model' first.")
        return

    items = data["items"]
    n = len(items)

    if n < 3:
        st.caption(f"Need at least 3 facts for visualization (currently {n}).")
        return

    # ── 计算聚类 (使用缓存避免重复计算) ──
    cache_key = f"viz_cache_{username}_{n}"
    if cache_key not in st.session_state:
        with st.spinner("Computing embeddings projection..."):
            st.session_state[cache_key] = _compute_viz_data(items)

    viz = st.session_state[cache_key]

    # ── 聚类摘要指标 ──
    n_clusters = viz["n_clusters"]
    noise_count = viz["noise_count"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Total Records", n)
    with mc2:
        st.metric("Habits Found", n_clusters)
    with mc3:
        st.metric("Unclustered", noise_count)
    with mc4:
        st.metric("Clustered", f"{n - noise_count}/{n}")

    st.caption(
        "This visualization shows how the system groups driving behaviors into habit patterns. "
        "Each record is embedded as a vector; similar records cluster together automatically."
    )

    # ── 图表选择 ──
    chart_tab1, chart_tab2, chart_tab3 = st.tabs([
        "t-SNE Scatter",
        "PCA Scatter",
        "Confidence Breakdown",
    ])

    with chart_tab1:
        _render_tsne_chart(viz)

    with chart_tab2:
        _render_pca_chart(viz)

    with chart_tab3:
        _render_confidence_chart(viz)


# ═══════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════

def _load_embedding_data(username: str) -> Optional[Dict]:
    """从 session state 取 pipeline 缓存的 items，或从 ChromaDB 取 facts + embeddings"""
    # 优先使用 pipeline 运行时捕获的 items（Chroma 数据在 pipeline 后被删除）
    cached_items = st.session_state.get(f"t1_viz_items_{username}")
    if cached_items:
        return {"items": cached_items}

    # Fallback: try Chroma (may be empty after pipeline)
    from panoramix_core.store.fact_store_chroma import FactStoreChroma
    store = FactStoreChroma(username)
    items = store.get_facts_with_embeddings(username)
    store.close()

    if not items:
        return None

    return {"items": items}


# ═══════════════════════════════════════════════════
# 聚类计算 + 降维
# ═══════════════════════════════════════════════════

def _derive_cluster_label(cluster_facts) -> str:
    """
    从聚类成员中推导出可读标签。
    优先使用 HABIT fact 的 scene_name，否则从文本中提取关键词。
    """
    import json as _json

    # 1) 优先: HABIT fact 的 scene_name
    for f in cluster_facts:
        if f.type.value == "HABIT" and f.json_metadata:
            try:
                meta = _json.loads(f.json_metadata) if isinstance(f.json_metadata, str) else f.json_metadata
                sn = meta.get("scene_name")
                if sn:
                    return sn
            except Exception:
                pass

    # 2) 从文本中提取主要动作关键词
    _KEYWORDS = [
        ("temperature", "AC / Climate"),
        ("air conditioning", "AC / Climate"),
        ("seat heat", "Seat Heating"),
        ("eco mode", "Eco Driving"),
        ("sport mode", "Sport Driving"),
        ("comfort mode", "Comfort Driving"),
        ("navigation", "Navigation"),
        ("destination", "Navigation"),
        ("media", "Media / Audio"),
        ("podcast", "Media / Audio"),
        ("radio", "Media / Audio"),
        ("music", "Media / Audio"),
        ("volume", "Media / Audio"),
        ("window", "Window Control"),
        ("acc", "ACC / Cruise"),
        ("follow distance", "ACC / Cruise"),
        ("keyless", "Keyless Entry"),
        ("engine", "Engine Control"),
        ("drive mode", "Drive Mode"),
    ]
    texts = " ".join(f.text.lower() for f in cluster_facts)
    for kw, label in _KEYWORDS:
        if kw in texts:
            return label

    # 3) 兜底: 用上下文的 time_bucket
    buckets = [f.context.time_bucket for f in cluster_facts
               if f.context and f.context.time_bucket != "unknown"]
    if buckets:
        from collections import Counter
        most_common = Counter(buckets).most_common(1)[0][0]
        return most_common.replace("_", " ").title() + " Pattern"

    return "Behavior Pattern"


def _compute_viz_data(items: List[Dict]) -> Dict:
    """一次性计算: hybrid距离 → DBSCAN → t-SNE → PCA → 置信度"""
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from panoramix_core.config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES, HYBRID_ALPHA
    from panoramix_core.clustering.habits_detector import HabitsDetector
    from engine.cluster_confidence import compute_all_cluster_confidences

    n = len(items)
    embeddings = np.array([it["embedding"] for it in items])
    facts = [it["fact"] for it in items]

    # ── Hybrid 距离矩阵 ──
    detector = HabitsDetector(llm_client=None)
    dist_matrix = detector._compute_hybrid_distance_matrix(items)

    # ── DBSCAN ──
    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric="precomputed",
    )
    labels = clustering.fit_predict(dist_matrix)
    core_indices = clustering.core_sample_indices_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))

    # ── t-SNE 降维 (perplexity 必须 < n) ──
    perp = max(2, min(15, n - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                init="pca", learning_rate="auto")
    tsne_coords = tsne.fit_transform(embeddings)

    # ── PCA 降维 ──
    n_components = min(2, n, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(embeddings)
    pca_var = pca.explained_variance_ratio_.tolist()
    # 确保 pca_coords 是 (n, 2)
    if pca_coords.shape[1] < 2:
        pca_coords = np.column_stack([pca_coords, np.zeros(n)])
        pca_var.append(0.0)

    # ── 聚类置信度 ──
    confidences = compute_all_cluster_confidences(
        dist_matrix, labels, core_indices, DBSCAN_EPS, DBSCAN_MIN_SAMPLES
    )

    # ── 统计 fact 类型分布 ──
    type_counts = {}
    for f in facts:
        t = f.type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    # ── 为每个聚类推导可读标签 ──
    cluster_labels_map = {}
    for cid in sorted(set(labels)):
        cid = int(cid)
        if cid == -1:
            cluster_labels_map[-1] = "Noise (unclustered)"
        else:
            member_facts = [facts[i] for i in range(n) if int(labels[i]) == cid]
            cluster_labels_map[cid] = _derive_cluster_label(member_facts)

    # ── 构建点信息 ──
    core_set = set(core_indices.tolist())
    points = []
    for i in range(n):
        f = facts[i]
        cid = int(labels[i])
        is_core = i in core_set

        # 上下文摘要
        ctx = f.context
        ctx_str = ", ".join(filter(None, [
            ctx.time_bucket.replace("_", " ").title() if ctx.time_bucket != "unknown" else None,
            ctx.vehicle_state.replace("_", " ").title() if ctx.vehicle_state != "unknown" else None,
            ctx.geofence.title() if ctx.geofence else None,
            "Weekday" if ctx.weekday is True else ("Weekend" if ctx.weekday is False else None),
        ])) or "—"

        # 文本截断
        text_short = f.text[:60] + ("..." if len(f.text) > 60 else "")

        points.append({
            "tsne_x": float(tsne_coords[i, 0]),
            "tsne_y": float(tsne_coords[i, 1]),
            "pca_x": float(pca_coords[i, 0]),
            "pca_y": float(pca_coords[i, 1]),
            "cluster": cid,
            "is_core": is_core,
            "text": text_short,
            "full_text": f.text,
            "context": ctx_str,
            "fact_type": f.type.value,
        })

    # ── 聚类信息 ──
    cluster_info = {}
    for cc in confidences:
        cid = cc.cluster_id
        cluster_info[cid] = {
            "confidence": cc.confidence,
            "cohesion": cc.cohesion,
            "core_ratio": cc.core_ratio,
            "size_factor": cc.size_factor,
            "size": cc.cluster_size,
            "mean_dist": cc.mean_intra_dist,
            "label": cluster_labels_map.get(cid, f"Cluster {cid}"),
        }

    return {
        "points": points,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "labels": labels.tolist(),
        "cluster_info": cluster_info,
        "cluster_labels": cluster_labels_map,
        "confidences": confidences,
        "pca_variance": pca_var,
        "type_counts": type_counts,
        "params": {
            "alpha": HYBRID_ALPHA,
            "eps": DBSCAN_EPS,
            "min_samples": DBSCAN_MIN_SAMPLES,
        },
    }


# ═══════════════════════════════════════════════════
# t-SNE 散点图 (Plotly)
# ═══════════════════════════════════════════════════

def _render_tsne_chart(viz: Dict):
    """Plotly 交互式 t-SNE 散点图"""
    import plotly.graph_objects as go

    points = viz["points"]
    params = viz["params"]
    cluster_labels = viz.get("cluster_labels", {})

    fig = go.Figure()

    # 按 cluster 分组绘制
    clusters_seen = sorted(set(p["cluster"] for p in points))

    for cid in clusters_seen:
        cluster_points = [p for p in points if p["cluster"] == cid]
        xs = [p["tsne_x"] for p in cluster_points]
        ys = [p["tsne_y"] for p in cluster_points]

        symbols = [
            "diamond" if p["fact_type"] == "HABIT" else ("x" if cid == -1 else "circle")
            for p in cluster_points
        ]

        if cid == -1:
            color = NOISE_COLOR
            label = cluster_labels.get(-1, "Noise")
            name = f"{label} ({len(cluster_points)} pts)"
            size = 6
            opacity = 0.5
        else:
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            info = viz["cluster_info"].get(cid, {})
            conf = info.get("confidence", 0)
            label = cluster_labels.get(cid, f"Cluster {cid}")
            name = f"{label} ({info.get('size', '?')} pts, conf={conf:.2f})"
            size = 9
            opacity = 0.85

        hover = [
            f"<b>{p['text']}</b><br>"
            f"Scene: {cluster_labels.get(p['cluster'], '?')}<br>"
            f"Type: {p['fact_type']}<br>"
            f"Context: {p['context']}<br>"
            f"Core: {'Yes' if p['is_core'] else 'No'}"
            for p in cluster_points
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=name,
            marker=dict(
                color=color,
                size=size,
                symbol=symbols,
                opacity=opacity,
                line=dict(width=0.5, color="white"),
            ),
            hovertext=hover,
            hoverinfo="text",
        ))

        # 在聚类质心处标注场景名称
        if cid != -1:
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            fig.add_annotation(
                x=cx, y=cy,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor="rgba(10,22,40,0.7)",
                borderpad=3,
                yshift=14,
            )

    fig.update_layout(
        title=dict(
            text=(f"t-SNE — Habit Clusters in Embedding Space<br>"
                  f"<sub>Each dot = one driving behavior record. "
                  f"Nearby dots share similar actions & context.</sub>"),
            font=dict(size=14, color=_TEXT_PRI),
        ),
        xaxis=dict(
            title="t-SNE dim 1",
            gridcolor=_BORDER,
            zerolinecolor=_BORDER,
            color=_TEXT_SEC,
        ),
        yaxis=dict(
            title="t-SNE dim 2",
            gridcolor=_BORDER,
            zerolinecolor=_BORDER,
            color=_TEXT_SEC,
        ),
        plot_bgcolor=_CARD_BG,
        paper_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT_PRI, size=11),
        legend=dict(
            font=dict(size=10, color=_TEXT_SEC),
            bgcolor="rgba(19,32,53,0.8)",
            bordercolor=_BORDER,
            borderwidth=1,
        ),
        height=480,
        margin=dict(l=50, r=20, t=70, b=40),
    )

    st.plotly_chart(fig, use_container_width=True, key="tsne_chart")
    st.caption(
        "Each point represents a driving behavior fact. "
        "Points that form a tight group (cluster) indicate a **repeated habit pattern**. "
        "Scattered gray points are one-off behaviors that haven't formed a habit yet. "
        "Hover over any point to see its details."
    )


# ═══════════════════════════════════════════════════
# PCA 散点图 (Plotly)
# ═══════════════════════════════════════════════════

def _render_pca_chart(viz: Dict):
    """Plotly 交互式 PCA 散点图"""
    import plotly.graph_objects as go

    points = viz["points"]
    pca_var = viz.get("pca_variance", [0, 0])
    cluster_labels = viz.get("cluster_labels", {})

    fig = go.Figure()

    clusters_seen = sorted(set(p["cluster"] for p in points))

    for cid in clusters_seen:
        cluster_points = [p for p in points if p["cluster"] == cid]
        xs = [p["pca_x"] for p in cluster_points]
        ys = [p["pca_y"] for p in cluster_points]

        symbols = [
            "diamond" if p["fact_type"] == "HABIT" else ("x" if cid == -1 else "circle")
            for p in cluster_points
        ]

        if cid == -1:
            color = NOISE_COLOR
            label = cluster_labels.get(-1, "Noise")
            name = f"{label} ({len(cluster_points)} pts)"
            size = 6
            opacity = 0.5
        else:
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            info = viz["cluster_info"].get(cid, {})
            conf = info.get("confidence", 0)
            label = cluster_labels.get(cid, f"Cluster {cid}")
            name = f"{label} ({info.get('size', '?')} pts, conf={conf:.2f})"
            size = 9
            opacity = 0.85

        hover = [
            f"<b>{p['text']}</b><br>"
            f"Scene: {cluster_labels.get(p['cluster'], '?')}<br>"
            f"Type: {p['fact_type']}<br>"
            f"Context: {p['context']}"
            for p in cluster_points
        ]

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            name=name,
            marker=dict(
                color=color,
                size=size,
                symbol=symbols,
                opacity=opacity,
                line=dict(width=0.5, color="white"),
            ),
            hovertext=hover,
            hoverinfo="text",
        ))

        # 聚类质心标注
        if cid != -1:
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            fig.add_annotation(
                x=cx, y=cy,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(size=10, color=color),
                bgcolor="rgba(10,22,40,0.7)",
                borderpad=3,
                yshift=14,
            )

    var1 = pca_var[0] * 100 if len(pca_var) > 0 else 0
    var2 = pca_var[1] * 100 if len(pca_var) > 1 else 0

    fig.update_layout(
        title=dict(
            text=(f"PCA — Habit Clusters (alternative view)<br>"
                  f"<sub>Linear projection preserving {var1:.1f}% + {var2:.1f}% = "
                  f"{var1+var2:.1f}% of variance</sub>"),
            font=dict(size=14, color=_TEXT_PRI),
        ),
        xaxis=dict(
            title=f"PC1 ({var1:.1f}%)",
            gridcolor=_BORDER,
            zerolinecolor=_BORDER,
            color=_TEXT_SEC,
        ),
        yaxis=dict(
            title=f"PC2 ({var2:.1f}%)",
            gridcolor=_BORDER,
            zerolinecolor=_BORDER,
            color=_TEXT_SEC,
        ),
        plot_bgcolor=_CARD_BG,
        paper_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT_PRI, size=11),
        legend=dict(
            font=dict(size=10, color=_TEXT_SEC),
            bgcolor="rgba(19,32,53,0.8)",
            bordercolor=_BORDER,
            borderwidth=1,
        ),
        height=480,
        margin=dict(l=50, r=20, t=70, b=40),
    )

    st.plotly_chart(fig, use_container_width=True, key="pca_chart")
    st.caption(
        "PCA is an alternative projection that shows cluster separation from a different angle. "
        "Well-separated groups indicate the system can reliably distinguish between different habit patterns."
    )


# ═══════════════════════════════════════════════════
# 聚类置信度分解图 (Plotly)
# ═══════════════════════════════════════════════════

def _render_confidence_chart(viz: Dict):
    """Plotly 聚类置信度三维度分解柱状图"""
    import plotly.graph_objects as go

    confidences = viz.get("confidences", [])
    cluster_labels_map = viz.get("cluster_labels", {})
    if not confidences:
        st.caption("No clusters detected — confidence chart unavailable.")
        return

    # 用场景名作为 X 轴标签
    bar_labels = [
        cluster_labels_map.get(cc.cluster_id, f"Cluster {cc.cluster_id}")
        for cc in confidences
    ]
    cohesions = [cc.cohesion * 0.50 for cc in confidences]
    core_ratios = [cc.core_ratio * 0.25 for cc in confidences]
    size_factors = [cc.size_factor * 0.25 for cc in confidences]
    totals = [cc.confidence for cc in confidences]
    sizes = [cc.cluster_size for cc in confidences]

    fig = go.Figure()

    # 三维度堆叠
    fig.add_trace(go.Bar(
        name="Tightness — how similar actions are within the group",
        x=bar_labels,
        y=cohesions,
        marker_color="#3B82F6",
        hovertemplate="Tightness (cohesion): %{customdata[0]:.3f}<br>Weighted: %{y:.3f}<extra></extra>",
        customdata=[[cc.cohesion] for cc in confidences],
    ))
    fig.add_trace(go.Bar(
        name="Stability — how reliably the pattern repeats",
        x=bar_labels,
        y=core_ratios,
        marker_color="#10B981",
        hovertemplate="Stability (core ratio): %{customdata[0]:.3f}<br>Weighted: %{y:.3f}<extra></extra>",
        customdata=[[cc.core_ratio] for cc in confidences],
    ))
    fig.add_trace(go.Bar(
        name="Evidence — how many observations support it",
        x=bar_labels,
        y=size_factors,
        marker_color="#F59E0B",
        hovertemplate="Evidence (size factor): %{customdata[0]:.3f}<br>Weighted: %{y:.3f}<extra></extra>",
        customdata=[[cc.size_factor] for cc in confidences],
    ))

    # 总分标注
    fig.add_trace(go.Scatter(
        x=bar_labels,
        y=[t + 0.03 for t in totals],
        mode="text",
        text=[f"{t:.2f}" for t in totals],
        textfont=dict(size=11, color=_TEXT_PRI),
        showlegend=False,
        hoverinfo="skip",
    ))

    # 阈值线
    fig.add_hline(
        y=0.7, line_dash="dash", line_color="#EF4444",
        annotation_text="High confidence threshold",
        annotation_position="top right",
        annotation_font_color="#EF4444",
        annotation_font_size=10,
    )

    # 底部数据量标注
    for i, (label, sz) in enumerate(zip(bar_labels, sizes)):
        fig.add_annotation(
            x=label, y=-0.06,
            text=f"{sz} records",
            showarrow=False,
            font=dict(size=9, color=_TEXT_SEC),
        )

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Habit Confidence Breakdown — How strong is each detected habit?",
            font=dict(size=14, color=_TEXT_PRI),
        ),
        xaxis=dict(
            title="Detected Habit Scene",
            color=_TEXT_SEC,
            gridcolor=_BORDER,
        ),
        yaxis=dict(
            title="Confidence Score (0–1)",
            range=[0, 1.1],
            color=_TEXT_SEC,
            gridcolor=_BORDER,
            zerolinecolor=_BORDER,
        ),
        plot_bgcolor=_CARD_BG,
        paper_bgcolor=_PANEL_BG,
        font=dict(color=_TEXT_PRI, size=11),
        legend=dict(
            font=dict(size=10, color=_TEXT_SEC),
            bgcolor="rgba(19,32,53,0.8)",
            bordercolor=_BORDER,
            borderwidth=1,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        height=420,
        margin=dict(l=50, r=20, t=70, b=60),
    )

    st.plotly_chart(fig, use_container_width=True, key="confidence_chart")
    st.caption(
        "Each bar shows how confident the system is that a habit exists. "
        "**Tightness** = how consistent the actions are; "
        "**Stability** = how reliably the pattern repeats; "
        "**Evidence** = how many observations support it. "
        "Habits above the red line are high-confidence patterns."
    )
