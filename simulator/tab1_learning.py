"""
Tab 1: User Profile Generator — 学习阶段 (完整实现)

功能:
  1. 用户选择器 (Mary / Tom / Alice / David / Lena)
  2. 两种数据输入模式:
     a) 手动模式: 上下文面板(6字段) + 行为表格(5大类) + 单次 Analysis
     b) 批量模式: "Load Mockup Data" 一键导入 5 天模拟数据
  3. "Analysis to model" → 运行 signal→fact→ChromaDB→DBSCAN pipeline
  4. Knowledge Graph 面板 — 实时刷新、按置信度排序、新行高亮
  5. 学习进度条 (cycle / 目标)
  6. Cycle 计数器
  7. t-SNE / PCA 聚类可视化 — 展示向量空间中聚类形成过程

Session state 按 username 隔离:
  - t1_cycles: dict[username → count]
  - t1_new_habit_ids_{username}: set  (本轮新 habit IDs，KG 高亮用)
  - t1_last_result_{username}: dict   (最近一次 pipeline 结果)
  - viz_cache_{username}_{n}: dict    (聚类可视化缓存)
"""
import streamlit as st
import json
import logging
from datetime import datetime

from simulator.components.context_panel import render_context_panel
from simulator.components.behavior_table import render_behavior_table, signals_summary
from simulator.components.knowledge_graph import render_knowledge_graph
from simulator.components.cluster_viz import render_cluster_visualization

log = logging.getLogger(__name__)

# 可选用户列表 (对齐 mockup 数据)
USERS = ["Mary", "Tom", "Alice", "David", "Lena"]
TARGET_CYCLES = 20  # 目标学习轮次


def _init_session():
    """初始化 session state"""
    if "t1_cycles" not in st.session_state:
        st.session_state["t1_cycles"] = {}


def render():
    """渲染 Tab 1 完整页面"""
    _init_session()

    # ── 顶栏: 用户选择器 + Cycle 徽章 + 批量/清理按钮 ──
    col_user, col_cycle, col_batch, col_clear = st.columns([2, 1.5, 1.5, 1.2])

    with col_user:
        username = st.selectbox("Driver Profile", USERS, key="t1_user")

    cycle_count = st.session_state["t1_cycles"].get(username, 0)

    with col_cycle:
        pct = min(100, int(cycle_count / TARGET_CYCLES * 100))
        st.metric("Cycle", f"{cycle_count} / {TARGET_CYCLES}")

    with col_batch:
        week_selection = st.multiselect(
            "Weeks",
            options=[1, 2, 3, 4],
            default=[1],
            key="t1_week_selection",
            help="Select which weeks of unified data to load (W1-W4).",
        )
        batch_clicked = st.button(
            "Load Unified Data",
            key="t1_batch_load",
            help="Load unified four-week data → full pipeline. "
                 "Clears the selected user's ChromaDB collection first.",
        )

    with col_clear:
        clear_clicked = st.button(
            "Clear Data",
            key="t1_clear_data",
            help="Wipe this user's ChromaDB collection (all facts + habits). "
                 "Useful to get rid of stale/duplicate state before re-training.",
        )

    # ── 学习进度条 ──
    st.html(
        f'<div style="font-family:sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;'
        f'color:#8899AA;margin-bottom:4px;">'
        f'<span>Learning progress</span><span>{pct}%</span></div>'
        f'<div style="height:5px;background:#1E3A5F;border-radius:10px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;border-radius:10px;'
        f'background:linear-gradient(90deg,#00BFC8,#00C896);"></div></div></div>'
    )

    # ── 批量加载 / 清理处理 ──
    if batch_clicked:
        _run_batch_load(username)
    if clear_clicked:
        _run_clear_data(username)

    st.divider()

    # ── 主布局: 左侧输入，右侧 KG ──
    left, right = st.columns([5, 7])

    with left:
        # 上下文面板
        ctx_data = render_context_panel(readonly=False, key_prefix="t1_ctx")
        st.markdown("")

        # 行为信号表
        selected_signals = render_behavior_table(key_prefix="t1_beh")
        st.markdown("")

        # Analysis 区域
        feat_count = len(selected_signals)
        st.caption(f"Ready to analyze · {feat_count} features selected")

        analyze_clicked = st.button(
            "Analysis to model",
            type="primary",
            use_container_width=True,
            key="t1_analyze",
            disabled=(feat_count == 0),
        )

        st.caption(
            "Each cycle updates the Knowledge Graph. "
            "Run multiple cycles to increase confidence scores."
        )

    with right:
        if analyze_clicked and selected_signals:
            _run_single_analysis(username, ctx_data, selected_signals)

        # KG 表格 + 聚类可视化 (子标签)
        kg_tab, viz_tab = st.tabs(["Knowledge Graph", "Cluster Visualization"])

        with kg_tab:
            _show_knowledge_graph(username)

        with viz_tab:
            render_cluster_visualization(username.lower())


# ═══════════════════════════════════════════════════
# Pipeline 执行
# ═══════════════════════════════════════════════════

def _run_single_analysis(username: str, ctx_data: dict, signals: dict):
    """单次分析: 手动输入 → pipeline"""
    progress_bar = st.progress(0, text="Building signal event...")

    try:
        from engine.habit_engine import HabitDemoEngine

        event = _build_event(ctx_data, signals)
        progress_bar.progress(20, text="Starting engine...")

        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        progress_bar.progress(40, text="Ingesting signals → facts...")

        result = engine.ingest_signal_batch([event])
        progress_bar.progress(80, text="Running DBSCAN clustering...")

        # 更新 cycle 计数
        _increment_cycle(username)

        # 记录新 habit IDs (按用户隔离)
        new_ids = {h["id"] for h in result.get("new_habits", [])}
        st.session_state[f"t1_new_habit_ids_{username}"] = new_ids
        st.session_state[f"t1_last_result_{username}"] = result

        progress_bar.progress(100, text="Knowledge Graph updated!")
        _show_pipeline_metrics(result)

        # 清除可视化缓存，下次渲染时重新计算
        _invalidate_viz_cache(username)

        engine.close()

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        log.exception("Single analysis pipeline error")


def _run_clear_data(username: str):
    """清空当前用户的全部 ChromaDB 数据 + 重置 session state"""
    try:
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        before = engine.get_status()
        engine.reset()
        after = engine.get_status()
        engine.close()

        # 清掉 session state 里与该用户有关的缓存
        st.session_state["t1_cycles"][username] = 0
        for k in (
            f"t1_new_habit_ids_{username}",
            f"t1_last_result_{username}",
        ):
            st.session_state.pop(k, None)
        _invalidate_viz_cache(username)

        st.success(
            f"Cleared ChromaDB for '{username}': "
            f"{before['total_facts']} facts → {after['total_facts']} facts "
            f"({before['habit_facts']} habits removed)"
        )
    except Exception as e:
        st.error(f"Clear data error: {e}")
        log.exception("Clear data error")


def _run_batch_load(username: str):
    """批量模式: 加载 unified 四周数据 → 完整 pipeline"""
    progress_bar = st.progress(0, text="Loading unified data...")

    try:
        from data.unified_loader import load_multi_week_events, to_simulator_format
        from signals.simulator import SignalSimulator
        from engine.habit_engine import HabitDemoEngine

        # 加载周数选择 (session state)
        week_nums = st.session_state.get("t1_week_selection", [1])

        # 从 unified 数据加载事件
        events, manifest = load_multi_week_events(week_nums)
        mockup = to_simulator_format(events)
        sim = SignalSimulator(mockup)
        week_label = ", ".join(f"W{w}" for w in week_nums)
        progress_bar.progress(10, text=f"Loaded {sim.total_events} events ({week_label})...")

        # 重置目标用户数据
        progress_bar.progress(20, text="Initializing engine + ChromaDB...")
        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        engine.reset()
        progress_bar.progress(30, text="Embedding facts (batched OpenAI call)...")

        # 运行完整 pipeline
        result = engine.ingest_from_simulator(sim)
        progress_bar.progress(85, text="Clustering complete. Finalizing...")

        # 更新 cycle 计数 (按周数)
        total_days = len(week_nums) * 7
        st.session_state["t1_cycles"][username] = (
            st.session_state["t1_cycles"].get(username, 0) + total_days
        )

        # 记录新 habit IDs (按用户隔离)
        new_ids = {h["id"] for h in result.get("new_habits", [])}
        st.session_state[f"t1_new_habit_ids_{username}"] = new_ids
        st.session_state[f"t1_last_result_{username}"] = result

        progress_bar.progress(100, text="Batch load complete!")

        st.success(
            f"Loaded {sim.total_events} events ({week_label}) → "
            f"{result['facts_ingested']} facts → "
            f"{result['habits_detected']} habits detected"
        )
        _show_pipeline_metrics(result)

        # 清除可视化缓存
        _invalidate_viz_cache(username)

        engine.close()

    except Exception as e:
        st.error(f"Batch load error: {e}")
        log.exception("Batch load error")


# ═══════════════════════════════════════════════════
# 显示逻辑
# ═══════════════════════════════════════════════════

def _show_pipeline_metrics(result: dict):
    """pipeline 结果指标行"""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Facts Ingested", result.get("facts_ingested", 0))
    with c2:
        st.metric("Before Clustering", result.get("total_before_clustering", 0))
    with c3:
        st.metric("Habits Detected", result.get("habits_detected", 0))
    with c4:
        st.metric("Facts Clustered", result.get("facts_clustered", 0))


def _show_knowledge_graph(username: str):
    """加载并展示 Knowledge Graph"""
    try:
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        status = engine.get_status()
        engine.close()

        # 头部指标
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Total Facts", status["total_facts"])
        with mc2:
            st.metric("Pref Facts", status["pref_facts"])
        with mc3:
            st.metric("Habit Facts", status["habit_facts"])

        # KG 表格 — 使用按用户隔离的 new_habit_ids
        new_ids = st.session_state.get(f"t1_new_habit_ids_{username}", set())
        render_knowledge_graph(
            habits=status["habits"],
            new_habit_ids=new_ids,
            username=username,
        )

    except Exception as e:
        st.info("No data available. Load mockup data or run analysis to start learning.")
        log.debug(f"KG load: {e}")


# ═══════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════

def _build_event(ctx_data: dict, signals: dict) -> dict:
    """从 UI 输入构造信号事件 dict"""
    date_str = ctx_data["date"]
    time_str = ctx_data["time"]
    ts = f"{date_str}T{time_str}"

    signal_list = [
        {"t": ts, "signal": "engine_status", "value": "on"},
    ]
    for sig_name, value in signals.items():
        signal_list.append({
            "t": ts,
            "signal": sig_name,
            "value": value,
        })

    return {
        "date": date_str,
        "signals": signal_list,
    }


def _increment_cycle(username: str):
    """递增 cycle 计数"""
    st.session_state["t1_cycles"][username] = (
        st.session_state["t1_cycles"].get(username, 0) + 1
    )


def _invalidate_viz_cache(username: str):
    """清除聚类可视化缓存，使下次渲染时重新计算"""
    keys_to_remove = [k for k in st.session_state if k.startswith(f"viz_cache_{username.lower()}_")]
    for k in keys_to_remove:
        del st.session_state[k]
