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
from pathlib import Path

from simulator.components.context_panel import render_context_panel
from simulator.components.behavior_table import render_behavior_table, signals_summary
from simulator.components.knowledge_graph import render_knowledge_graph
from simulator.components.cluster_viz import render_cluster_visualization

log = logging.getLogger(__name__)

# 可选用户列表 (对齐 mockup 数据)
USERS = ["Mary", "Tom", "Alice", "David", "Lena"]
TARGET_CYCLES = 20  # 目标学习轮次


@st.cache_resource(show_spinner=False)
def _get_llm_client():
    """Per-session singleton LLMClient — drives GPT-4.1 habit-text rewording
    via the OpenAI → OpenRouter → tunnel → cache fallback chain."""
    from panoramix_core.llm_client import LLMClient
    return LLMClient()


def _init_session():
    """初始化 session state"""
    if "t1_cycles" not in st.session_state:
        st.session_state["t1_cycles"] = {}


def render():
    """渲染 Tab 1 完整页面"""
    _init_session()

    # ── 顶栏: 用户选择器 + Cycle 徽章 + 批量/单步/清理按钮 ──
    col_user, col_cycle, col_batch, col_step, col_clear = st.columns([2, 1.3, 1.6, 1.6, 1.2])

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

    with col_step:
        step_cursor = st.session_state.get(f"t1_step_cursor_{username}", 0)
        step_total = len(st.session_state.get(f"t1_step_events_{username}", []))
        step_label = (
            f"Load Next Event ({step_cursor}/{step_total})"
            if step_total else "Load Next Event"
        )
        step_clicked = st.button(
            step_label,
            key="t1_step_load",
            help="Feed mockup events into the pipeline one-by-one so you can "
                 "watch the Knowledge Graph evolve per action. Uses the same "
                 "Weeks selection as Load Unified Data; changing the selection "
                 "resets the step queue.",
        )
        step_reset_clicked = st.button(
            "Reset Step Queue",
            key="t1_step_reset",
            help="Drop the cached step-by-step event queue (does NOT clear "
                 "ChromaDB — use Clear Data for that).",
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

    # ── 批量加载 / 单步加载 / 清理处理 ──
    if batch_clicked:
        _run_batch_load(username)
    if step_clicked:
        _run_step_load(username)
    if step_reset_clicked:
        _reset_step_queue(username)
    if clear_clicked:
        _run_clear_data(username)

    st.divider()

    # ── Day-1 场景视频（Load Unified Data 后显示，Clear Data 后移除）──
    _render_day1_videos(username)

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

        # 左下: 最近一次 step-load 的 event preview（trigger / action / 抽取的 facts）
        _render_loaded_event_preview(username)

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

        engine = HabitDemoEngine(username=username.lower(), llm_client=_get_llm_client())
        progress_bar.progress(40, text="Ingesting signals → facts...")

        result = engine.ingest_signal_batch([event])
        progress_bar.progress(80, text="Running DBSCAN clustering...")

        # 更新 cycle 计数
        _increment_cycle(username)

        # 记录新 habit IDs (按用户隔离)
        new_ids = {h["id"] for h in result.get("new_habits", [])}
        st.session_state[f"t1_new_habit_ids_{username}"] = new_ids
        st.session_state[f"t1_last_result_{username}"] = result

        # 保存可视化数据
        viz_items = result.pop("_viz_items", None)
        if viz_items:
            st.session_state[f"t1_viz_items_{username.lower()}"] = viz_items
        _invalidate_viz_cache(username)

        progress_bar.progress(100, text="Knowledge Graph updated!")
        _show_pipeline_metrics(result)

        engine.close()

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        log.exception("Single analysis pipeline error")


def _run_clear_data(username: str):
    """清空当前用户的全部 ChromaDB 数据 + 重置 session state"""
    try:
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username=username.lower(), llm_client=_get_llm_client())
        before = engine.get_status()
        engine.reset()
        after = engine.get_status()
        engine.close()

        # 清掉 session state 里与该用户有关的缓存
        st.session_state["t1_cycles"][username] = 0
        for k in (
            f"t1_new_habit_ids_{username}",
            f"t1_last_result_{username}",
            f"t1_step_events_{username}",
            f"t1_step_cursor_{username}",
            f"t1_step_weeks_{username}",
            f"t1_last_loaded_event_{username}",
            f"t1_last_loaded_facts_{username}",
            f"t1_show_videos_{username}",
        ):
            st.session_state.pop(k, None)
        _invalidate_viz_cache(username)

        st.success(
            f"Cleared data for '{username}': "
            f"{before['pref_facts']} PREF facts, "
            f"{before['habits_count']} habits, "
            f"{before['scene_cards_count']} scene cards removed"
        )
    except Exception as e:
        st.error(f"Clear data error: {e}")
        log.exception("Clear data error")


_DAY1_VIDEO_ORDER = ("morning_commute", "toll_parking_entry", "arriving_home")
_DAY1_VIDEO_LABELS = {
    "morning_commute": "Morning Commute",
    "toll_parking_entry": "Toll & Parking Entry",
    "arriving_home": "Arriving Home",
}


def _render_day1_videos(username: str):
    """Load Unified Data 后横向展示三个 Day-1 场景视频。Clear Data 后隐藏。"""
    if not st.session_state.get(f"t1_show_videos_{username}"):
        return

    video_dir = Path(__file__).resolve().parent.parent / "output" / "videos" / username.lower() / "day1"
    if not video_dir.is_dir():
        return

    videos = [(name, video_dir / name / "scene.mp4") for name in _DAY1_VIDEO_ORDER]
    videos = [(name, p) for name, p in videos if p.is_file()]
    if not videos:
        return

    st.markdown("**Day-1 Scene Videos**")
    cols = st.columns(len(videos))
    for col, (name, path) in zip(cols, videos):
        with col:
            st.caption(_DAY1_VIDEO_LABELS.get(name, name))
            st.video(str(path))
    st.divider()


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

        # 初始化引擎（不 reset，保留已有 accepted 场景卡）
        progress_bar.progress(20, text="Initializing engine + ChromaDB...")
        engine = HabitDemoEngine(username=username.lower(), llm_client=_get_llm_client())
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

        # 保存可视化数据（pipeline 删除 Chroma 前捕获的 items）
        viz_items = result.pop("_viz_items", None)
        if viz_items:
            st.session_state[f"t1_viz_items_{username.lower()}"] = viz_items
        _invalidate_viz_cache(username)

        progress_bar.progress(100, text="Batch load complete!")

        # 标记本用户已加载 unified data → 渲染 Day-1 场景视频
        st.session_state[f"t1_show_videos_{username}"] = True

        st.success(
            f"Loaded {sim.total_events} events ({week_label}) → "
            f"{result['facts_ingested']} facts → "
            f"{result['habits_count']} habits detected"
        )
        _show_pipeline_metrics(result)

        engine.close()

    except Exception as e:
        st.error(f"Batch load error: {e}")
        log.exception("Batch load error")


def _reset_step_queue(username: str):
    """丢弃缓存的步进事件队列（不触碰 ChromaDB）"""
    for k in (
        f"t1_step_events_{username}",
        f"t1_step_cursor_{username}",
        f"t1_step_weeks_{username}",
        f"t1_last_loaded_event_{username}",
        f"t1_last_loaded_facts_{username}",
    ):
        st.session_state.pop(k, None)
    st.info(f"Step queue reset for '{username}'. ChromaDB untouched.")


# Action signal 名（与 behavior_table.ACTION_LIST 对齐）
_ACTION_SIGNAL_NAMES = {
    "hvac_power", "hvac_temp_target",
    "nav_destination", "nav_route_pref",
    "media_volume",
    "window_position", "keyless_entry",
}
_WINDOW_PCT_TO_STATE = {0: "Closed", 50: "Half Open", 100: "Open"}


def _apply_event_to_widgets(event: dict):
    """把 event 的 signal/上下文回灌到左侧 Trigger/Action 面板的 widget 状态，
    让 '下一次渲染' 直接显示这条 mockup 的内容。
    只写 session_state，不新建 widget，避免与现有控件冲突。"""
    from datetime import datetime as _dt
    from data.unified_loader import PLACE_COORDS
    from simulator.components.context_panel import (
        LOCATION_OPTIONS, WEATHER_OPTIONS, GEAR_OPTIONS, WIPER_OPTIONS,
    )

    signals_by_name: dict = {}
    for s in event.get("signals", []):
        signals_by_name[s.get("signal")] = s.get("value")

    # ── date / time: 取第一条带时间戳的 signal ──
    first_ts = next(
        (s.get("t") for s in event.get("signals", []) if "T" in str(s.get("t", ""))),
        None,
    )
    if first_ts:
        try:
            parsed = _dt.fromisoformat(first_ts)
            st.session_state["t1_ctx_date"] = parsed.date()
            st.session_state["t1_ctx_time"] = parsed.time()
        except ValueError:
            pass

    # ── Location: GPS 反查 place_id ──
    lat = signals_by_name.get("gps_latitude")
    lng = signals_by_name.get("gps_longitude")
    if lat is not None and lng is not None:
        for pid, coords in PLACE_COORDS.items():
            if pid not in LOCATION_OPTIONS:
                continue
            plat, plng = coords
            if abs(plat - lat) < 1e-4 and abs(plng - lng) < 1e-4:
                st.session_state["t1_ctx_location"] = pid
                break

    # ── Weather: 直接来自 event header ──
    weather = event.get("weather")
    if weather in WEATHER_OPTIONS:
        st.session_state["t1_ctx_weather"] = weather

    # ── Speed / Gear / Wiper ──
    if "vehicle_speed" in signals_by_name:
        try:
            st.session_state["t1_ctx_speed"] = float(signals_by_name["vehicle_speed"])
        except (TypeError, ValueError):
            pass
    gear = signals_by_name.get("gear_position")
    if gear in GEAR_OPTIONS:
        st.session_state["t1_ctx_gear"] = gear
    wiper = signals_by_name.get("wiper_state")
    if wiper in WIPER_OPTIONS:
        st.session_state["t1_ctx_wiper"] = wiper

    # ── Action 面板: 打开命中的 signal、关闭未命中的 ──
    for sig_name in _ACTION_SIGNAL_NAMES:
        en_key = f"t1_beh_{sig_name}_en"
        val_key = f"t1_beh_{sig_name}_val"
        if sig_name in signals_by_name:
            st.session_state[en_key] = True
            val = signals_by_name[sig_name]
            if sig_name == "window_position":
                try:
                    st.session_state[val_key] = _WINDOW_PCT_TO_STATE.get(
                        int(val), "Closed"
                    )
                except (TypeError, ValueError):
                    st.session_state[val_key] = "Closed"
            elif sig_name == "hvac_power":
                pass  # toggle 型，只需打开 _en
            elif sig_name == "hvac_temp_target" or sig_name == "media_volume":
                try:
                    st.session_state[val_key] = int(val)
                except (TypeError, ValueError):
                    pass
            else:
                st.session_state[val_key] = val
        else:
            st.session_state[en_key] = False


def _render_loaded_event_preview(username: str):
    """左侧最底部: 展示最后一次 step-load 的 event 详情 + 抽取到的 PREF facts。"""
    event = st.session_state.get(f"t1_last_loaded_event_{username}")
    facts = st.session_state.get(f"t1_last_loaded_facts_{username}")
    if not event:
        return

    date_str = event.get("date", "?")
    weekday = event.get("weekday", "")
    header = f"Last Loaded Event · {date_str} ({weekday})"

    with st.expander(header, expanded=True):
        signals = event.get("signals", [])
        trigger_sigs = [s for s in signals if s.get("signal") not in _ACTION_SIGNAL_NAMES]
        action_sigs = [s for s in signals if s.get("signal") in _ACTION_SIGNAL_NAMES]

        col_t, col_a = st.columns(2)
        with col_t:
            st.markdown("**Trigger signals**")
            if trigger_sigs:
                st.dataframe(
                    [
                        {"signal": s.get("signal"), "value": s.get("value"),
                         "t": s.get("t", "").split("T")[-1]}
                        for s in trigger_sigs
                    ],
                    hide_index=True, use_container_width=True,
                )
            else:
                st.caption("—")
        with col_a:
            st.markdown("**Action signals**")
            if action_sigs:
                st.dataframe(
                    [
                        {"signal": s.get("signal"), "value": s.get("value"),
                         "t": s.get("t", "").split("T")[-1]}
                        for s in action_sigs
                    ],
                    hide_index=True, use_container_width=True,
                )
            else:
                st.caption("—")

        st.markdown("**Extracted PREF facts**")
        if facts:
            st.dataframe(
                [{"text": f.get("text"), "durability": f.get("durability")}
                 for f in facts],
                hide_index=True, use_container_width=True,
            )
        else:
            st.caption("No PREF fact extracted from this event.")


def _run_step_load(username: str):
    """单步模式: 每次点击把 mockup 中的下一条事件喂给 pipeline，观察 KG 逐步演化。"""
    try:
        from data.unified_loader import load_multi_week_events, to_simulator_format
        from signals.simulator import SignalSimulator
        from engine.habit_engine import HabitDemoEngine

        week_nums = tuple(sorted(st.session_state.get("t1_week_selection", [1])))
        if not week_nums:
            st.warning("Please select at least one week first.")
            return

        # 若用户切换了周数或还未初始化，则重建队列（但不清 ChromaDB）
        cached_weeks = st.session_state.get(f"t1_step_weeks_{username}")
        events = st.session_state.get(f"t1_step_events_{username}")
        if cached_weeks != week_nums or events is None:
            raw_events, _manifest = load_multi_week_events(list(week_nums))
            mockup = to_simulator_format(raw_events)
            sim = SignalSimulator(mockup)
            events = list(sim._events)  # 已按时间排序展平
            st.session_state[f"t1_step_events_{username}"] = events
            st.session_state[f"t1_step_cursor_{username}"] = 0
            st.session_state[f"t1_step_weeks_{username}"] = week_nums
            week_label = ", ".join(f"W{w}" for w in week_nums)
            st.info(
                f"Step queue built: {len(events)} events "
                f"({week_label}). Click again to feed the first event."
            )
            return

        cursor = st.session_state.get(f"t1_step_cursor_{username}", 0)
        if cursor >= len(events):
            st.warning(
                f"All {len(events)} events already fed. "
                "Use 'Reset Step Queue' to start over."
            )
            return

        raw_event = events[cursor]
        # 剥掉离线展平阶段推断的 scene 标签——scene 只应由下游场景卡总结得出
        event = {k: v for k, v in raw_event.items() if k != "scene"}
        progress_bar = st.progress(0, text=f"Feeding event {cursor + 1}/{len(events)}...")

        # 预先抽取 PREF facts（用于左下 preview 面板）
        from engine.signal_to_fact import signals_to_facts as _s2f
        try:
            preview_facts = _s2f(event)
        except Exception as e:
            log.warning(f"Preview fact extraction failed: {e}")
            preview_facts = []
        st.session_state[f"t1_last_loaded_event_{username}"] = event
        st.session_state[f"t1_last_loaded_facts_{username}"] = [
            {
                "text": f.text,
                "durability": getattr(f.durability, "value", str(f.durability)),
            }
            for f in preview_facts
        ]

        # 把 event 回灌到左侧 Trigger / Action 面板 widget 状态
        _apply_event_to_widgets(event)

        engine = HabitDemoEngine(username=username.lower(), llm_client=_get_llm_client())
        progress_bar.progress(30, text="Ingesting signals → facts...")
        result = engine.ingest_signal_batch([event])
        progress_bar.progress(85, text="Running DBSCAN clustering...")

        # 更新 cycle 计数 (步进模式 +1)
        _increment_cycle(username)

        # 记录新 habit IDs (按用户隔离)
        new_ids = {h["id"] for h in result.get("new_habits", [])}
        st.session_state[f"t1_new_habit_ids_{username}"] = new_ids
        st.session_state[f"t1_last_result_{username}"] = result

        viz_items = result.pop("_viz_items", None)
        if viz_items:
            st.session_state[f"t1_viz_items_{username.lower()}"] = viz_items
        _invalidate_viz_cache(username)

        st.session_state[f"t1_step_cursor_{username}"] = cursor + 1
        progress_bar.progress(100, text="Knowledge Graph updated!")

        sig_count = len(event.get("signals", []))
        st.success(
            f"Step {cursor + 1}/{len(events)} fed · day={event.get('day', '?')} "
            f"date={event.get('date', '?')} signals={sig_count}"
        )
        _show_pipeline_metrics(result)

        engine.close()

    except Exception as e:
        st.error(f"Step load error: {e}")
        log.exception("Step load error")


# ═══════════════════════════════════════════════════
# 显示逻辑
# ═══════════════════════════════════════════════════

def _show_pipeline_metrics(result: dict):
    """pipeline 结果指标行"""
    cls = result.get("classification", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Facts Ingested", result.get("facts_ingested", 0))
    with c2:
        st.metric("Habits Detected", result.get("habits_count", 0))
    with c3:
        st.metric("New Pending", cls.get("new_pending", 0))
    with c4:
        st.metric("Reinforce / Drift", f"{cls.get('reinforce', 0)} / {cls.get('modify_drift', 0)}")


def _show_knowledge_graph(username: str):
    """加载并展示 Knowledge Graph"""
    try:
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username=username.lower(), llm_client=_get_llm_client())
        status = engine.get_status()
        engine.close()

        # 头部指标
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("PREF Facts", status["pref_facts"])
        with mc2:
            st.metric("Habits", status["habits_count"])
        with mc3:
            st.metric("Scene Cards", status["scene_cards_count"])
        with mc4:
            st.metric("Batch ID", status["batch_id"])

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
    """从 UI 输入构造信号事件 dict (trigger + action signals)"""
    date_str = ctx_data["date"]
    time_str = ctx_data["time"]
    ts = f"{date_str}T{time_str}"

    signal_list = []

    # ── Trigger signals (vehicle state context) ──
    if ctx_data.get("gps_latitude") is not None:
        signal_list.append({"t": ts, "signal": "gps_latitude", "value": ctx_data["gps_latitude"]})
        signal_list.append({"t": ts, "signal": "gps_longitude", "value": ctx_data["gps_longitude"]})

    signal_list.append({"t": ts, "signal": "vehicle_speed", "value": ctx_data.get("speed_kph", 0.0)})
    signal_list.append({"t": ts, "signal": "gear_position", "value": ctx_data.get("gear", "P")})

    wiper = ctx_data.get("wiper", "off")
    if wiper != "off":
        signal_list.append({"t": ts, "signal": "wiper_state", "value": wiper})

    # ── Action signals (user behaviors) ──
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
