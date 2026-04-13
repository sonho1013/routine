"""
Tab 2: Smart Habit Recommender — 推理阶段 (完整实现)

功能:
  1. 三栏布局: Mini KG (只读) | Scenario+Context | Suggestion Panel
  2. Scenario 选择器 (6 场景) + Intention 选择器 (5 意图) + 推断标签
  3. Context Data 面板 (6 行 icon 展示)
  4. "Get Recommendation" → ProactiveExecutor 匹配 → 推荐卡片
  5. Apply / Dismiss / Modify 反馈 (三按钮)
  6. 数据不足提示
  7. Mini KG 只读联动 (高亮匹配场景)
"""
import streamlit as st
import logging
from datetime import datetime, time as dt_time
from typing import Optional

from engine.proactive_executor import parse_habit_actions

log = logging.getLogger(__name__)

# 对齐 Tab 1 用户列表
USERS = ["Mary", "Tom", "Alice", "David", "Lena"]

# ═══════════════════════════════════════════════════
# Scenario / Intention 定义
# ═══════════════════════════════════════════════════

SCENARIOS = {
    "Morning Commute": {
        "time": dt_time(7, 30),
        "time_bucket": "early_morning",
        "vehicle_state": "engine_started",
        "geofence": "home",
        "weekday": True,
        "weather": "Cloudy",
        "outside_temp": 8.0,
        "battery_soc": 85,
        "passengers": 1,
        "icon": "🌅",
    },
    "Weekend Leisure": {
        "time": dt_time(10, 0),
        "time_bucket": "morning",
        "vehicle_state": "engine_started",
        "geofence": None,
        "weekday": False,
        "weather": "Sunny",
        "outside_temp": 18.0,
        "battery_soc": 92,
        "passengers": 2,
        "icon": "🌤",
    },
    "Evening Return": {
        "time": dt_time(18, 30),
        "time_bucket": "evening",
        "vehicle_state": "engine_started",
        "geofence": None,
        "weekday": True,
        "weather": "Cloudy",
        "outside_temp": 12.0,
        "battery_soc": 40,
        "passengers": 1,
        "icon": "🌆",
    },
    "Traffic Jam": {
        "time": dt_time(8, 45),
        "time_bucket": "morning",
        "vehicle_state": "engine_started",
        "geofence": "toll",
        "weekday": True,
        "weather": "Rainy",
        "outside_temp": 10.0,
        "battery_soc": 70,
        "passengers": 1,
        "icon": "🚗",
    },
    "Night Drive": {
        "time": dt_time(22, 30),
        "time_bucket": "night",
        "vehicle_state": "engine_started",
        "geofence": None,
        "weekday": False,
        "weather": "Cloudy",
        "outside_temp": 5.0,
        "battery_soc": 55,
        "passengers": 1,
        "icon": "🌙",
    },
    "Charging Station": {
        "time": dt_time(14, 0),
        "time_bucket": "afternoon",
        "vehicle_state": "parking",
        "geofence": None,
        "weekday": True,
        "weather": "Sunny",
        "outside_temp": 22.0,
        "battery_soc": 15,
        "passengers": 0,
        "icon": "🔋",
    },
}

INTENTIONS = [
    "Routine Driving",
    "Leisure",
    "Return Home",
    "Reduce Stress",
    "Improve Visibility",
]

# Scenario → 推断 Intention 映射
_INFERRED_INTENT = {
    "Morning Commute": "Routine Driving",
    "Weekend Leisure": "Leisure",
    "Evening Return": "Return Home",
    "Traffic Jam": "Reduce Stress",
    "Night Drive": "Improve Visibility",
    "Charging Station": "Routine Driving",
}


# ═══════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════

def render():
    """渲染 Tab 2 完整页面 — 三栏布局"""

    # ── 顶栏: 用户选择器 ──
    col_user, col_spacer = st.columns([2, 3])
    with col_user:
        username = st.selectbox("Driver Profile", USERS, key="t2_user")

    # ── 按用户初始化 session state ──
    _ensure_user_state(username)

    st.divider()

    # ── 三栏主布局 ──
    col_kg, col_ctx, col_sug = st.columns([3, 3.5, 5.5])

    with col_kg:
        _render_mini_kg(username)

    with col_ctx:
        _render_scenario_context(username)

    with col_sug:
        _render_suggestion_panel(username)


def _ensure_user_state(username: str):
    """按用户隔离 session state，避免切换用户时数据泄漏"""
    if f"t2_recommendations_{username}" not in st.session_state:
        st.session_state[f"t2_recommendations_{username}"] = None
    if f"t2_feedback_log_{username}" not in st.session_state:
        st.session_state[f"t2_feedback_log_{username}"] = []
    if f"t2_matched_scene_{username}" not in st.session_state:
        st.session_state[f"t2_matched_scene_{username}"] = None


# ═══════════════════════════════════════════════════
# COL A — Mini Knowledge Graph (只读)
# ═══════════════════════════════════════════════════

def _render_mini_kg(username: str):
    """只读 Mini KG — 展示已学习的习惯，高亮匹配场景"""
    st.markdown("**Knowledge Graph**")

    try:
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        status = engine.get_status()
        engine.close()

        habits = status["habits"]
        if not habits:
            st.caption("No habits learned yet. Use Tab 1 to train the model first.")
            return

        # 统计
        total = len(habits)
        accepted = sum(1 for h in habits if h.get("accepted"))
        st.caption(f"{total} patterns · {accepted} accepted")

        matched_scene = st.session_state.get(f"t2_matched_scene_{username}")

        # 按置信度排序
        sorted_habits = sorted(
            habits,
            key=lambda h: h.get("clustering_confidence") or 0,
            reverse=True,
        )

        rows = []
        for h in sorted_habits:
            scene = h.get("scene_name") or "—"
            conf = h.get("clustering_confidence") or 0.0
            accepted_flag = h.get("accepted", False)

            is_match = (
                matched_scene
                and scene.lower().replace(" ", "") == matched_scene.lower().replace(" ", "")
            )
            row_style = "background:rgba(0,200,150,0.1);border-left:2px solid #00C896;" if is_match else ""
            status_icon = "✅" if accepted_flag else "⏳"

            if conf >= 0.85:
                conf_color, conf_bg = "#00C896", "rgba(0,200,150,0.15)"
            elif conf >= 0.70:
                conf_color, conf_bg = "#00BFC8", "rgba(0,191,200,0.15)"
            else:
                conf_color, conf_bg = "#8899AA", "rgba(136,153,170,0.12)"

            rows.append(
                f'<tr style="{row_style}">'
                f'<td style="padding:6px 4px;font-size:0.7rem;">{status_icon}</td>'
                f'<td style="padding:6px 4px;font-size:0.75rem;font-weight:500;'
                f'color:#E8EDF3;">{scene}</td>'
                f'<td style="padding:6px 4px;text-align:center;">'
                f'<span style="display:inline-block;padding:1px 6px;border-radius:4px;'
                f'font-size:0.75rem;font-weight:600;color:{conf_color};'
                f'background:{conf_bg};">{conf:.2f}</span></td>'
                f'</tr>'
            )

        table_html = (
            '<div style="font-family:sans-serif;color:#E8EDF3;">'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead><tr style="border-bottom:2px solid #1E3A5F;">'
            '<th style="padding:6px 4px;font-size:0.7rem;color:#00BFC8;width:24px;"></th>'
            '<th style="padding:6px 4px;font-size:0.7rem;color:#00BFC8;text-align:left;">Scenario</th>'
            '<th style="padding:6px 4px;font-size:0.7rem;color:#00BFC8;text-align:center;">Conf.</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table>'
            '<div style="font-size:0.65rem;color:#8899AA;margin-top:8px;">'
            '<span style="color:#00C896;">●</span> High ≥0.85 &nbsp;'
            '<span style="color:#00BFC8;">●</span> Mid 0.70+ &nbsp;'
            '<span style="color:#8899AA;">●</span> Low &lt;0.70'
            '</div></div>'
        )
        st.html(table_html)

    except Exception as e:
        st.caption("No data available. Train the model in Tab 1 first.")
        log.debug(f"Mini KG load: {e}")


# ═══════════════════════════════════════════════════
# COL B — Scenario + Intention + Context + Button
# ═══════════════════════════════════════════════════

def _render_scenario_context(username: str):
    """场景/意图选择器 + 上下文面板 + Get Recommendation 按钮"""

    # ── Scenario 选择器 ──
    st.markdown("**Scenario & Context**")

    scenario_name = st.selectbox(
        "Scenario",
        list(SCENARIOS.keys()),
        key="t2_scenario",
    )
    scenario = SCENARIOS[scenario_name]

    # ── Intention 选择器 ──
    default_intent = _INFERRED_INTENT.get(scenario_name, "Routine Driving")
    intent_idx = INTENTIONS.index(default_intent) if default_intent in INTENTIONS else 0

    intention = st.selectbox(
        "Intention",
        INTENTIONS,
        index=intent_idx,
        key="t2_intention",
    )

    # ── 推断意图标签 ──
    inferred = _INFERRED_INTENT.get(scenario_name, "")
    if inferred:
        st.caption(f"Inferred intent: **{inferred}**")

    # ── Context Data 面板 (icon 行) ──
    ctx_rows = [
        ("📅", "Day", "Weekday" if scenario["weekday"] else "Weekend"),
        ("🕐", "Time", scenario["time"].strftime("%H:%M") + f' ({scenario["time_bucket"].replace("_", " ").title()})'),
        ("🌤", "Weather", f'{scenario["weather"]}, {scenario["outside_temp"]}°C'),
        ("🚗", "Vehicle", scenario["vehicle_state"].replace("_", " ").title()),
        ("👤", "Passengers", str(scenario["passengers"])),
        ("🔋", "Battery", f'{scenario["battery_soc"]}%'),
    ]

    ctx_html_rows = "".join(
        f'<div style="display:flex;align-items:center;padding:5px 0;'
        f'border-bottom:1px solid rgba(30,58,95,0.4);">'
        f'<span style="width:24px;font-size:0.85rem;">{icon}</span>'
        f'<span style="flex:1;font-size:0.75rem;color:#8899AA;">{label}</span>'
        f'<span style="font-size:0.8rem;color:#E8EDF3;font-weight:500;">{value}</span>'
        f'</div>'
        for icon, label, value in ctx_rows
    )
    st.html(
        f'<div style="background:#132035;border:1px solid #1E3A5F;border-radius:6px;'
        f'padding:10px 14px;font-family:sans-serif;">{ctx_html_rows}</div>'
    )

    # ── Get Recommendation 按钮 ──
    recommend_clicked = st.button(
        "Get Recommendation",
        type="primary",
        use_container_width=True,
        key="t2_recommend",
    )

    if recommend_clicked:
        _run_recommendation(username, scenario_name, scenario, intention)


# ═══════════════════════════════════════════════════
# COL C — Suggestion Panel
# ═══════════════════════════════════════════════════

def _render_suggestion_panel(username: str):
    """推荐结果展示: 卡片 + 反馈"""
    st.markdown("**Recommendations**")

    data = st.session_state.get(f"t2_recommendations_{username}")

    if data is None:
        st.info("Select a scenario and click 'Get Recommendation' to find matching habits.")
        return

    # ── 指标栏 ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Habits", data["total_habits"])
    with c2:
        st.metric("Candidates", data["candidates_checked"])
    with c3:
        st.metric("Matches", len(data["actions"]))

    if not data["has_recommendations"]:
        _render_insufficient_data(data)
        return

    # ── 顶部场景标签 + 最佳置信度 + 进度条 ──
    top = data["actions"][0]
    top_conf_pct = int(top["combined_confidence"] * 100)
    bar_pct = min(100, top_conf_pct)
    if bar_pct >= 70:
        bar_color = "#00C896"
    elif bar_pct >= 40:
        bar_color = "#F5A623"
    else:
        bar_color = "#E74C3C"

    st.html(
        f'<div style="font-family:sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:10px;">'
        f'<span style="display:inline-block;background:rgba(0,191,200,0.15);color:#00BFC8;'
        f'padding:4px 12px;border-radius:4px;font-size:0.85rem;font-weight:600;">'
        f'{data["scenario_name"]}</span>'
        f'<span style="color:{bar_color};font-size:1.2rem;font-weight:700;">'
        f'{top_conf_pct}%</span></div>'
        f'<div style="height:4px;background:#1E3A5F;border-radius:4px;overflow:hidden;">'
        f'<div style="height:100%;width:{bar_pct}%;background:{bar_color};'
        f'border-radius:4px;"></div></div></div>'
    )

    # ── 推荐卡片列表 ──
    for i, action in enumerate(data["actions"]):
        _render_action_card(action, i, data["username"])

    # ── 反馈日志 ──
    feedback_log = st.session_state.get(f"t2_feedback_log_{username}", [])
    if feedback_log:
        with st.expander("Feedback History", expanded=False):
            for entry in reversed(feedback_log[-10:]):
                icon = {"accept": "✅", "dismiss": "❌", "modify": "✏️"}.get(entry["action"], "•")
                st.caption(
                    f'{icon} {entry["action"].title()} — {entry["scene"]} '
                    f'({entry["time"]})'
                )


def _render_action_card(action: dict, index: int, username: str):
    """渲染单张推荐卡片 + Apply/Dismiss/Modify 反馈"""
    conf = action["combined_confidence"]
    match_pct = int(action["match_confidence"] * 100)
    conf_pct = int(conf * 100)
    scene = action["scene_name"] or "Unknown"
    habit_id = action["habit_id"]

    # 置信度颜色
    if conf_pct >= 70:
        conf_color = "#00C896"
    elif conf_pct >= 40:
        conf_color = "#F5A623"
    else:
        conf_color = "#E74C3C"

    # 卡片 HTML
    st.html(
        f'<div style="background:#1A2A42;border:1px solid #1E3A5F;'
        f'border-left:3px solid {conf_color};border-radius:8px;'
        f'padding:14px 16px;font-family:sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="display:inline-block;background:rgba(0,191,200,0.15);'
        f'color:#00BFC8;padding:3px 10px;border-radius:4px;font-size:0.78rem;font-weight:500;">'
        f'{scene}</span>'
        f'<span style="color:{conf_color};font-weight:700;font-size:1.15rem;">'
        f'{conf_pct}%</span></div>'
        f'<div style="margin:10px 0 6px 0;font-size:0.88rem;color:#E8EDF3;line-height:1.5;">'
        f'{action["habit_text"]}</div>'
        f'<div style="font-size:0.72rem;color:#8899AA;">'
        f'Match: {match_pct}% · Distance: {action["context_distance"]:.3f}'
        f'</div></div>'
    )

    # 解析出的控制参数
    parsed = action.get("parsed_actions", {})
    if parsed:
        _LABELS = {
            "hvac_temp_target": "AC",
            "hvac_power": "AC Power",
            "seat_heating": "Seat Heat",
            "nav_destination": "Nav",
            "nav_route_pref": "Route",
            "media_source": "Media",
            "media_content_id": "Content",
            "media_volume": "Vol",
            "media_off": "Media",
            "drive_mode": "Mode",
            "acc_distance": "ACC",
            "window_position": "Window",
        }
        params_html = []
        for k, v in parsed.items():
            label = _LABELS.get(k, k)
            if k == "hvac_temp_target":
                params_html.append(f'<span style="background:#132035;border:1px solid #1E3A5F;'
                                   f'padding:2px 8px;border-radius:4px;font-size:0.75rem;'
                                   f'color:#E8EDF3;margin:2px;">{label}={v}°C</span>')
            elif k == "media_volume":
                params_html.append(f'<span style="background:#132035;border:1px solid #1E3A5F;'
                                   f'padding:2px 8px;border-radius:4px;font-size:0.75rem;'
                                   f'color:#E8EDF3;margin:2px;">{label}={v}%</span>')
            elif k == "media_off":
                params_html.append(f'<span style="background:#132035;border:1px solid #1E3A5F;'
                                   f'padding:2px 8px;border-radius:4px;font-size:0.75rem;'
                                   f'color:#E8EDF3;margin:2px;">Media=OFF</span>')
            else:
                params_html.append(f'<span style="background:#132035;border:1px solid #1E3A5F;'
                                   f'padding:2px 8px;border-radius:4px;font-size:0.75rem;'
                                   f'color:#E8EDF3;margin:2px;">{label}={v}</span>')

        st.html(
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;font-family:sans-serif;">'
            + "".join(params_html)
            + '</div>'
        )

    # ── Apply / Dismiss / Modify 按钮 ──
    c_apply, c_dismiss, c_modify = st.columns(3)

    with c_apply:
        if st.button("Apply", key=f"t2_apply_{index}_{habit_id}", type="primary",
                      use_container_width=True):
            _handle_feedback(username, habit_id, scene, "accept")

    with c_dismiss:
        if st.button("Dismiss", key=f"t2_dismiss_{index}_{habit_id}",
                      use_container_width=True):
            _handle_feedback(username, habit_id, scene, "dismiss")

    with c_modify:
        if st.button("Modify", key=f"t2_modify_{index}_{habit_id}",
                      use_container_width=True):
            st.session_state[f"t2_modify_open_{habit_id}"] = True

    # ── Modify 展开区域 ──
    if st.session_state.get(f"t2_modify_open_{habit_id}"):
        with st.container():
            st.caption("Adjustment note (will be used for retraining):")
            note = st.text_input(
                "Feedback note",
                key=f"t2_modify_note_{habit_id}",
                label_visibility="collapsed",
                placeholder="e.g. Prefer AC at 24°C instead of 22°C",
            )
            if st.button("Save Feedback", key=f"t2_modify_save_{habit_id}"):
                _handle_feedback(username, habit_id, scene, "modify", note=note)
                st.session_state[f"t2_modify_open_{habit_id}"] = False


# ═══════════════════════════════════════════════════
# Pipeline 执行
# ═══════════════════════════════════════════════════

def _run_recommendation(username: str, scenario_name: str, scenario: dict, intention: str):
    """执行推荐: 构造 StructuredContext → ProactiveExecutor.recommend"""
    try:
        from panoramix_core.models.fact import StructuredContext
        from engine.habit_engine import HabitDemoEngine

        hour = scenario["time"].hour
        current_ctx = StructuredContext(
            time_bucket=scenario["time_bucket"],
            hour=hour,
            weekday=scenario["weekday"],
            vehicle_state=scenario["vehicle_state"],
            geofence=scenario.get("geofence"),
        )

        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        result = engine.get_recommendation(current_ctx, top_k=5)
        engine.close()

        # 缓存结果 (按用户隔离)
        st.session_state[f"t2_recommendations_{username}"] = {
            "actions": [
                {
                    "habit_id": a.habit_id,
                    "habit_text": a.habit_text,
                    "scene_name": a.scene_name,
                    "match_confidence": a.match_confidence,
                    "combined_confidence": a.combined_confidence,
                    "context_distance": a.context_distance,
                    "parsed_actions": a.parsed_actions,
                }
                for a in result.actions
            ],
            "has_recommendations": result.has_recommendations,
            "candidates_checked": result.candidates_checked,
            "threshold": result.threshold,
            "total_habits": result.total_habits,
            "username": username,
            "scenario_name": scenario_name,
            "intention": intention,
        }

        # 更新匹配场景 → Mini KG 高亮联动 (按用户隔离)
        if result.actions:
            st.session_state[f"t2_matched_scene_{username}"] = result.actions[0].scene_name
        else:
            st.session_state[f"t2_matched_scene_{username}"] = None

    except Exception as e:
        st.error(f"Recommendation error: {e}")
        log.exception("Tab 2 recommendation error")


# ═══════════════════════════════════════════════════
# 反馈处理
# ═══════════════════════════════════════════════════

def _handle_feedback(username: str, habit_id: str, scene: str,
                     action: str, note: str = ""):
    """统一处理 Apply / Dismiss / Modify 反馈"""
    try:
        from engine.habit_engine import HabitDemoEngine

        engine = HabitDemoEngine(username=username.lower(), llm_client=None)

        if action == "accept":
            ok = engine.accept_habit(habit_id)
            engine.close()
            if ok:
                st.success(f"Habit applied! The system will proactively execute this pattern.")
            else:
                st.warning("Habit not found.")

        elif action == "dismiss":
            ok = engine.reject_habit(habit_id)
            engine.close()
            if ok:
                st.info("Habit dismissed and removed from knowledge graph.")
            else:
                st.warning("Habit not found.")

        elif action == "modify":
            # Modify: 接受但记录调整备注 (用于后续 retraining)
            ok = engine.accept_habit(habit_id)
            engine.close()
            if ok:
                st.success(f"Feedback saved. Habit accepted with modification note.")
            else:
                st.warning("Habit not found.")

        # 记录反馈日志 (按用户隔离)
        st.session_state[f"t2_feedback_log_{username}"].append({
            "habit_id": habit_id,
            "scene": scene,
            "action": action,
            "note": note,
            "time": datetime.now().strftime("%H:%M:%S"),
        })

    except Exception as e:
        st.error(f"Feedback error: {e}")
        log.exception("Tab 2 feedback error")


# ═══════════════════════════════════════════════════
# 数据不足提示
# ═══════════════════════════════════════════════════

def _render_insufficient_data(data: dict):
    """无推荐时 — 显示数据不足提示 + 引导"""
    candidates = data["candidates_checked"]
    threshold = data["threshold"]
    total = data["total_habits"]
    scenario = data.get("scenario_name", "")

    if total == 0:
        st.html(
            '<div style="background:#1A2A42;border:1px solid #1E3A5F;border-radius:8px;'
            'padding:24px;text-align:center;font-family:sans-serif;">'
            '<div style="font-size:1.5rem;margin-bottom:8px;">📊</div>'
            '<div style="font-size:0.95rem;color:#E8EDF3;font-weight:600;margin-bottom:8px;">'
            'No Habits Learned Yet</div>'
            '<div style="font-size:0.8rem;color:#8899AA;line-height:1.6;">'
            'The model needs training data to make recommendations.<br>'
            'Go to <b>Tab 1</b> and run "Load Mockup Data" or manually add<br>'
            'behavior patterns to start building the knowledge graph.</div>'
            '</div>'
        )
    elif candidates == 0:
        st.html(
            '<div style="background:#1A2A42;border:1px solid #1E3A5F;border-radius:8px;'
            'padding:24px;text-align:center;font-family:sans-serif;">'
            '<div style="font-size:1.5rem;margin-bottom:8px;">⏳</div>'
            '<div style="font-size:0.95rem;color:#E8EDF3;font-weight:600;margin-bottom:8px;">'
            'No Accepted Habits</div>'
            '<div style="font-size:0.8rem;color:#8899AA;line-height:1.6;">'
            f'{total} habit(s) detected but none accepted yet.<br>'
            'Accept habits in the Knowledge Graph to enable recommendations.</div>'
            '</div>'
        )
    else:
        st.html(
            f'<div style="background:#1A2A42;border:1px solid #1E3A5F;border-radius:8px;'
            f'padding:24px;text-align:center;font-family:sans-serif;">'
            f'<div style="font-size:1.5rem;margin-bottom:8px;">🔍</div>'
            f'<div style="font-size:0.95rem;color:#E8EDF3;font-weight:600;margin-bottom:8px;">'
            f'No Match for "{scenario}"</div>'
            f'<div style="font-size:0.8rem;color:#8899AA;line-height:1.6;">'
            f'Checked {candidates} accepted habit(s), none within<br>'
            f'context distance threshold ({threshold:.2f}).<br>'
            f'Try a different scenario or add more training data in Tab 1.</div>'
            f'</div>'
        )
