"""
Knowledge Graph Panel — 习惯知识图谱展示

对齐 cockpit_ai_simulator 参考设计:
  - KG 表格: Scene / Context / Actions / Control Params / Confidence
  - 置信度色标: High(green ≥0.85) / Mid(teal ≥0.70) / Low(grey <0.70)
  - 新行高亮动画 (new-row class)
  - 摘要卡片: Top Scenario / Avg Confidence / Data Coverage / Total Facts
"""
import json
import streamlit as st
from typing import Dict, List, Optional

from engine.proactive_executor import parse_habit_actions


def _conf_badge(conf: float) -> str:
    """生成置信度 badge HTML"""
    if conf is None:
        conf = 0.0
    if conf >= 0.85:
        color, bg = "#00C896", "rgba(0,200,150,0.15)"
    elif conf >= 0.70:
        color, bg = "#00BFC8", "rgba(0,191,200,0.15)"
    else:
        color, bg = "#8899AA", "rgba(136,153,170,0.12)"
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:4px;'
        f'font-size:0.8rem;font-weight:600;color:{color};background:{bg};">'
        f'{conf:.2f}</span>'
    )


def _format_context(habit_dict: dict) -> str:
    """将 habit context 格式化为简洁描述"""
    ctx = habit_dict.get("context", {})
    parts = []
    tb = ctx.get("time_bucket", "")
    if tb:
        parts.append(tb.replace("_", " ").title())
    vs = ctx.get("vehicle_state", "")
    if vs:
        parts.append(vs.replace("_", " ").title())
    geo = ctx.get("geofence")
    if geo:
        parts.append(geo.title())
    wd = ctx.get("weekday")
    if wd is True:
        parts.append("Weekday")
    elif wd is False:
        parts.append("Weekend")
    return ", ".join(parts) if parts else "—"


def _format_actions(habit_text: str) -> str:
    """从 habit text 解析控制参数并格式化"""
    actions = parse_habit_actions(habit_text)
    if not actions:
        return "—"
    params = []
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
        "keyless_entry": "Keyless",
        "engine_status": "Engine",
    }
    for k, v in actions.items():
        label = _LABELS.get(k, k)
        if k == "hvac_temp_target":
            params.append(f"{label}={v}°C")
        elif k == "media_volume":
            params.append(f"{label}={v}%")
        elif k == "window_position":
            params.append(f"{label}={v}%")
        elif k == "media_off":
            params.append("Media=OFF")
        elif k == "hvac_power":
            params.append(f"{label}={v}")
        else:
            params.append(f"{label}={v}")
    return ", ".join(params)


def render_knowledge_graph(
    habits: List[dict],
    new_habit_ids: Optional[set] = None,
    username: str = "",
):
    """
    渲染 KG 表格 + 摘要卡片。

    Args:
        habits: habit dict 列表 (from engine.get_status()["habits"])
        new_habit_ids: 本轮新检测到的 habit IDs (用于高亮)
        username: 当前用户名
    """
    if new_habit_ids is None:
        new_habit_ids = set()

    if not habits:
        st.info("No habits detected yet. Run 'Analysis to model' to start learning.")
        return

    # ── 按置信度排序 ──
    sorted_habits = sorted(
        habits,
        key=lambda h: h.get("clustering_confidence") or 0,
        reverse=True,
    )

    # ── 构建 HTML 表格（使用 st.html 确保渲染） ──
    rows_html = []
    for h in sorted_habits:
        hid = h.get("id", "")
        is_new = hid in new_habit_ids
        row_style = 'background:#1a3a2a;border-left:3px solid #00C896;' if is_new else ""

        scene = h.get("scene_name") or "—"
        context_str = _format_context(h)
        text = h.get("text", "")
        params_str = _format_actions(text)
        conf = h.get("clustering_confidence") or 0.0
        accepted = h.get("accepted", False)
        status_icon = "✅" if accepted else "⏳"

        new_badge = (
            ' <span style="background:#F5A623;color:#fff;padding:1px 5px;'
            'border-radius:3px;font-size:0.6rem;font-weight:700;">NEW</span>'
            if is_new else ""
        )

        rows_html.append(
            f'<tr style="{row_style}">'
            f'<td style="padding:6px 4px;font-size:0.75rem;color:#8899AA;">{status_icon}</td>'
            f'<td style="padding:6px 4px;">'
            f'<div style="font-weight:600;color:#E8EDF3;">{scene}{new_badge}</div>'
            f'<div style="font-size:0.7rem;color:#8899AA;margin-top:2px;">{text[:60]}</div></td>'
            f'<td style="padding:6px 4px;font-size:0.8rem;color:#C0CCDA;">{context_str}</td>'
            f'<td style="padding:6px 4px;font-size:0.8rem;"><code style="font-size:0.75rem;color:#C0CCDA;">{params_str}</code></td>'
            f'<td style="padding:6px 4px;">{_conf_badge(conf)}</td>'
            f'</tr>'
        )

    full_html = (
        '<div style="font-family:sans-serif;color:#E8EDF3;">'
        # 表头 + 图例
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'margin-bottom:8px;padding:4px 0;">'
        '<span style="font-weight:600;font-size:0.9rem;">User Knowledge Graph</span>'
        '<span style="font-size:0.7rem;color:#8899AA;">'
        '<span style="color:#00C896;">●</span> High ≥0.85 &nbsp;'
        '<span style="color:#00BFC8;">●</span> Mid 0.70+ &nbsp;'
        '<span style="color:#8899AA;">●</span> Low &lt;0.70'
        '</span></div>'
        # 表格
        '<table style="width:100%;border-collapse:collapse;font-size:0.85rem;">'
        '<thead><tr style="border-bottom:2px solid #1E3A5F;text-align:left;">'
        '<th style="padding:8px 4px;color:#00BFC8;font-size:0.75rem;width:30px;"></th>'
        '<th style="padding:8px 4px;color:#00BFC8;font-size:0.75rem;">Scene / Pattern</th>'
        '<th style="padding:8px 4px;color:#00BFC8;font-size:0.75rem;">Context</th>'
        '<th style="padding:8px 4px;color:#00BFC8;font-size:0.75rem;">Control Params</th>'
        '<th style="padding:8px 4px;color:#00BFC8;font-size:0.75rem;width:60px;">Conf.</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table></div>'
    )
    st.html(full_html)

    # ── 摘要卡片 ──
    _render_summary_cards(sorted_habits)


def _render_summary_cards(habits: List[dict]):
    """渲染底部摘要卡片行"""
    if not habits:
        return

    confidences = [h.get("clustering_confidence") or 0 for h in habits]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    top_scene = habits[0].get("scene_name") or "—" if habits else "—"
    accepted_count = sum(1 for h in habits if h.get("accepted"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Top Scene", top_scene)
    with c2:
        st.metric("Avg Confidence", f"{avg_conf:.2f}")
    with c3:
        st.metric("Patterns Found", len(habits))
    with c4:
        st.metric("Accepted", accepted_count)
