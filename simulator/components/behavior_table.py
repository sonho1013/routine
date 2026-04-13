"""
Behavior Data Table — 5 大类行为信号表

用户可勾选信号并设置参数值，模拟一次驾驶事件中的用户行为。
类别: Comfort / Media / Driving Assist / Convenience / Vehicle Settings
"""
import streamlit as st
from typing import Dict, List, Tuple


# ── 信号定义: (signal_name, display_label, default_value, value_type, options) ──
# value_type: "number", "select", "toggle"

BEHAVIOR_CATEGORIES: Dict[str, List[Tuple]] = {
    "Comfort": [
        ("hvac_temp_target", "AC Temperature (°C)", 22, "number", {"min": 16, "max": 30, "step": 1}),
        ("hvac_power", "AC Power", "on", "select", ["on", "off"]),
        ("hvac_fan_speed", "Fan Speed", "auto", "select", ["auto", "low", "medium", "high"]),
        ("seat_heating", "Seat Heating Level", 2, "number", {"min": 0, "max": 3, "step": 1}),
    ],
    "Media": [
        ("media_source", "Media Source", "podcast", "select", ["podcast", "music", "radio", "off"]),
        ("media_content_id", "Content ID", "Tech Daily", "text", None),
        ("media_volume", "Volume (%)", 60, "number", {"min": 0, "max": 100, "step": 5}),
    ],
    "Driving Assist": [
        ("drive_mode", "Drive Mode", "eco", "select", ["eco", "normal", "sport", "comfort"]),
        ("acc_distance", "ACC Distance", "medium", "select", ["short", "medium", "long"]),
    ],
    "Convenience": [
        ("nav_destination", "Navigation To", "work", "text", None),
        ("nav_route_pref", "Route Preference", "fastest", "select", ["fastest", "shortest", "eco"]),
        ("keyless_entry", "Keyless Entry", "enabled", "select", ["enabled", "disabled"]),
    ],
    "Vehicle Settings": [
        ("window_position", "Window Position (%)", 0, "number", {"min": 0, "max": 100, "step": 5}),
        ("engine_status", "Engine", "on", "select", ["on", "off"]),
    ],
}

# 类别颜色标记 (CSS class → emoji fallback for Streamlit)
CATEGORY_ICONS = {
    "Comfort": "🌡️",
    "Media": "🎵",
    "Driving Assist": "🚗",
    "Convenience": "📍",
    "Vehicle Settings": "⚙️",
}


def render_behavior_table(key_prefix: str = "beh") -> Dict[str, any]:
    """
    渲染行为数据表格（5 大类，每类可勾选启用信号并设置值）。

    Returns:
        dict: {signal_name: value, ...} 仅包含被勾选启用的信号
    """
    st.markdown("**Behavior Signals**")

    selected_signals = {}

    for cat_name, signals in BEHAVIOR_CATEGORIES.items():
        icon = CATEGORY_ICONS.get(cat_name, "")
        with st.expander(f"{icon} {cat_name}", expanded=True):
            for sig_name, label, default, vtype, opts in signals:
                # Three-column row: tiny checkbox, dedicated label column
                # (ellipsis-clipped so it can never spill into the value
                # widget), value widget. Old [1, 3] layout put the full
                # label inside the narrow checkbox column, which wrapped
                # and overlapped the next row on common screen widths.
                col_check, col_label, col_value = st.columns([1, 5, 6])

                with col_check:
                    enabled = st.checkbox(
                        label,
                        value=False,
                        key=f"{key_prefix}_{sig_name}_en",
                        label_visibility="collapsed",
                    )

                with col_label:
                    label_color = "#E8EDF3" if enabled else "#8899AA"
                    st.markdown(
                        f"<div title='{label}' style='"
                        f"padding-top:6px;"
                        f"font-size:0.82rem;"
                        f"color:{label_color};"
                        f"white-space:nowrap;"
                        f"overflow:hidden;"
                        f"text-overflow:ellipsis;"
                        f"'>{label}</div>",
                        unsafe_allow_html=True,
                    )

                with col_value:
                    if not enabled:
                        st.text_input(
                            " ",  # placeholder
                            value=str(default),
                            disabled=True,
                            key=f"{key_prefix}_{sig_name}_val_disabled",
                            label_visibility="collapsed",
                        )
                    elif vtype == "number":
                        val = st.number_input(
                            label,
                            min_value=opts["min"],
                            max_value=opts["max"],
                            value=default,
                            step=opts["step"],
                            key=f"{key_prefix}_{sig_name}_val",
                            label_visibility="collapsed",
                        )
                        selected_signals[sig_name] = val
                    elif vtype == "select":
                        val = st.selectbox(
                            label,
                            opts,
                            index=opts.index(default) if default in opts else 0,
                            key=f"{key_prefix}_{sig_name}_val",
                            label_visibility="collapsed",
                        )
                        selected_signals[sig_name] = val
                    elif vtype == "text":
                        val = st.text_input(
                            label,
                            value=str(default),
                            key=f"{key_prefix}_{sig_name}_val",
                            label_visibility="collapsed",
                        )
                        selected_signals[sig_name] = val

    return selected_signals


def signals_summary(signals: Dict) -> str:
    """生成已选信号的简短摘要"""
    if not signals:
        return "No signals selected"
    parts = [f"{k}={v}" for k, v in signals.items()]
    return ", ".join(parts)
