"""
Action Panel — 用户动作输入面板

对齐 mockup 数据中的 action 列表 (trigger list xlsx):
  - HVAC: HVAC Start / HVAC Temperature Set
  - Navigation: Map Start / Map Destination Set / Map Route Set
  - Media: Volume Set
  - Vehicle: Window Set / Approach Unlock Set

每个 action 可勾选启用并设置参数值，模拟一次驾驶事件中的用户操作。
"""
import streamlit as st
from typing import Dict, List, Tuple


# ── Action 定义: (signal_name, display_label, default_value, value_type, options) ──
# signal_name: 内部信号名 (对齐 rules.yaml / unified_loader ACTION_TO_SIGNALS)
# value_type: "number", "select", "toggle", "text"

ACTION_LIST: List[Tuple] = [
    # HVAC
    ("hvac_power", "HVAC Start", "on", "toggle", None),
    ("hvac_temp_target", "HVAC Temperature Set (°C)", 22, "number", {"min": 16, "max": 30, "step": 1}),

    # Navigation (Google Map)
    ("nav_destination", "Map Destination Set", "work", "text", None),
    ("nav_route_pref", "Map Route Preference", "fastest", "select", ["fastest", "shortest", "eco"]),

    # Media
    ("media_volume", "Volume Set (%)", 35, "number", {"min": 0, "max": 100, "step": 5}),

    # Vehicle
    ("window_position", "Window Set", 0, "select_window", ["Closed", "Half Open", "Open"]),
    ("keyless_entry", "Approach Unlock Set", "enabled", "select", ["enabled", "disabled"]),
]

# Window state → percentage mapping
WINDOW_STATE_TO_PCT = {
    "Closed": 0,
    "Half Open": 50,
    "Open": 100,
}


def render_behavior_table(key_prefix: str = "beh") -> Dict[str, any]:
    """
    Render action panel with checkboxes and value inputs.

    Returns:
        dict: {signal_name: value, ...} for enabled actions only
    """
    st.markdown("**Action**")

    selected_signals = {}

    for sig_name, label, default, vtype, opts in ACTION_LIST:
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
                    " ",
                    value=str(default),
                    disabled=True,
                    key=f"{key_prefix}_{sig_name}_val_disabled",
                    label_visibility="collapsed",
                )
            elif vtype == "toggle":
                selected_signals[sig_name] = default
                st.text_input(
                    " ",
                    value=str(default),
                    disabled=True,
                    key=f"{key_prefix}_{sig_name}_val",
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
            elif vtype == "select_window":
                val = st.selectbox(
                    label,
                    opts,
                    index=0,
                    key=f"{key_prefix}_{sig_name}_val",
                    label_visibility="collapsed",
                )
                selected_signals[sig_name] = WINDOW_STATE_TO_PCT[val]
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
    """Generate a short summary of selected action signals."""
    if not signals:
        return "No actions selected"
    parts = [f"{k}={v}" for k, v in signals.items()]
    return ", ".join(parts)
