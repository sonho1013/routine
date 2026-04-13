"""
Context Input Panel — 上下文输入面板

Tab 1: 可编辑的上下文输入 (6 字段 3 列布局)
Tab 2: 只读上下文展示 + 可选场景预设
"""
import streamlit as st
from datetime import datetime, time


# 场景预设 — 快速填充上下文
SCENE_PRESETS = {
    "Morning Commute": {
        "date": datetime(2025, 10, 6).date(),
        "time": time(8, 0),
        "weather": "Cloudy",
        "outside_temp": 8.0,
        "battery_soc": 85,
        "passengers": 1,
    },
    "Arriving Home": {
        "date": datetime(2025, 10, 6).date(),
        "time": time(18, 30),
        "weather": "Sunny",
        "outside_temp": 15.0,
        "battery_soc": 45,
        "passengers": 1,
    },
    "Toll / Parking Entry": {
        "date": datetime(2025, 10, 7).date(),
        "time": time(9, 15),
        "weather": "Sunny",
        "outside_temp": 12.0,
        "battery_soc": 72,
        "passengers": 2,
    },
    "Custom": {
        "date": datetime.today().date(),
        "time": time(12, 0),
        "weather": "Sunny",
        "outside_temp": 20.0,
        "battery_soc": 90,
        "passengers": 1,
    },
}

WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]


def render_context_panel(readonly: bool = False, key_prefix: str = "ctx") -> dict:
    """
    渲染上下文输入面板。

    Args:
        readonly: True 时不可编辑 (Tab 2 使用)
        key_prefix: Streamlit widget key 前缀

    Returns:
        dict: 当前上下文数据
    """
    st.markdown("**Context Data**")

    # 场景预设选择
    preset_name = st.selectbox(
        "Scene Preset",
        list(SCENE_PRESETS.keys()),
        key=f"{key_prefix}_preset",
    )
    preset = SCENE_PRESETS[preset_name]

    # 3 列布局，6 个字段
    col1, col2, col3 = st.columns(3)

    with col1:
        date_val = st.date_input(
            "Date",
            value=preset["date"],
            key=f"{key_prefix}_date",
            disabled=readonly,
        )
        outside_temp = st.number_input(
            "Outside Temp (°C)",
            min_value=-20.0, max_value=50.0,
            value=preset["outside_temp"],
            step=0.5,
            key=f"{key_prefix}_temp",
            disabled=readonly,
        )

    with col2:
        time_val = st.time_input(
            "Time",
            value=preset["time"],
            key=f"{key_prefix}_time",
            disabled=readonly,
        )
        battery_soc = st.number_input(
            "Battery SOC (%)",
            min_value=0, max_value=100,
            value=preset["battery_soc"],
            key=f"{key_prefix}_battery",
            disabled=readonly,
        )

    with col3:
        weather = st.selectbox(
            "Weather",
            WEATHER_OPTIONS,
            index=WEATHER_OPTIONS.index(preset["weather"]),
            key=f"{key_prefix}_weather",
            disabled=readonly,
        )
        passengers = st.number_input(
            "Passengers",
            min_value=0, max_value=7,
            value=preset["passengers"],
            key=f"{key_prefix}_passengers",
            disabled=readonly,
        )

    return {
        "date": str(date_val),
        "time": str(time_val),
        "weather": weather,
        "outside_temp": outside_temp,
        "battery_soc": battery_soc,
        "passengers": passengers,
        "preset": preset_name,
    }
