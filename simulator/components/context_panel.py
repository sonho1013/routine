"""
Trigger Input Panel — 触发条件输入面板

对齐 mockup 数据中的 trigger 维度：
  - 时空: Date / Time / Weekday / Location / Trip Role
  - 环境: Weather / Outside Temp
  - 车辆状态: Speed / Gear / Wiper / Door Lock / Window / Volume / Approach Unlock

Tab 1: 可编辑触发条件输入
Tab 2: 只读展示 + 场景预设
"""
import streamlit as st
from datetime import datetime, time

from data.unified_loader import PLACE_COORDS


# ── 地点选项 (对齐 unified 数据 place_id) ──
LOCATION_OPTIONS = ["home", "work", "office_gate_01", "park", "mall"]
LOCATION_LABELS = {
    "home": "Home",
    "work": "Work",
    "office_gate_01": "Office Park Gate",
    "park": "Park",
    "mall": "Mall",
}

TRIP_ROLE_OPTIONS = ["origin", "destination", "waypoint"]
WEATHER_OPTIONS = ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
GEAR_OPTIONS = ["P", "D", "R", "N"]
WIPER_OPTIONS = ["off", "low", "medium", "high"]
DOOR_LOCK_OPTIONS = ["locked", "unlocked"]
WINDOW_STATE_OPTIONS = ["Closed", "Half Open", "Open"]
APPROACH_UNLOCK_OPTIONS = ["enabled", "disabled"]


# ── 场景预设 — 对齐 mockup 典型场景 ──
SCENE_PRESETS = {
    "Morning Commute": {
        "date": datetime(2026, 3, 30).date(),
        "time": time(8, 43),
        "location": "home",
        "trip_role": "origin",
        "weather": "Sunny",
        "outside_temp": 20.0,
        "speed_kph": 0.0,
        "gear": "P",
        "wiper": "off",
        "door_lock": "unlocked",
        "window_state": "Closed",
        "current_volume": 35,
        "approach_unlock": "enabled",
    },
    "Office Gate Entry": {
        "date": datetime(2026, 3, 30).date(),
        "time": time(9, 6),
        "location": "office_gate_01",
        "trip_role": "waypoint",
        "weather": "Sunny",
        "outside_temp": 20.0,
        "speed_kph": 3.0,
        "gear": "D",
        "wiper": "off",
        "door_lock": "locked",
        "window_state": "Closed",
        "current_volume": 35,
        "approach_unlock": "enabled",
    },
    "Arriving Home": {
        "date": datetime(2026, 3, 30).date(),
        "time": time(18, 37),
        "location": "home",
        "trip_role": "destination",
        "weather": "Sunny",
        "outside_temp": 18.0,
        "speed_kph": 0.0,
        "gear": "P",
        "wiper": "off",
        "door_lock": "locked",
        "window_state": "Closed",
        "current_volume": 35,
        "approach_unlock": "enabled",
    },
    "Custom": {
        "date": datetime.today().date(),
        "time": time(12, 0),
        "location": "home",
        "trip_role": "origin",
        "weather": "Sunny",
        "outside_temp": 22.0,
        "speed_kph": 0.0,
        "gear": "P",
        "wiper": "off",
        "door_lock": "unlocked",
        "window_state": "Closed",
        "current_volume": 35,
        "approach_unlock": "enabled",
    },
}


def render_context_panel(readonly: bool = False, key_prefix: str = "ctx") -> dict:
    """
    Render trigger input panel.

    Returns:
        dict with trigger fields including signals for the pipeline
    """
    st.markdown("**Trigger**")

    # Scene preset selector
    preset_name = st.selectbox(
        "Scene Preset",
        list(SCENE_PRESETS.keys()),
        key=f"{key_prefix}_preset",
    )
    preset = SCENE_PRESETS[preset_name]

    # ── Row 1: Date / Time ──
    c1, c2 = st.columns(2)
    with c1:
        date_val = st.date_input(
            "Date",
            value=preset["date"],
            key=f"{key_prefix}_date",
            disabled=readonly,
        )
    with c2:
        time_val = st.time_input(
            "Time",
            value=preset["time"],
            key=f"{key_prefix}_time",
            disabled=readonly,
        )

    # ── Row 2: Location / Trip Role ──
    c3, c4 = st.columns(2)
    with c3:
        loc_idx = LOCATION_OPTIONS.index(preset["location"]) if preset["location"] in LOCATION_OPTIONS else 0
        location = st.selectbox(
            "Location",
            LOCATION_OPTIONS,
            index=loc_idx,
            format_func=lambda x: LOCATION_LABELS.get(x, x),
            key=f"{key_prefix}_location",
            disabled=readonly,
        )
    with c4:
        role_idx = TRIP_ROLE_OPTIONS.index(preset["trip_role"]) if preset["trip_role"] in TRIP_ROLE_OPTIONS else 0
        trip_role = st.selectbox(
            "Trip Role",
            TRIP_ROLE_OPTIONS,
            index=role_idx,
            key=f"{key_prefix}_trip_role",
            disabled=readonly,
        )

    # ── Row 3: Weather / Outside Temp ──
    c5, c6 = st.columns(2)
    with c5:
        weather = st.selectbox(
            "Weather",
            WEATHER_OPTIONS,
            index=WEATHER_OPTIONS.index(preset["weather"]),
            key=f"{key_prefix}_weather",
            disabled=readonly,
        )
    with c6:
        outside_temp = st.number_input(
            "Outside Temp (°C)",
            min_value=-20.0, max_value=50.0,
            value=preset["outside_temp"],
            step=0.5,
            key=f"{key_prefix}_temp",
            disabled=readonly,
        )

    # ── Row 4: Speed / Gear ──
    c7, c8 = st.columns(2)
    with c7:
        speed_kph = st.number_input(
            "Vehicle Speed (km/h)",
            min_value=0.0, max_value=200.0,
            value=preset["speed_kph"],
            step=1.0,
            key=f"{key_prefix}_speed",
            disabled=readonly,
        )
    with c8:
        gear_idx = GEAR_OPTIONS.index(preset["gear"]) if preset["gear"] in GEAR_OPTIONS else 0
        gear = st.selectbox(
            "Gear",
            GEAR_OPTIONS,
            index=gear_idx,
            key=f"{key_prefix}_gear",
            disabled=readonly,
        )

    # ── Row 5: Wiper / Door Lock ──
    c9, c10 = st.columns(2)
    with c9:
        wiper_idx = WIPER_OPTIONS.index(preset["wiper"]) if preset["wiper"] in WIPER_OPTIONS else 0
        wiper = st.selectbox(
            "Wiper Speed",
            WIPER_OPTIONS,
            index=wiper_idx,
            key=f"{key_prefix}_wiper",
            disabled=readonly,
        )
    with c10:
        door_idx = DOOR_LOCK_OPTIONS.index(preset["door_lock"]) if preset["door_lock"] in DOOR_LOCK_OPTIONS else 0
        door_lock = st.selectbox(
            "Door Lock",
            DOOR_LOCK_OPTIONS,
            index=door_idx,
            key=f"{key_prefix}_door_lock",
            disabled=readonly,
        )

    # ── Row 6: Window State / Volume / Approach Unlock ──
    c11, c12, c13 = st.columns(3)
    with c11:
        win_idx = WINDOW_STATE_OPTIONS.index(preset["window_state"]) if preset["window_state"] in WINDOW_STATE_OPTIONS else 0
        window_state = st.selectbox(
            "Window Position",
            WINDOW_STATE_OPTIONS,
            index=win_idx,
            key=f"{key_prefix}_window",
            disabled=readonly,
        )
    with c12:
        current_volume = st.number_input(
            "Volume (%)",
            min_value=0, max_value=100,
            value=preset["current_volume"],
            step=5,
            key=f"{key_prefix}_volume",
            disabled=readonly,
        )
    with c13:
        au_idx = APPROACH_UNLOCK_OPTIONS.index(preset["approach_unlock"]) if preset["approach_unlock"] in APPROACH_UNLOCK_OPTIONS else 0
        approach_unlock = st.selectbox(
            "Approach Unlock",
            APPROACH_UNLOCK_OPTIONS,
            index=au_idx,
            key=f"{key_prefix}_approach_unlock",
            disabled=readonly,
        )

    # Resolve GPS from location
    gps = PLACE_COORDS.get(location)
    lat = gps[0] if gps else None
    lng = gps[1] if gps else None

    return {
        "date": str(date_val),
        "time": str(time_val),
        "location": location,
        "trip_role": trip_role,
        "weather": weather,
        "outside_temp": outside_temp,
        "speed_kph": speed_kph,
        "gear": gear,
        "wiper": wiper,
        "door_lock": door_lock,
        "window_state": window_state,
        "current_volume": current_volume,
        "approach_unlock": approach_unlock,
        "gps_latitude": lat,
        "gps_longitude": lng,
        "preset": preset_name,
    }
