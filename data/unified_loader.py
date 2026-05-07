"""
Unified Data 抽象层 — 将四周 unified JSON 数据转为系统内部事件格式

统一数据源:
  data/four_week_week_case_01/week_XX/unified_week_case_01_wXX_<weekday>.json

每个 unified 文件包含:
  - header: 日期、天气、车辆/司机信息
  - motion_plan: trips（含 location_profile、speed_profile 等）
  - raw_events: 车辆事件序列（user_action / vehicle_status）

本模块职责:
  1. 加载指定周数/天数的 unified 文件
  2. 将 raw_events 中的 action 转换为系统内部信号格式:
       {"t": "2026-03-30T08:43:00", "signal": "hvac_temp_target", "value": 22}
  3. 将 vehicle_status 事件转换为上下文信号（GPS、车速、挡位等）
  4. 输出与 SignalSimulator / HabitDemoEngine.ingest_signal_batch 兼容的数据
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── 数据根目录 ──
DATA_ROOT = Path(__file__).parent / "four_week_week_case_01"

WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# ── 地点坐标映射 (从 unified 数据的 location_profile 提取) ──
PLACE_COORDS = {
    "home":           (31.2304, 121.4737),
    "work":           (31.2396, 121.4997),
    "office_gate_01": (31.2388, 121.4975),
    "park":           (31.2270, 121.4610),
    "mall":           (31.2350, 121.4900),
}

# ── unified action → 内部信号的映射 ──
# 每个映射函数接收 (action_name, payload, event) → List[(signal_name, value)]
ACTION_TO_SIGNALS = {
    "hvac_temperature_set": lambda p, e: [
        ("hvac_temp_target", p.get("target_celsius", 22)),
    ],
    "hvac_start": lambda p, e: [
        ("hvac_power", "on"),
    ],
    "map_app_start": lambda p, e: [
        # map 启动本身不产生导航信号，route_set 才产生
    ],
    "route_set_origin_destination": lambda p, e: [
        ("nav_destination", _normalize_destination(p.get("destination", ""))),
    ],
    "google_map_route_set": lambda p, e: [
        ("nav_destination", _normalize_destination(p.get("destination", ""))),
        ("nav_route_pref", "fastest"),
    ],
    "media_volume_set": lambda p, e: [
        ("media_volume", p.get("target_volume_percent", 0)),
    ],
    "window_open_request": lambda p, e: [
        ("window_position", _normalize_window_position(p.get("window_position", "CLOSED"))),
    ],
    "approach_unlock_set": lambda p, e: [
        ("keyless_entry", "enabled" if p.get("enabled", True) else "disabled"),
    ],
    "vehicle_power_on": lambda p, e: [
        ("engine_status", "on"),
    ],
}


def _normalize_destination(dest: str) -> str:
    """将 unified 目的地名映射为系统内部的标准名"""
    mapping = {
        "office_park": "work",
        "current_position": "current",
        "home": "home",
        "park": "park",
        "mall": "mall",
    }
    return mapping.get(dest, dest)


def _normalize_window_position(pos) -> int:
    """将 unified 的窗户位置转为百分比数值"""
    if isinstance(pos, (int, float)):
        return int(pos)
    pos_str = str(pos).upper()
    if pos_str == "OPEN":
        return 100
    if pos_str == "HALF":
        return 50
    if pos_str == "CLOSED":
        return 0
    return 0


def _extract_context_signals(event: dict, header_date: str) -> List[dict]:
    """
    从 vehicle_status 事件提取上下文信号。
    这些信号不对应用户动作，但为 StructuredContext 提供维度（对齐 trigger list 2.xlsx）。
    """
    signals = []
    ts_str = event.get("timestamp", "")
    full_ts = f"{header_date}T{ts_str}" if "T" not in ts_str else ts_str

    vs = event.get("vehicle_state", {})

    # 车速
    if "velocity_kph" in vs:
        signals.append({
            "t": full_ts, "signal": "vehicle_speed", "value": vs["velocity_kph"]
        })

    # 挡位
    if "gear" in vs:
        signals.append({
            "t": full_ts, "signal": "gear_position", "value": vs["gear"]
        })

    # 雨刮档位 → 作为前置条件维度
    if "wiper_speed_level" in vs:
        wiper_level = vs["wiper_speed_level"]
        wiper_map = {0: "off", 1: "low", 2: "medium", 3: "high", 4: "max"}
        signals.append({
            "t": full_ts, "signal": "wiper_state",
            "value": wiper_map.get(wiper_level, "off"),
        })

    # 车外温度（摄氏度）
    if "out_temp_c" in event:
        signals.append({
            "t": full_ts, "signal": "outside_temp_c", "value": event["out_temp_c"],
        })

    # 车窗当前位置（快照，独立于 window_set 动作）
    if "window_driver_position" in vs:
        signals.append({
            "t": full_ts, "signal": "window_state_snapshot",
            "value": vs["window_driver_position"],
        })

    # 靠近解锁是否开启（快照，独立于 approach_unlock_set 动作）
    if "approach_unlock_enabled" in vs:
        signals.append({
            "t": full_ts, "signal": "approach_unlock_state",
            "value": vs["approach_unlock_enabled"],
        })

    # 门锁状态（raw_events 的 door_lock signal 可能不在 vehicle_state 内，
    # 这里只抓 vehicle_state 里带的；lifecycle_plan 的离散门锁事件暂不追踪）
    if "door_lock" in vs:
        signals.append({
            "t": full_ts, "signal": "door_lock_state", "value": vs["door_lock"],
        })

    # POI 类型（site_entrance / home / work / ...）
    ps = event.get("place_semantic", {})
    if ps.get("place_type"):
        signals.append({
            "t": full_ts, "signal": "poi_type", "value": ps["place_type"],
        })

    # GPS: 从 place_semantic 查找坐标
    place_id = ps.get("place_id")
    if place_id and place_id in PLACE_COORDS:
        lat, lng = PLACE_COORDS[place_id]
        signals.append({"t": full_ts, "signal": "gps_latitude", "value": lat})
        signals.append({"t": full_ts, "signal": "gps_longitude", "value": lng})

    return signals


def _convert_event_to_signals(event: dict, header_date: str) -> List[dict]:
    """
    将单条 unified raw_event 转为内部信号列表。

    - user_action 且有 action → 转为行为信号
    - vehicle_status → 转为上下文信号
    - 所有事件的 vehicle_state 提供辅助上下文
    """
    signals = []
    ts_str = event.get("timestamp", "")
    full_ts = f"{header_date}T{ts_str}" if "T" not in ts_str else ts_str

    action = event.get("action")
    if action and action.get("name"):
        action_name = action["name"]
        payload = action.get("payload", {})
        mapper = ACTION_TO_SIGNALS.get(action_name)
        if mapper:
            for sig_name, sig_value in mapper(payload, event):
                signals.append({
                    "t": full_ts,
                    "signal": sig_name,
                    "value": sig_value,
                })
        else:
            log.debug(f"Unmapped action: {action_name}")

    # 附加上下文信号
    ctx_signals = _extract_context_signals(event, header_date)
    signals.extend(ctx_signals)

    return signals


def _infer_scene_label(trip: Optional[dict], event: dict, header: dict) -> str:
    """
    从 trip 信息和事件上下文推断场景标签。
    用于验证/展示，不参与管线。
    """
    weekday = header.get("weekday", "")
    place_id = event.get("place_semantic", {}).get("place_id", "")
    place_type = event.get("place_semantic", {}).get("place_type", "")
    trip_role = event.get("place_semantic", {}).get("trip_role", "")

    ts_str = event.get("timestamp", "")
    hour = -1
    try:
        hour = int(ts_str.split(":")[0])
    except (ValueError, IndexError):
        pass

    # 场景推断
    if place_type == "site_entrance" or place_id == "office_gate_01":
        vs = event.get("vehicle_state", {})
        speed = vs.get("velocity_kph", 999)
        if speed < 5:
            return "toll_parking_entry"

    if weekday in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
        if 5 <= hour < 12 and trip_role in ("origin",) and place_id == "home":
            return "morning_commute"
        if 16 <= hour < 23 and place_id == "home" and trip_role == "destination":
            return "arriving_home"

    if place_id == "park":
        return "weekend_park"
    if place_id == "mall":
        return "weekend_mall"

    return "en_route"


def load_unified_day(week_num: int, weekday: str) -> dict:
    """
    加载单天的 unified 数据并返回原始 JSON。

    Args:
        week_num: 周数 (1-4)
        weekday: 星期名 (monday, tuesday, ... sunday)

    Returns:
        原始 unified JSON dict
    """
    week_dir = DATA_ROOT / f"week_{week_num:02d}"
    filename = f"unified_week_case_01_w{week_num:02d}_{weekday.lower()}.json"
    filepath = week_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Unified data not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def unified_day_to_events(week_num: int, weekday: str) -> Tuple[List[dict], dict]:
    """
    加载一天的 unified 数据，转为内部事件格式列表。

    按 trip 和 phase 分割:
      - pre_departure: 启动到出发前的用户动作 (如 HVAC、导航设置)
      - waypoint:      途中经过 POI 时的用户动作 (如开窗)
      - post_arrival:  到达后的用户动作 (如音量归零、关闭无钥匙进入)

    每个 phase 产出一个 event，只包含该 phase 内的用户动作信号
    加上最近一条 vehicle_status 的上下文。

    Returns:
        (events, header): events 是内部事件列表，header 是当天元信息
    """
    raw = load_unified_day(week_num, weekday)
    header = raw.get("header", {})
    date_str = header.get("date", "")
    raw_events = raw.get("raw_events", [])
    trips = raw.get("motion_plan", {}).get("trips", [])

    events = []

    for trip in trips:
        trip_events = _get_trip_events(raw_events, trip)
        phases = _split_trip_into_phases(trip_events, trip)

        for phase_name, phase_events in phases:
            # 只从 user_action 事件收集行为信号
            action_signals = []
            for evt in phase_events:
                if evt.get("action"):
                    sigs = _convert_action_signals(evt, date_str)
                    action_signals.extend(sigs)

            if not action_signals:
                continue

            # 取该 phase 最近的 vehicle_status 作为上下文
            ctx_signals = _get_phase_context(phase_events, date_str)

            all_signals = action_signals + ctx_signals
            scene_label = _infer_phase_scene(
                phase_name, trip, phase_events, header
            )

            events.append({
                "scene": scene_label,
                "date": date_str,
                "weekday": header.get("weekday", ""),
                "day": 0,
                "weather": header.get("environment", {}).get("weather", ""),
                "signals": all_signals,
            })

    return events, header


def _get_trip_events(raw_events: list, trip: dict) -> List[dict]:
    """获取属于指定 trip 时间范围内的事件"""
    pt = trip.get("phase_times", {})
    start = pt.get("startup_time", "")
    end = pt.get("shutdown_time", "")
    if not start or not end:
        return []
    return [e for e in raw_events if start <= e.get("timestamp", "") <= end]


def _split_trip_into_phases(
    events: list, trip: dict
) -> List[Tuple[str, List[dict]]]:
    """
    将一个 trip 的事件按 phase 切分。

    Phase 边界由 place_id 变化定义：
    - 同一 place_id 下的连续事件归为一个 phase
    - phase_name 基于 place_id + trip_role (如 "home_origin", "office_gate_01_waypoint")
    """
    if not events:
        return []

    phases = []
    current_place = None
    current_events = []

    for evt in events:
        place_id = evt.get("place_semantic", {}).get("place_id", "unknown")
        trip_role = evt.get("place_semantic", {}).get("trip_role", "")

        phase_key = f"{place_id}_{trip_role}"

        if phase_key != current_place and current_events:
            phases.append((current_place, current_events))
            current_events = []
        current_place = phase_key
        current_events.append(evt)

    if current_events:
        phases.append((current_place, current_events))

    return phases


def _convert_action_signals(event: dict, header_date: str) -> List[dict]:
    """将 user_action 事件的 action 转为内部信号列表（只处理行为信号）"""
    signals = []
    ts_str = event.get("timestamp", "")
    full_ts = f"{header_date}T{ts_str}" if "T" not in ts_str else ts_str

    action = event.get("action")
    if not action or not action.get("name"):
        return signals

    action_name = action["name"]
    payload = action.get("payload", {})
    mapper = ACTION_TO_SIGNALS.get(action_name)
    if mapper:
        for sig_name, sig_value in mapper(payload, event):
            signals.append({
                "t": full_ts,
                "signal": sig_name,
                "value": sig_value,
            })
    else:
        log.debug(f"Unmapped action: {action_name}")

    return signals


def _get_phase_context(phase_events: list, header_date: str) -> List[dict]:
    """
    从 phase 事件中提取上下文信号。

    只取第一条 vehicle_status 的上下文（避免重复），
    加上 phase 内所有事件共享的 GPS 坐标。
    """
    signals = []
    status_found = False

    for evt in phase_events:
        if evt.get("event_type") == "vehicle_status" and not status_found:
            signals.extend(_extract_context_signals(evt, header_date))
            status_found = True
            continue
        # 即使没有 vehicle_status，也尝试从 user_action 的 vehicle_state 获取基础上下文
        if not status_found and evt.get("vehicle_state"):
            signals.extend(_extract_context_signals(evt, header_date))
            status_found = True

    return signals


def _infer_phase_scene(
    phase_name: str, trip: dict, events: list, header: dict
) -> str:
    """
    从 phase 信息推断场景标签。
    """
    weekday = header.get("weekday", "")
    is_weekday = weekday in (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
    )

    # 从 phase 事件取出首个 place 信息
    place_id = ""
    place_type = ""
    trip_role = ""
    hour = -1
    for evt in events:
        ps = evt.get("place_semantic", {})
        place_id = ps.get("place_id", "")
        place_type = ps.get("place_type", "")
        trip_role = ps.get("trip_role", "")
        ts_str = evt.get("timestamp", "")
        try:
            hour = int(ts_str.split(":")[0])
        except (ValueError, IndexError):
            pass
        break

    # trip 的起止信息
    trip_dest = trip.get("destination", {}).get("place_id", "") if trip else ""
    trip_origin = trip.get("origin", {}).get("place_id", "") if trip else ""

    # 办公区入口/低速场景
    if place_type == "site_entrance" or place_id == "office_gate_01":
        return "toll_parking_entry"

    # 工作日早高峰从家出发
    if is_weekday and 5 <= hour < 12 and place_id == "home" and trip_role == "origin":
        return "morning_commute"

    # 工作日晚间到家
    if is_weekday and 16 <= hour < 23 and place_id == "home" and trip_role == "destination":
        return "arriving_home"

    # 周末: 根据 trip 目的地推断
    if not is_weekday:
        if trip_dest == "park" or place_id == "park":
            return "weekend_park"
        if trip_dest == "mall" or place_id == "mall":
            return "weekend_mall"
        if place_id == "home" and trip_role == "origin":
            # 从家出发去某处 — 用 trip 目的地命名
            if trip_dest:
                return f"weekend_{trip_dest}"

    # 到达工作地点
    if place_id == "work" and trip_role == "destination":
        return "arriving_work"

    # 从工作地点出发
    if place_id == "work" and trip_role == "origin":
        return "leaving_work"

    return "en_route"


def load_week_events(week_num: int) -> Tuple[List[dict], dict]:
    """
    加载一整周 (7 天) 的 unified 数据并转为内部事件列表。

    Returns:
        (events, manifest_info): events 按时间排序的内部事件列表
    """
    all_events = []
    day_counter = (week_num - 1) * 7

    for i, weekday in enumerate(WEEKDAY_NAMES):
        try:
            day_events, header = unified_day_to_events(week_num, weekday)
            day_counter += 1
            for evt in day_events:
                evt["day"] = day_counter
            all_events.extend(day_events)
        except FileNotFoundError:
            log.warning(f"Missing unified data: week {week_num}, {weekday}")
            continue

    # 按首条信号的时间戳排序
    all_events.sort(key=lambda e: e.get("signals", [{}])[0].get("t", ""))

    manifest_path = DATA_ROOT / "four_week_manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return all_events, manifest


def load_multi_week_events(week_nums: List[int]) -> Tuple[List[dict], dict]:
    """
    加载多周数据，按时间排序返回。

    Args:
        week_nums: 要加载的周数列表，如 [1, 2] 或 [1, 2, 3, 4]

    Returns:
        (events, manifest): events 按时间排序
    """
    all_events = []
    manifest = {}

    for wn in week_nums:
        week_events, m = load_week_events(wn)
        all_events.extend(week_events)
        if not manifest:
            manifest = m

    all_events.sort(key=lambda e: e.get("signals", [{}])[0].get("t", ""))
    return all_events, manifest


def load_all_events() -> Tuple[List[dict], dict]:
    """加载全部四周数据"""
    return load_multi_week_events([1, 2, 3, 4])


def to_simulator_format(events: List[dict]) -> dict:
    """
    将内部事件列表转为 SignalSimulator 兼容的 mockup 格式。

    输出结构:
      {
        "user": "driver_001",
        "user_id": "driver_001",
        "base_date": "2026-03-30",
        "scenes": {
          "morning_commute": [...],
          "arriving_home": [...],
          ...
        }
      }
    """
    scenes: Dict[str, List[dict]] = {}
    for evt in events:
        scene = evt.get("scene", "unknown")
        scenes.setdefault(scene, []).append(evt)

    base_date = ""
    if events:
        base_date = events[0].get("date", "")

    return {
        "user": "driver_001",
        "user_id": "driver_001",
        "base_date": base_date,
        "scenes": scenes,
    }
