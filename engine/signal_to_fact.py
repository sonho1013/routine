"""
Signal-to-Fact 适配器 — 对齐 panoramix_core 模型

输入: 外部埋点系统提供的信号批数据 (event_data dict)
输出: List[Fact]，每条 Fact 包含裸动作文本 + StructuredContext

信号分流:
  - PRD Table 4 (触发/上下文信号) → StructuredContext (ctx_* metadata)
  - PRD Table 5 (行为控制信号)   → Fact.text 裸动作描述
"""
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional

from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources

log = logging.getLogger(__name__)

# ── 地理围栏预配置 (lat, lon, radius_m) ──
# Shanghai coordinates (对齐 unified 数据的 location_profile)
KNOWN_GEOFENCES = {
    "home":           (31.2304, 121.4737, 200),
    "workplace":      (31.2396, 121.4997, 100),
    "office_gate_01": (31.2388, 121.4975, 100),
    "park":           (31.2270, 121.4610, 200),
    "mall":           (31.2350, 121.4900, 150),
}


# ═══════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════

def signals_to_facts(event_data: Dict) -> List[Fact]:
    """
    将一条信号事件转换为 Fact 列表。

    Args:
        event_data: 埋点系统提供的单次事件数据，结构:
            {
              "date": "2025-10-06",
              "day": 1,
              "signals": [
                {"t": "2025-10-06T08:15:00", "signal": "engine_status", "value": "on"},
                {"t": "...", "signal": "hvac_temp_target", "value": 22},
                ...
              ],
              // 可选: "scene" (仅 mockup 数据中用于验证，不影响 Fact 生成)
            }

    Returns:
        List[Fact]: 裸动作 Fact 列表，每条含 StructuredContext
    """
    signals_list = event_data.get("signals", [])
    if not signals_list:
        return []

    # 合并同名信号（取最后一个值）
    sig_map = {}
    for s in signals_list:
        sig_map[s["signal"]] = s["value"]

    # 提取结构化上下文 (Table 4 → StructuredContext)
    ctx = _extract_structured_context(event_data)

    # 解析事件时间戳
    ts = _parse_timestamp(event_data)

    # 构建 Fact 列表 (Table 5 → Fact.text)
    facts: List[Fact] = []

    # ── 空调温度 ──
    if "hvac_temp_target" in sig_map:
        v = sig_map["hvac_temp_target"]
        facts.append(_make_fact(
            f"set cabin air conditioning temperature to {v} degrees",
            ctx, ts, "hvac_temp_target", v))

    # ── 空调关闭 ──
    if sig_map.get("hvac_power") == "off":
        facts.append(_make_fact(
            "turned off cabin air conditioning",
            ctx, ts, "hvac_power", "off"))

    # ── 座椅加热 ──
    if "seat_heating" in sig_map:
        v = sig_map["seat_heating"]
        if isinstance(v, (int, float)) and v > 0:
            facts.append(_make_fact(
                f"turned on seat heating to level {v}",
                ctx, ts, "seat_heating", v))

    # ── 导航目的地 ──
    if "nav_destination" in sig_map:
        v = sig_map["nav_destination"]
        facts.append(_make_fact(
            f"started navigation to {v}",
            ctx, ts, "nav_destination", v))

    # ── 路线偏好 ──
    if "nav_route_pref" in sig_map:
        v = sig_map["nav_route_pref"]
        facts.append(_make_fact(
            f"selected {v} route preference",
            ctx, ts, "nav_route_pref", v))

    # ── 媒体播放 ──
    media_type = sig_map.get("media_source")
    content_id = sig_map.get("media_content_id", "unknown")
    media_handled = False

    if media_type == "podcast":
        facts.append(_make_fact(
            f"resumed listening to podcast {content_id}",
            ctx, ts, "media_source", media_type))
        media_handled = True
    elif media_type == "music":
        facts.append(_make_fact(
            f"started playing music {content_id}",
            ctx, ts, "media_source", media_type))
        media_handled = True
    elif media_type == "radio":
        facts.append(_make_fact(
            f"tuned to radio station {content_id}",
            ctx, ts, "media_source", media_type))
        media_handled = True
    elif media_type == "off":
        facts.append(_make_fact(
            "stopped all media playback",
            ctx, ts, "media_off", "off"))
        media_handled = True

    # 独立的 media_content_id（无 media_source 时）
    if "media_content_id" in sig_map and not media_handled:
        facts.append(_make_fact(
            f"selected media content {content_id}",
            ctx, ts, "media_content_id", content_id))

    # ── 媒体音量 ──
    if "media_volume" in sig_map:
        v = sig_map["media_volume"]
        if isinstance(v, (int, float)) and v > 0:
            facts.append(_make_fact(
                f"adjusted media volume to {v} percent",
                ctx, ts, "media_volume", v))
        elif v == 0 and media_type != "off":
            facts.append(_make_fact(
                "stopped all media playback",
                ctx, ts, "media_off", "off"))

    # ── 驾驶模式 ──
    if "drive_mode" in sig_map:
        v = sig_map["drive_mode"]
        facts.append(_make_fact(
            f"switched to {v} driving mode",
            ctx, ts, "drive_mode", v))

    # ── ACC 跟车距离 ──
    if "acc_distance" in sig_map:
        v = sig_map["acc_distance"]
        facts.append(_make_fact(
            f"set ACC following distance to {v}",
            ctx, ts, "acc_distance", v))

    # ── 车窗 ──
    if "window_position" in sig_map:
        v = sig_map["window_position"]
        if isinstance(v, (int, float)):
            if v > 0:
                facts.append(_make_fact(
                    f"lowered driver window to {v} percent",
                    ctx, ts, "window_position", v))
            else:
                facts.append(_make_fact(
                    "closed all vehicle windows",
                    ctx, ts, "window_position", 0))

    # ── 无钥匙进入 ──
    if sig_map.get("keyless_entry") == "disabled":
        facts.append(_make_fact(
            "disabled keyless proximity entry",
            ctx, ts, "keyless_entry", "disabled"))

    # ── 熄火 ──
    if sig_map.get("engine_status") == "off":
        facts.append(_make_fact(
            "shut down the vehicle engine",
            ctx, ts, "engine_status", "off"))

    return facts


# ═══════════════════════════════════════════════════
# StructuredContext 提取 (PRD Table 4 → ctx_*)
# ═══════════════════════════════════════════════════

def _extract_structured_context(event_data: Dict) -> StructuredContext:
    """从原始信号事件提取结构化上下文维度"""
    signals_list = event_data.get("signals", [])
    sig_map = {s["signal"]: s["value"] for s in signals_list}

    return StructuredContext(
        time_bucket=_classify_time_bucket(signals_list),
        hour=_extract_hour(signals_list),
        weekday=_extract_weekday(event_data),
        vehicle_state=_classify_vehicle_state(signals_list),
        geofence=_match_geofence(sig_map.get("gps_latitude"),
                                  sig_map.get("gps_longitude")),
    )


def _classify_time_bucket(signals_list: list) -> str:
    """从首条信号时间戳分类时段"""
    hour = _extract_hour(signals_list)
    if hour < 0:
        return "unknown"
    if 5 <= hour < 9:
        return "early_morning"
    if 9 <= hour < 12:
        return "morning"
    if 12 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "night"


def _extract_hour(signals_list: list) -> int:
    """从首条信号时间戳提取小时"""
    if not signals_list:
        return -1
    try:
        ts_str = signals_list[0]["t"]
        return int(ts_str.split("T")[1].split(":")[0])
    except (IndexError, ValueError, KeyError, AttributeError):
        return -1


def _extract_weekday(event_data: Dict) -> Optional[bool]:
    """从 date 字段判断工作日/周末"""
    date_str = event_data.get("date", "")
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.weekday() < 5
    except ValueError:
        return None


def _classify_vehicle_state(signals_list: list) -> str:
    """从信号列表推断车辆状态"""
    sig_map = {s["signal"]: s["value"] for s in signals_list}

    has_engine_on = any(
        s["signal"] == "engine_status" and s["value"] == "on"
        for s in signals_list)
    has_engine_off = any(
        s["signal"] == "engine_status" and s["value"] == "off"
        for s in signals_list)
    has_gear_park = sig_map.get("gear_position") == "P"
    speeds = [s["value"] for s in signals_list
              if s["signal"] == "vehicle_speed"
              and isinstance(s["value"], (int, float))]

    if has_engine_off or has_gear_park:
        return "parked"
    if has_engine_on:
        if speeds and all(0 < sp < 5 for sp in speeds):
            return "crawling"
        return "engine_started"
    if speeds and all(0 < sp < 5 for sp in speeds):
        return "crawling"
    return "unknown"


def _match_geofence(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """GPS 坐标匹配已知地理围栏"""
    if lat is None or lon is None:
        return None
    for name, (glat, glon, radius_m) in KNOWN_GEOFENCES.items():
        dlat = (lat - glat) * 111_320
        dlon = (lon - glon) * 111_320 * math.cos(math.radians(glat))
        dist = math.sqrt(dlat ** 2 + dlon ** 2)
        if dist < radius_m:
            return name
    return None


# ═══════════════════════════════════════════════════
# Fact 构造
# ═══════════════════════════════════════════════════

def _parse_timestamp(event_data: Dict) -> datetime:
    """从事件数据解析时间戳"""
    signals_list = event_data.get("signals", [])
    if signals_list:
        try:
            return datetime.fromisoformat(signals_list[0]["t"])
        except (ValueError, KeyError):
            pass
    date_str = event_data.get("date", "")
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            pass
    return datetime.now()


def _make_fact(text: str, ctx: StructuredContext, ts: datetime,
               signal_name: str, raw_value) -> Fact:
    """构造 Fact 对象"""
    import json
    return Fact(
        text=text,
        type=FactType.PREF,
        durability=FactDurability.LONG_TERM,
        time_stamp=ts,
        source=FactSources.SIGNAL,
        context=ctx,
        json_metadata=json.dumps({
            "signal": signal_name,
            "raw_value": raw_value,
        }),
    )
