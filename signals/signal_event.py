"""
信号抽象层 — SignalEvent 定义 + 信号清单

覆盖:
  PRD Table 4: 10 维触发上下文变量 → StructuredContext
  PRD Table 5: 3 场景涉及的 ~8 项行为控制 → Fact.text

职责边界:
  信号采集/埋点由外部系统完成，不在本项目范围。
  本模块定义信号的类型规范，并提供 mockup 数据解析工具。
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union


# ═══════════════════════════════════════════════════
# SignalEvent 定义
# ═══════════════════════════════════════════════════

@dataclass
class SignalEvent:
    """单条车辆信号事件"""
    timestamp: datetime
    signal_name: str       # 信号名，必须属于 SIGNAL_CATALOG
    value: Any             # 信号值，类型由 SIGNAL_CATALOG 定义
    source: str = "simulator"  # "simulator" | "vhal" | "replay"


class SignalRole(str, Enum):
    """信号角色分类"""
    CONTEXT = "context"    # PRD Table 4: 触发上下文 → StructuredContext
    BEHAVIOR = "behavior"  # PRD Table 5: 行为控制 → Fact.text
    BOTH = "both"          # 同时属于两个表 (如 hvac_temp_target)
    DERIVED = "derived"    # 不直接出现在信号流，从其他信号推算 (如 time, day)


# ═══════════════════════════════════════════════════
# 信号清单 (PRD Table 4 + Table 5 完整映射)
# ═══════════════════════════════════════════════════

@dataclass
class SignalSpec:
    """信号规格定义"""
    name: str                    # 信号名
    role: SignalRole             # 角色分类
    value_type: str              # "float" | "int" | "str" | "bool"
    value_range: str             # 取值范围描述
    prd_table: str               # "T4" | "T5" | "T4+T5"
    prd_variable: str            # PRD 原文变量名
    scenarios: str               # 涉及的场景 "S1,S2,S3"
    description: str             # 说明
    maps_to: str                 # 目标字段: "ctx.time_bucket" | "fact.text" 等


# PRD Table 4: 触发上下文变量 (10 维)
# PRD Table 5: 行为控制维度 (~8 项)
SIGNAL_CATALOG: Dict[str, SignalSpec] = {s.name: s for s in [

    # ─── PRD Table 4: 触发上下文 → StructuredContext ───

    SignalSpec(
        name="vehicle_speed",
        role=SignalRole.CONTEXT,
        value_type="float",
        value_range="0-250 kph (PRD: 0,5,30,80,120)",
        prd_table="T4", prd_variable="vehicle speed",
        scenarios="S3",
        description="当前车速",
        maps_to="ctx.vehicle_state (→ crawling if <5)",
    ),
    SignalSpec(
        name="gps_latitude",
        role=SignalRole.CONTEXT,
        value_type="float",
        value_range="-90.0 ~ 90.0",
        prd_table="T4", prd_variable="memorized address / GPS geofence",
        scenarios="S1,S2,S3",
        description="GPS 纬度",
        maps_to="ctx.geofence (经纬度匹配围栏)",
    ),
    SignalSpec(
        name="gps_longitude",
        role=SignalRole.CONTEXT,
        value_type="float",
        value_range="-180.0 ~ 180.0",
        prd_table="T4", prd_variable="memorized address / GPS geofence",
        scenarios="S1,S2,S3",
        description="GPS 经度",
        maps_to="ctx.geofence (经纬度匹配围栏)",
    ),
    SignalSpec(
        name="engine_status",
        role=SignalRole.BOTH,
        value_type="str",
        value_range="on | off",
        prd_table="T4+T5", prd_variable="vehicle start/end",
        scenarios="S1,S2",
        description="引擎状态",
        maps_to="ctx.vehicle_state + fact.text (熄火动作)",
    ),
    SignalSpec(
        name="gear_position",
        role=SignalRole.CONTEXT,
        value_type="str",
        value_range="P | R | N | D",
        prd_table="T4", prd_variable="vehicle start/end",
        scenarios="S1,S2",
        description="挡位",
        maps_to="ctx.vehicle_state (P → parked)",
    ),
    SignalSpec(
        name="door_status",
        role=SignalRole.CONTEXT,
        value_type="str",
        value_range="open | closed | locked",
        prd_table="T4", prd_variable="door open/locked",
        scenarios="S2",
        description="车门状态",
        maps_to="ctx (辅助判断 vehicle_state)",
    ),
    SignalSpec(
        name="wiper_state",
        role=SignalRole.CONTEXT,
        value_type="str",
        value_range="off | low | medium | high | max",
        prd_table="T4", prd_variable="(天气条件)",
        scenarios="S3",
        description="雨刮器状态，间接指示天气",
        maps_to="ctx (辅助条件信号)",
    ),

    # ─── PRD Table 4 派生维度 (从 timestamp/date 推算) ───

    SignalSpec(
        name="_time",
        role=SignalRole.DERIVED,
        value_type="datetime",
        value_range="从 SignalEvent.timestamp 推算",
        prd_table="T4", prd_variable="time",
        scenarios="S1,S2",
        description="时间窗口 (06:00-10:00, 17:00-21:00)",
        maps_to="ctx.time_bucket + ctx.hour",
    ),
    SignalSpec(
        name="_day",
        role=SignalRole.DERIVED,
        value_type="str",
        value_range="从 date 字段推算 weekday/weekend",
        prd_table="T4", prd_variable="day",
        scenarios="S1",
        description="工作日/周末",
        maps_to="ctx.weekday",
    ),
    SignalSpec(
        name="_poi_type",
        role=SignalRole.DERIVED,
        value_type="str",
        value_range="从 geofence 匹配结果推算",
        prd_table="T4", prd_variable="POI type",
        scenarios="S3",
        description="POI 类型 (toll, parking, site entrance)",
        maps_to="ctx.geofence (围栏名包含语义)",
    ),

    # ─── PRD Table 5: 行为控制 → Fact.text ───

    SignalSpec(
        name="hvac_temp_target",
        role=SignalRole.BOTH,
        value_type="int",
        value_range="16-28 °C",
        prd_table="T4+T5", prd_variable="A/C Temperature / temperature setpoint",
        scenarios="S1",
        description="空调目标温度",
        maps_to="ctx (T4) + fact.text (T5): 'set cabin air conditioning temperature to {v} degrees'",
    ),
    SignalSpec(
        name="hvac_power",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="on | off",
        prd_table="T5", prd_variable="A/C",
        scenarios="S2",
        description="空调开关",
        maps_to="fact.text: 'turned off cabin air conditioning'",
    ),
    SignalSpec(
        name="hvac_fan_speed",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="auto | 1 | 2 | 3 | 4 | 5",
        prd_table="T5", prd_variable="A/C",
        scenarios="S1",
        description="空调风速",
        maps_to="fact.text (MS1 不生成 Fact，仅记录)",
    ),
    SignalSpec(
        name="seat_heating",
        role=SignalRole.BEHAVIOR,
        value_type="int",
        value_range="0 (OFF), 1, 2, 3",
        prd_table="T5", prd_variable="Seat Heating/Ventilation",
        scenarios="S1",
        description="座椅加热等级",
        maps_to="fact.text: 'turned on seat heating to level {v}'",
    ),
    SignalSpec(
        name="window_position",
        role=SignalRole.BEHAVIOR,
        value_type="int",
        value_range="0 (closed), 50, 80, 100 %",
        prd_table="T5", prd_variable="Window Opening Level",
        scenarios="S2,S3",
        description="车窗开度 (驾驶员侧)",
        maps_to="fact.text: 'lowered driver window to {v} percent' / 'closed all vehicle windows'",
    ),
    SignalSpec(
        name="media_source",
        role=SignalRole.BOTH,
        value_type="str",
        value_range="podcast | music | radio | off",
        prd_table="T4+T5", prd_variable="audio source / Music/Podcast",
        scenarios="S1,S2",
        description="媒体来源",
        maps_to="ctx (T4) + fact.text (T5): 'resumed listening to podcast {id}'",
    ),
    SignalSpec(
        name="media_content_id",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="播客/音乐/电台名称",
        prd_table="T5", prd_variable="Music/Podcast Playlist",
        scenarios="S1",
        description="媒体内容标识",
        maps_to="fact.text: 嵌入媒体播放描述",
    ),
    SignalSpec(
        name="media_volume",
        role=SignalRole.BOTH,
        value_type="int",
        value_range="0-100 %",
        prd_table="T4+T5", prd_variable="audio volume / Volume",
        scenarios="S1,S2",
        description="媒体音量",
        maps_to="ctx (T4) + fact.text (T5): 'adjusted media volume to {v} percent'",
    ),
    SignalSpec(
        name="nav_destination",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="work | home | supermarket | POI name",
        prd_table="T5", prd_variable="Navigation Destination",
        scenarios="S1",
        description="导航目的地",
        maps_to="fact.text: 'started navigation to {v}'",
    ),
    SignalSpec(
        name="nav_route_pref",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="fastest | no_toll | highway",
        prd_table="T5", prd_variable="Navigation Route",
        scenarios="S1",
        description="路线偏好",
        maps_to="fact.text: 'selected {v} route preference'",
    ),
    SignalSpec(
        name="keyless_entry",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="enabled | disabled",
        prd_table="T5", prd_variable="Lock Auto Lock",
        scenarios="S2",
        description="无钥匙进入开关",
        maps_to="fact.text: 'disabled keyless proximity entry'",
    ),
    SignalSpec(
        name="drive_mode",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="eco | normal | sport",
        prd_table="T5", prd_variable="Drive Mode",
        scenarios="S1",
        description="驾驶模式",
        maps_to="fact.text: 'switched to {v} driving mode'",
    ),
    SignalSpec(
        name="acc_distance",
        role=SignalRole.BEHAVIOR,
        value_type="str",
        value_range="short | medium | long | off",
        prd_table="T5", prd_variable="ACC Following Distance",
        scenarios="S1",
        description="ACC 跟车距离",
        maps_to="fact.text: 'set ACC following distance to {v}'",
    ),
]}


# ═══════════════════════════════════════════════════
# 便捷查询
# ═══════════════════════════════════════════════════

# 按角色分组的信号名集合
CONTEXT_SIGNALS = {
    name for name, spec in SIGNAL_CATALOG.items()
    if spec.role in (SignalRole.CONTEXT, SignalRole.BOTH)
    and not name.startswith("_")
}

BEHAVIOR_SIGNALS = {
    name for name, spec in SIGNAL_CATALOG.items()
    if spec.role in (SignalRole.BEHAVIOR, SignalRole.BOTH)
    and not name.startswith("_")
}

ALL_PHYSICAL_SIGNALS = CONTEXT_SIGNALS | BEHAVIOR_SIGNALS


def is_known_signal(signal_name: str) -> bool:
    """信号名是否在清单中"""
    return signal_name in SIGNAL_CATALOG


# ═══════════════════════════════════════════════════
# Mockup 数据解析
# ═══════════════════════════════════════════════════

def parse_event_signals(event_data: Dict) -> List[SignalEvent]:
    """
    将 mockup event_data 中的信号列表解析为 SignalEvent 对象列表。

    Args:
        event_data: mockup 事件数据，结构:
            {
              "signals": [
                {"t": "2025-10-06T08:15:00", "signal": "engine_status", "value": "on"},
                ...
              ]
            }

    Returns:
        List[SignalEvent]: 按时间排序的信号事件
    """
    events = []
    for raw in event_data.get("signals", []):
        try:
            ts = datetime.fromisoformat(raw["t"])
        except (ValueError, KeyError):
            continue
        events.append(SignalEvent(
            timestamp=ts,
            signal_name=raw["signal"],
            value=raw["value"],
            source="simulator",
        ))
    return events
