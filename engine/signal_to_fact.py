"""
Signal-to-Fact shim — 委托给 SignalRuleEngine

向后兼容层：保留 `signals_to_facts()` 函数签名让 engine/habit_engine.py
和老测试继续工作，内部完全走新的 yaml-driven 规则引擎。

新代码应直接用 `engine.signal_rules.SignalRuleEngine`。
"""
from typing import Dict, List, Optional

from engine.signal_rules import SignalRuleEngine
from panoramix_core.models.fact import Fact

_engine: SignalRuleEngine | None = None


def _get_engine() -> SignalRuleEngine:
    global _engine
    if _engine is None:
        _engine = SignalRuleEngine()
    return _engine


def signals_to_facts(event_data: Dict) -> List[Fact]:
    """向后兼容包装器。"""
    return _get_engine().signals_to_facts(event_data)


# ── Re-exports of internal helpers (for backwards compatibility with old tests) ──

def _extract_structured_context(event_data: Dict):
    """向后兼容：从事件数据提取 StructuredContext。"""
    eng = _get_engine()
    signals_list = event_data.get("signals", [])
    sig_map: Dict = {s["signal"]: s["value"] for s in signals_list}
    return eng._extract_structured_context(event_data, sig_map)


def _classify_time_bucket(signals_list: list) -> str:
    """向后兼容：时段分类。"""
    return _get_engine()._classify_time_bucket(signals_list)


def _extract_hour(signals_list: list) -> int:
    """向后兼容：提取信号小时。"""
    return _get_engine()._extract_hour(signals_list)


def _extract_weekday(event_data: Dict) -> Optional[bool]:
    """向后兼容：工作日判断。"""
    return _get_engine()._extract_weekday(event_data)


def _classify_vehicle_state(signals_list: list) -> str:
    """向后兼容：车辆状态推断（接受 signals_list 参数）。"""
    eng = _get_engine()
    sig_map: Dict = {s["signal"]: s["value"] for s in signals_list}
    return eng._classify_vehicle_state(signals_list, sig_map)


def _match_geofence(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    """向后兼容：GPS 地理围栏匹配。"""
    return _get_engine()._match_geofence(lat, lon)


def _build_known_geofences():
    engine = _get_engine()
    return {
        name: (cfg["lat"], cfg["lon"], cfg["radius_m"])
        for name, cfg in engine.get_geofences().items()
    }


KNOWN_GEOFENCES = _build_known_geofences()
