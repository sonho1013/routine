"""
signal_to_fact 单元测试

覆盖:
  - signals_to_facts(): 每种信号类型的 Fact 生成
  - _extract_structured_context(): PRD Table 4 上下文提取
  - _classify_time_bucket(): 时段分类
  - _classify_vehicle_state(): 车辆状态推断
  - _match_geofence(): GPS 围栏匹配
  - 边界情况: 空信号、未知信号、混合信号
"""
import os
import sys
import json
import pytest
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from engine.signal_to_fact import (
    signals_to_facts,
    _extract_structured_context,
    _classify_time_bucket,
    _classify_vehicle_state,
    _match_geofence,
    _extract_hour,
    _extract_weekday,
    KNOWN_GEOFENCES,
)
from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources


# ═══════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════

def _make_event(signals, date="2025-10-06", day=1):
    """构造最小 event_data"""
    return {"date": date, "day": day, "signals": signals}


def _sig(t, signal, value):
    """构造单条信号"""
    return {"t": t, "signal": signal, "value": value}


# ═══════════════════════════════════════════════════
# signals_to_facts — 各信号类型
# ═══════════════════════════════════════════════════

class TestHvacTempTarget:
    def test_basic(self):
        event = _make_event([_sig("2025-10-06T08:15:00", "hvac_temp_target", 22)])
        facts = signals_to_facts(event)
        assert len(facts) == 1
        assert facts[0].text == "set cabin air conditioning temperature to 22 degrees"

    def test_different_values(self):
        for temp in [16, 20, 28]:
            facts = signals_to_facts(_make_event([
                _sig("2025-10-06T10:00:00", "hvac_temp_target", temp)
            ]))
            assert f"to {temp} degrees" in facts[0].text


class TestHvacPower:
    def test_off(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "hvac_power", "off")
        ]))
        assert any(f.text == "turned off cabin air conditioning" for f in facts)

    def test_on_no_fact(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "hvac_power", "on")
        ]))
        assert not any("air conditioning" in f.text for f in facts)


class TestSeatHeating:
    def test_level_2(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "seat_heating", 2)
        ]))
        assert len(facts) == 1
        assert facts[0].text == "turned on seat heating to level 2"

    def test_level_0_no_fact(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "seat_heating", 0)
        ]))
        assert len(facts) == 0


class TestNavDestination:
    def test_work(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "nav_destination", "work")
        ]))
        assert facts[0].text == "started navigation to work"

    def test_home(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "nav_destination", "home")
        ]))
        assert facts[0].text == "started navigation to home"


class TestNavRoutePref:
    def test_fastest(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "nav_route_pref", "fastest")
        ]))
        assert facts[0].text == "selected fastest route preference"

    def test_no_toll(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "nav_route_pref", "no_toll")
        ]))
        assert facts[0].text == "selected no_toll route preference"


class TestMediaSource:
    def test_podcast(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_source", "podcast"),
            _sig("2025-10-06T08:00:01", "media_content_id", "Tech Daily"),
        ]))
        texts = [f.text for f in facts]
        assert "resumed listening to podcast Tech Daily" in texts

    def test_music(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_source", "music"),
            _sig("2025-10-06T08:00:01", "media_content_id", "Chill Beats"),
        ]))
        assert any("started playing music Chill Beats" in f.text for f in facts)

    def test_radio(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_source", "radio"),
            _sig("2025-10-06T08:00:01", "media_content_id", "FM 95.5"),
        ]))
        assert any("tuned to radio station FM 95.5" in f.text for f in facts)

    def test_off(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_source", "off"),
        ]))
        assert any("stopped all media playback" in f.text for f in facts)

    def test_podcast_without_content_id(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_source", "podcast"),
        ]))
        assert any("podcast unknown" in f.text for f in facts)

    def test_standalone_content_id(self):
        """media_content_id without media_source"""
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_content_id", "My Playlist"),
        ]))
        assert any("selected media content My Playlist" in f.text for f in facts)


class TestMediaVolume:
    def test_nonzero(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_volume", 58)
        ]))
        assert any("adjusted media volume to 58 percent" in f.text for f in facts)

    def test_zero_without_media_off(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_volume", 0)
        ]))
        assert any("stopped all media playback" in f.text for f in facts)

    def test_zero_with_media_off_no_duplicate(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "media_source", "off"),
            _sig("2025-10-06T08:00:01", "media_volume", 0),
        ]))
        stop_facts = [f for f in facts if "stopped all media playback" in f.text]
        assert len(stop_facts) == 1  # 不重复


class TestDriveMode:
    def test_eco(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "drive_mode", "eco")
        ]))
        assert facts[0].text == "switched to eco driving mode"

    def test_sport(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "drive_mode", "sport")
        ]))
        assert facts[0].text == "switched to sport driving mode"


class TestAccDistance:
    def test_medium(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "acc_distance", "medium")
        ]))
        assert facts[0].text == "set ACC following distance to medium"


class TestWindowPosition:
    def test_lowered(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "window_position", 50)
        ]))
        assert facts[0].text == "lowered driver window to 50 percent"

    def test_closed(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "window_position", 0)
        ]))
        assert facts[0].text == "closed all vehicle windows"


class TestKeylessEntry:
    def test_disabled(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "keyless_entry", "disabled")
        ]))
        assert any("disabled keyless proximity entry" in f.text for f in facts)

    def test_enabled_no_fact(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "keyless_entry", "enabled")
        ]))
        assert not any("keyless" in f.text for f in facts)


class TestEngineStatus:
    def test_off(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T18:00:00", "engine_status", "off")
        ]))
        assert any("shut down the vehicle engine" in f.text for f in facts)

    def test_on_no_shutdown_fact(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "engine_status", "on")
        ]))
        assert not any("shut down" in f.text for f in facts)


# ═══════════════════════════════════════════════════
# Fact 属性验证
# ═══════════════════════════════════════════════════

class TestFactProperties:
    def test_type_and_durability(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "drive_mode", "eco")
        ]))
        f = facts[0]
        assert f.type == FactType.PREF
        assert f.durability == FactDurability.LONG_TERM
        assert f.source == FactSources.SIGNAL

    def test_timestamp_from_signal(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:15:00", "drive_mode", "eco")
        ]))
        assert facts[0].time_stamp == datetime(2025, 10, 6, 8, 15, 0)

    def test_json_metadata(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "seat_heating", 3)
        ]))
        meta = json.loads(facts[0].json_metadata)
        assert meta["signal"] == "seat_heating"
        assert meta["raw_value"] == 3

    def test_unique_ids(self):
        facts = signals_to_facts(_make_event([
            _sig("2025-10-06T08:00:00", "hvac_temp_target", 22),
            _sig("2025-10-06T08:00:01", "drive_mode", "eco"),
            _sig("2025-10-06T08:00:02", "seat_heating", 2),
        ]))
        ids = [f.id for f in facts]
        assert len(ids) == len(set(ids))


# ═══════════════════════════════════════════════════
# StructuredContext 提取
# ═══════════════════════════════════════════════════

class TestExtractStructuredContext:
    def test_morning_weekday(self):
        event = _make_event([
            _sig("2025-10-06T08:15:00", "engine_status", "on"),
            _sig("2025-10-06T08:15:10", "gear_position", "D"),
        ], date="2025-10-06")  # Monday
        ctx = _extract_structured_context(event)
        assert ctx.time_bucket == "early_morning"
        assert ctx.hour == 8
        assert ctx.weekday is True
        assert ctx.vehicle_state == "engine_started"

    def test_evening_weekend(self):
        event = _make_event([
            _sig("2025-10-11T19:30:00", "engine_status", "off"),
            _sig("2025-10-11T19:30:05", "gear_position", "P"),
        ], date="2025-10-11")  # Saturday
        ctx = _extract_structured_context(event)
        assert ctx.time_bucket == "evening"
        assert ctx.hour == 19
        assert ctx.weekday is False
        assert ctx.vehicle_state == "parked"

    def test_geofence_home(self):
        lat, lon, _ = KNOWN_GEOFENCES["home"]
        event = _make_event([
            _sig("2025-10-06T08:00:00", "gps_latitude", lat),
            _sig("2025-10-06T08:00:00", "gps_longitude", lon),
        ])
        ctx = _extract_structured_context(event)
        assert ctx.geofence == "home"

    def test_geofence_workplace(self):
        lat, lon, _ = KNOWN_GEOFENCES["workplace"]
        event = _make_event([
            _sig("2025-10-06T09:00:00", "gps_latitude", lat + 0.0001),
            _sig("2025-10-06T09:00:00", "gps_longitude", lon - 0.0001),
        ])
        ctx = _extract_structured_context(event)
        assert ctx.geofence == "workplace"

    def test_no_geofence(self):
        event = _make_event([
            _sig("2025-10-06T08:00:00", "gps_latitude", 40.0),
            _sig("2025-10-06T08:00:00", "gps_longitude", -74.0),
        ])
        ctx = _extract_structured_context(event)
        assert ctx.geofence is None

    def test_no_gps_signals(self):
        event = _make_event([
            _sig("2025-10-06T08:00:00", "engine_status", "on"),
        ])
        ctx = _extract_structured_context(event)
        assert ctx.geofence is None


# ═══════════════════════════════════════════════════
# _classify_time_bucket
# ═══════════════════════════════════════════════════

class TestClassifyTimeBucket:
    @pytest.mark.parametrize("hour,expected", [
        ("04:30:00", "night"),
        ("05:00:00", "early_morning"),
        ("08:59:00", "early_morning"),
        ("09:00:00", "morning"),
        ("11:59:00", "morning"),
        ("12:00:00", "midday"),
        ("13:59:00", "midday"),
        ("14:00:00", "afternoon"),
        ("17:59:00", "afternoon"),
        ("18:00:00", "evening"),
        ("21:59:00", "evening"),
        ("22:00:00", "night"),
        ("23:59:00", "night"),
    ])
    def test_buckets(self, hour, expected):
        signals = [{"t": f"2025-10-06T{hour}", "signal": "engine_status", "value": "on"}]
        result = _classify_time_bucket(signals)
        assert result == expected, f"hour={hour}: got {result}, expected {expected}"

    def test_empty_signals(self):
        assert _classify_time_bucket([]) == "unknown"


# ═══════════════════════════════════════════════════
# _classify_vehicle_state
# ═══════════════════════════════════════════════════

class TestClassifyVehicleState:
    def test_engine_on(self):
        signals = [{"signal": "engine_status", "value": "on", "t": "2025-10-06T08:00:00"}]
        assert _classify_vehicle_state(signals) == "engine_started"

    def test_engine_off(self):
        signals = [{"signal": "engine_status", "value": "off", "t": "2025-10-06T18:00:00"}]
        assert _classify_vehicle_state(signals) == "parked"

    def test_gear_park(self):
        signals = [{"signal": "gear_position", "value": "P", "t": "2025-10-06T18:00:00"}]
        assert _classify_vehicle_state(signals) == "parked"

    def test_crawling(self):
        signals = [
            {"signal": "engine_status", "value": "on", "t": "2025-10-06T08:00:00"},
            {"signal": "vehicle_speed", "value": 3.0, "t": "2025-10-06T08:00:01"},
        ]
        assert _classify_vehicle_state(signals) == "crawling"

    def test_unknown(self):
        signals = [{"signal": "hvac_temp_target", "value": 22, "t": "2025-10-06T08:00:00"}]
        assert _classify_vehicle_state(signals) == "unknown"


# ═══════════════════════════════════════════════════
# _match_geofence
# ═══════════════════════════════════════════════════

class TestMatchGeofence:
    def test_exact_match(self):
        for name, (lat, lon, _) in KNOWN_GEOFENCES.items():
            assert _match_geofence(lat, lon) == name

    def test_within_radius(self):
        lat, lon, radius = KNOWN_GEOFENCES["home"]
        # 偏移约 50m — 仍在 200m 半径内
        assert _match_geofence(lat + 0.0004, lon) == "home"

    def test_outside_radius(self):
        lat, lon, radius = KNOWN_GEOFENCES["workplace"]
        # 偏移约 500m — 超出 100m 半径
        assert _match_geofence(lat + 0.005, lon) is None

    def test_none_coords(self):
        assert _match_geofence(None, None) is None
        assert _match_geofence(48.8566, None) is None
        assert _match_geofence(None, 2.3522) is None


# ═══════════════════════════════════════════════════
# _extract_hour / _extract_weekday
# ═══════════════════════════════════════════════════

class TestExtractHour:
    def test_normal(self):
        signals = [{"t": "2025-10-06T14:30:00", "signal": "x", "value": 1}]
        assert _extract_hour(signals) == 14

    def test_empty(self):
        assert _extract_hour([]) == -1

    def test_bad_format(self):
        signals = [{"t": "not-a-time", "signal": "x", "value": 1}]
        assert _extract_hour(signals) == -1


class TestExtractWeekday:
    def test_monday(self):
        assert _extract_weekday({"date": "2025-10-06"}) is True  # Monday

    def test_saturday(self):
        assert _extract_weekday({"date": "2025-10-11"}) is False  # Saturday

    def test_sunday(self):
        assert _extract_weekday({"date": "2025-10-12"}) is False  # Sunday

    def test_no_date(self):
        assert _extract_weekday({}) is None

    def test_bad_date(self):
        assert _extract_weekday({"date": "not-a-date"}) is None


# ═══════════════════════════════════════════════════
# 边界情况
# ═══════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_signals(self):
        assert signals_to_facts({"date": "2025-10-06", "signals": []}) == []

    def test_no_signals_key(self):
        assert signals_to_facts({"date": "2025-10-06"}) == []

    def test_context_only_signals_no_facts(self):
        """纯上下文信号（Table 4）不产生 Fact"""
        event = _make_event([
            _sig("2025-10-06T08:00:00", "gps_latitude", 48.8566),
            _sig("2025-10-06T08:00:00", "gps_longitude", 2.3522),
            _sig("2025-10-06T08:00:00", "door_status", "closed"),
            _sig("2025-10-06T08:00:00", "wiper_state", "off"),
        ])
        facts = signals_to_facts(event)
        assert len(facts) == 0

    def test_duplicate_signal_takes_last(self):
        """同名信号取最后一个值"""
        event = _make_event([
            _sig("2025-10-06T08:00:00", "hvac_temp_target", 20),
            _sig("2025-10-06T08:00:30", "hvac_temp_target", 24),
        ])
        facts = signals_to_facts(event)
        assert len(facts) == 1
        assert "24 degrees" in facts[0].text

    def test_multi_signal_event(self):
        """混合多种信号 — 类似真实 morning_commute 事件"""
        event = _make_event([
            _sig("2025-10-06T08:15:00", "engine_status", "on"),
            _sig("2025-10-06T08:15:06", "door_status", "closed"),
            _sig("2025-10-06T08:15:13", "gear_position", "D"),
            _sig("2025-10-06T08:15:56", "media_source", "podcast"),
            _sig("2025-10-06T08:16:19", "nav_destination", "work"),
            _sig("2025-10-06T08:16:44", "seat_heating", 2),
            _sig("2025-10-06T08:17:04", "drive_mode", "eco"),
            _sig("2025-10-06T08:17:20", "nav_route_pref", "fastest"),
            _sig("2025-10-06T08:17:30", "media_content_id", "Tech Daily"),
            _sig("2025-10-06T08:17:47", "media_volume", 58),
            _sig("2025-10-06T08:18:08", "hvac_fan_speed", "auto"),
            _sig("2025-10-06T08:18:19", "acc_distance", "medium"),
            _sig("2025-10-06T08:18:22", "hvac_temp_target", 22),
        ])
        facts = signals_to_facts(event)
        texts = [f.text for f in facts]

        # 验证各信号都有对应 Fact
        assert any("22 degrees" in t for t in texts)
        assert any("seat heating" in t for t in texts)
        assert any("navigation to work" in t for t in texts)
        assert any("fastest route" in t for t in texts)
        assert any("podcast Tech Daily" in t for t in texts)
        assert any("volume to 58" in t for t in texts)
        assert any("eco driving" in t for t in texts)
        assert any("ACC following distance" in t for t in texts)

        # 所有 Fact 都有正确的 StructuredContext
        for f in facts:
            assert f.context.time_bucket == "early_morning"
            assert f.context.hour == 8
            assert f.context.weekday is True
            assert f.context.vehicle_state == "engine_started"


# ═══════════════════════════════════════════════════
# 与 mockup 数据集成验证
# ═══════════════════════════════════════════════════

class TestMockupIntegration:
    """使用真实 mockup 数据验证"""

    @pytest.fixture
    def mockup_data(self):
        data_file = os.path.join(os.path.dirname(__file__), "step1_mockup_data.json")
        if not os.path.exists(data_file):
            pytest.skip("mockup data not found")
        import json
        with open(data_file) as f:
            return json.load(f)

    def test_total_fact_count(self, mockup_data):
        """验证全量 mockup 数据产出 83 facts"""
        all_facts = []
        for scene_events in mockup_data["scenes"].values():
            for event in scene_events:
                all_facts.extend(signals_to_facts(event))
        assert len(all_facts) == 83

    def test_all_facts_have_context(self, mockup_data):
        """所有 Fact 都含 StructuredContext"""
        for scene_events in mockup_data["scenes"].values():
            for event in scene_events:
                for f in signals_to_facts(event):
                    assert isinstance(f.context, StructuredContext)
                    assert f.context.time_bucket != "unknown"
                    assert f.context.hour >= 0

    def test_no_scene_label_in_text(self, mockup_data):
        """Fact.text 不应包含场景标签（裸动作原则）"""
        scene_labels = {"morning_commute", "arriving_home", "toll_parking_entry", "noise"}
        for scene_events in mockup_data["scenes"].values():
            for event in scene_events:
                for f in signals_to_facts(event):
                    for label in scene_labels:
                        assert label not in f.text.lower(), \
                            f"Scene label '{label}' found in fact text: '{f.text}'"
