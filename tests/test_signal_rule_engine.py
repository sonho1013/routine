"""SignalRuleEngine 单元测试。"""
import pytest

from engine.signal_rules import SignalRuleEngine
from panoramix_core.models.fact_enums import FactType


@pytest.fixture
def engine():
    return SignalRuleEngine()


def test_engine_loads_default_rules(engine):
    signals = engine.list_signals()
    assert "hvac_temp_target" in signals
    assert "media_source" in signals
    assert len(signals) >= 11

def test_engine_loads_geofences(engine):
    geos = engine.get_geofences()
    assert "home" in geos
    assert geos["home"]["radius_m"] == 200

def test_get_rule_returns_rule_dict(engine):
    rule = engine.get_rule("hvac_temp_target")
    assert rule["category"] == "numeric"
    assert rule["drift_metric"] == "numeric_mean_diff"
    assert rule["drift_threshold"] == 1.5

def test_get_rule_missing_returns_none(engine):
    assert engine.get_rule("nonexistent_signal") is None

def test_signals_to_facts_hvac_temp(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 22},
        ],
    }
    facts = engine.signals_to_facts(event)
    assert len(facts) == 1
    assert facts[0].type == FactType.PREF
    assert "22" in facts[0].text
    assert "temperature" in facts[0].text.lower()

def test_signals_to_facts_media_music(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "media_source", "value": "music"},
            {"t": "2026-04-08T08:00:01", "signal": "media_content_id", "value": "jazz_playlist"},
        ],
    }
    facts = engine.signals_to_facts(event)
    music_facts = [f for f in facts if "playing music" in f.text]
    assert len(music_facts) == 1
    assert "jazz_playlist" in music_facts[0].text

def test_signals_to_facts_seat_heating_off_skipped(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "seat_heating", "value": 0},
        ],
    }
    facts = engine.signals_to_facts(event)
    assert all("seat heating" not in f.text for f in facts)

def test_signals_to_facts_hvac_out_of_range_dropped(engine, caplog):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 100},
        ],
    }
    facts = engine.signals_to_facts(event)
    temp_facts = [f for f in facts if "temperature" in f.text]
    assert len(temp_facts) == 0

def test_signals_to_facts_unknown_signal_dropped(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "totally_unknown_xyz", "value": 42},
        ],
    }
    facts = engine.signals_to_facts(event)
    assert facts == []

def test_fact_json_metadata_contains_signal_and_raw_value(engine):
    import json
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 22},
        ],
    }
    facts = engine.signals_to_facts(event)
    meta = json.loads(facts[0].json_metadata)
    assert meta["signal"] == "hvac_temp_target"
    assert meta["raw_value"] == 22

def test_structured_context_extraction(engine):
    event = {
        "date": "2026-04-08",
        "signals": [
            {"t": "2026-04-08T08:00:00", "signal": "hvac_temp_target", "value": 22},
            {"t": "2026-04-08T08:00:01", "signal": "engine_status", "value": "on"},
            {"t": "2026-04-08T08:00:02", "signal": "gps_latitude", "value": 31.2304},
            {"t": "2026-04-08T08:00:03", "signal": "gps_longitude", "value": 121.4737},
        ],
    }
    facts = engine.signals_to_facts(event)
    ctx = facts[0].context
    assert ctx.time_bucket == "early_morning"
    assert ctx.weekday is True
    assert ctx.geofence == "home"
    assert ctx.vehicle_state == "engine_started"
