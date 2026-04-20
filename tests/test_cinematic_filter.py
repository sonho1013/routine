from scripts.video_gen.scene_extractor import filter_cinematic_actions


def test_filters_out_non_visual_signals():
    signals = [
        {"signal": "engine_status",    "value": "on"},
        {"signal": "seat_heating",     "value": 2},        # filtered
        {"signal": "nav_destination",  "value": "work"},
        {"signal": "acc_distance",     "value": "medium"}, # filtered
        {"signal": "hvac_temp_target", "value": 22},
    ]
    kept = filter_cinematic_actions(signals)
    kept_names = [s["signal"] for s in kept]
    assert kept_names == ["engine_status", "nav_destination", "hvac_temp_target"]


def test_deduplicates_repeated_signals_keeping_last_value():
    signals = [
        {"signal": "hvac_temp_target", "value": 21},
        {"signal": "hvac_temp_target", "value": 22},
    ]
    kept = filter_cinematic_actions(signals)
    assert len(kept) == 1
    assert kept[0]["value"] == 22
