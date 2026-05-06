"""Tab 2 control-param rendering: verify it reads raw_value_stats directly
instead of regex-parsing the habit_text. Regression guard after we started
letting GPT-4.1 write habit descriptions narratively (context-anchored
sentences), which defeated the old `parse_habit_actions` regex and made
control-param chips render as empty."""
from simulator.tab2_recommendation import _format_params_from_stats


def test_numeric_temp_renders_with_decimal():
    stats = {"type": "numeric", "mean": 22.5, "std": 0.8, "count": 5}
    assert _format_params_from_stats("hvac_temp_target", stats) == "AC=22.5°C"


def test_numeric_temp_integer_mean_drops_decimal():
    stats = {"type": "numeric", "mean": 23.0, "count": 4}
    assert _format_params_from_stats("hvac_temp_target", stats) == "AC=23°C"


def test_numeric_percent_rounds_to_int():
    stats = {"type": "numeric", "mean": 87.4, "count": 6}
    assert _format_params_from_stats("driver_window_position", stats) == "Driver Window=87%"


def test_numeric_volume_percent():
    stats = {"type": "numeric", "mean": 45.0, "count": 3}
    assert _format_params_from_stats("media_volume", stats) == "Vol=45%"


def test_categorical_drive_mode():
    stats = {"type": "categorical", "dominant_value": "silent",
             "value_counts": {"silent": 4, "comfort": 1}, "count": 5}
    assert _format_params_from_stats("drive_mode", stats) == "Mode=silent"


def test_categorical_without_dominant_value_returns_dash():
    stats = {"type": "categorical", "value_counts": {}, "count": 0}
    assert _format_params_from_stats("drive_mode", stats) == "—"


def test_empty_stats_returns_dash():
    assert _format_params_from_stats("hvac_temp_target", {}) == "—"


def test_empty_signal_returns_dash():
    stats = {"type": "numeric", "mean": 22, "count": 3}
    assert _format_params_from_stats("", stats) == "—"


def test_unknown_signal_uses_raw_name_as_label():
    stats = {"type": "numeric", "mean": 5.0, "count": 3}
    assert _format_params_from_stats("unknown_signal", stats) == "unknown_signal=5"
