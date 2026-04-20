"""
Scene Extractor — Convert mockup data into a ComfyUI-ready scene manifest.

Reads from scenarios/mock_data_generator.py and produces a JSON manifest
with 15 entries (5 days x 3 habit scenes), each containing structured
scene data for LLM prompt generation.

Usage:
    python -m scripts.video_gen.scene_extractor
"""
import json
import os
import sys

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scenarios.mock_data_generator import generate_full_dataset
from scripts.video_gen.config import MANIFEST_PATH, LLM_SYSTEM_PROMPT, STYLE_SUFFIX, CINEMATIC_SIGNALS


def filter_cinematic_actions(signals: list[dict]) -> list[dict]:
    """Return only signals whose name is in CINEMATIC_SIGNALS.

    For signals that repeat within the scene (e.g. hvac_temp_target set
    twice), keep only the last occurrence — the final state is what gets
    shown on screen.
    """
    last_by_name: dict[str, dict] = {}
    order: list[str] = []
    for sig in signals:
        name = sig.get("signal", "")
        if name not in CINEMATIC_SIGNALS:
            continue
        if name not in last_by_name:
            order.append(name)
        last_by_name[name] = sig
    return [last_by_name[n] for n in order]


# ── Signal-to-action mapping ──
SIGNAL_LABELS = {
    "hvac_temp_target": "Sets AC to {value}°C",
    "hvac_fan_speed": "Fan speed: {value}",
    "hvac_power": "AC power: {value}",
    "seat_heating": "Seat heating level {value}",
    "media_source": "Media: {value}",
    "media_content_id": "Playing '{value}'",
    "media_volume": "Volume at {value}%",
    "drive_mode": "Drive mode: {value}",
    "acc_distance": "ACC distance: {value}",
    "nav_destination": "Navigates to {value}",
    "nav_route_pref": "Route preference: {value}",
    "keyless_entry": "Keyless entry: {value}",
    "window_position": "Window position: {value}%",
    "engine_status": "Engine: {value}",
    "door_status": "Door: {value}",
    "gear_position": "Gear: {value}",
    "vehicle_speed": "Speed: {value} km/h",
    "wiper_state": "Wipers: {value}",
}

# Signals to skip in action list (infrastructure, not user behavior)
SKIP_SIGNALS = {"gps_latitude", "gps_longitude"}

# Location derivation
SCENE_LOCATIONS = {
    "morning_commute": "Departing from home, residential Paris street",
    "arriving_home": "Arriving at home, Parisian apartment neighborhood",
    "toll_parking_entry": "Approaching {poi}",
}

# Time-of-day palette hints
TIME_PALETTES = {
    "morning": "warm amber, soft peach, pale blue sky — fresh, hopeful mood",
    "midday": "bright cream, muted teal, sage green — active, clear mood",
    "afternoon": "golden ochre, dusty rose, warm gray — settled, calm mood",
    "evening": "deep indigo, warm orange glow, violet — intimate, transitional mood",
}


def _extract_time(signals: list[dict]) -> str:
    """Extract time-of-day from the first signal timestamp."""
    if signals:
        ts = signals[0].get("t", "")
        if "T" in ts:
            return ts.split("T")[1][:5]  # HH:MM
    return "08:00"


def _extract_actions(signals: list[dict]) -> list[str]:
    """Convert raw signals into human-readable action descriptions."""
    actions = []
    seen = set()
    for sig in signals:
        name = sig.get("signal", "")
        value = sig.get("value", "")
        if name in SKIP_SIGNALS or name in seen:
            continue
        seen.add(name)
        template = SIGNAL_LABELS.get(name)
        if template:
            actions.append(template.format(value=value))
    return actions


def _derive_location(event: dict) -> str:
    """Derive human-readable location from scene context."""
    scene = event.get("scene", "")
    template = SCENE_LOCATIONS.get(scene, "Paris, France")
    poi = event.get("poi", "toll/parking area")
    if poi:
        poi = poi.replace("_", " ").replace("toll ", "Toll ").replace("parking ", "Parking ")
    return template.format(poi=poi) if "{poi}" in template else template


def _get_time_palette(time_str: str) -> str:
    """Return palette hint based on hour."""
    try:
        hour = int(time_str.split(":")[0])
    except (ValueError, IndexError):
        hour = 12
    if 6 <= hour < 10:
        return TIME_PALETTES["morning"]
    elif 10 <= hour < 14:
        return TIME_PALETTES["midday"]
    elif 14 <= hour < 17:
        return TIME_PALETTES["afternoon"]
    else:
        return TIME_PALETTES["evening"]


def _build_llm_user_prompt(entry: dict) -> str:
    """Build the user prompt string from a manifest entry."""
    actions_str = "\n".join(f"- {a}" for a in entry["actions"])
    weather_str = entry.get("weather", "")
    temp_str = f", {entry['temp_c']}°C" if entry.get("temp_c") else ""
    palette = _get_time_palette(entry["time"])

    return (
        f"Scene: {entry['scene'].replace('_', ' ').title()}\n"
        f"Day: {entry['day']} ({entry['weekday']})\n"
        f"Time: {entry['time']}\n"
        f"Weather: {weather_str}{temp_str}\n"
        f"Location: {entry['location']}\n"
        f"Color palette hint: {palette}\n"
        f"Actions performed:\n{actions_str}"
    )


def build_scene_manifest() -> list[dict]:
    """Convert mockup data into a scene manifest for ComfyUI batch processing."""
    dataset = generate_full_dataset()
    manifest = []

    for scene_type, events in dataset["scenes"].items():
        if scene_type == "noise":
            continue  # Skip non-habitual events

        for event in events:
            time_str = _extract_time(event["signals"])
            actions = _extract_actions(event["signals"])
            location = _derive_location(event)

            entry = {
                "id": f"day{event['day']}_{scene_type}",
                "scene": scene_type,
                "day": event["day"],
                "weekday": event.get("weekday", ""),
                "date": event.get("date", ""),
                "time": time_str,
                "weather": event.get("weather", ""),
                "temp_c": event.get("outside_temp_c", ""),
                "location": location,
                "actions": actions,
                "output_path": f"mary/day{event['day']}/{scene_type}",
            }
            # Pre-build the LLM user prompt
            entry["llm_user_prompt"] = _build_llm_user_prompt(entry)
            manifest.append(entry)

    return manifest


def save_manifest(manifest: list[dict], path: str = None):
    """Write manifest to JSON file."""
    path = path or MANIFEST_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(manifest)} scenes to {path}")


if __name__ == "__main__":
    manifest = build_scene_manifest()
    save_manifest(manifest)

    # Preview
    print(f"\n{'='*60}")
    print(f"Generated {len(manifest)} scene entries:")
    for entry in manifest:
        print(f"  [{entry['id']}] {entry['scene']} — {entry['time']} — {len(entry['actions'])} actions")
    print(f"\nSample LLM prompt for first scene:")
    print(f"{'─'*60}")
    print(manifest[0]["llm_user_prompt"])
