"""
SignalRuleEngine — yaml-driven signal→Fact translation.

Loads rules from rules.yaml and implements all signal-to-fact conversion
logic previously hardcoded in engine/signal_to_fact.py.
"""
import json
import logging
import math
import os
from datetime import datetime
from typing import Dict, List, Optional

import yaml

from panoramix_core.models.fact import Fact, StructuredContext
from panoramix_core.models.fact_enums import FactType, FactDurability, FactSources

log = logging.getLogger(__name__)

_DEFAULT_RULES_PATH = os.path.join(os.path.dirname(__file__), "rules.yaml")


class SignalRuleEngine:
    """YAML-driven engine for translating vehicle signals into Fact objects."""

    def __init__(self, rules_path: Optional[str] = None):
        path = rules_path or _DEFAULT_RULES_PATH
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._geofences: Dict = data.get("geofences", {})
        self._signals: Dict = data.get("signals", {})
        self._context_signal_list: List[str] = data.get("context_signals", [])

    # ── Public API ──────────────────────────────────────────────────────────

    def list_signals(self) -> List[str]:
        """Return list of known signal names."""
        return list(self._signals.keys())

    def get_rule(self, signal_name: str) -> Optional[Dict]:
        """Return rule dict for a signal, or None if not found."""
        return self._signals.get(signal_name)

    def get_geofences(self) -> Dict:
        """Return geofences dict with {name: {lat, lon, radius_m}}."""
        return self._geofences

    def context_signals(self) -> List[str]:
        """Return list of context-only signal names (no Fact generated)."""
        return self._context_signal_list

    def signals_to_facts(self, event_data: Dict) -> List[Fact]:
        """
        Convert a signal event dict into a list of Fact objects.

        Args:
            event_data: dict with 'date', 'signals' list of {t, signal, value}

        Returns:
            List[Fact] with StructuredContext attached to each fact.
        """
        signals_list = event_data.get("signals", [])
        if not signals_list:
            return []

        # Build sig_map: last value wins for duplicate signals
        sig_map: Dict = {}
        for s in signals_list:
            sig_map[s["signal"]] = s["value"]

        ctx = self._extract_structured_context(event_data, sig_map)
        ts = self._parse_timestamp(event_data)
        facts: List[Fact] = []

        # Track whether media was handled (to avoid duplicate content_id fact)
        media_handled = False

        # Process signals in definition order from YAML, but also handle special cases
        # We iterate over sig_map keys to produce facts
        for signal_name, value in sig_map.items():
            # Skip context-only signals
            if signal_name in self._context_signal_list:
                continue

            rule = self._signals.get(signal_name)
            if rule is None:
                # Unknown signal — drop silently
                log.debug("Unknown signal '%s' — dropped", signal_name)
                continue

            # Special handling for media_source (uses content_id from sig_map)
            if signal_name == "media_source":
                text = self._render_media_source(value, rule, sig_map)
                if text is not None:
                    facts.append(self._make_fact(text, ctx, ts, signal_name, value))
                media_handled = True
                continue

            # Special handling for media_content_id (only emit if no media_source)
            if signal_name == "media_content_id":
                if not media_handled and "media_source" not in sig_map:
                    text = self._render_text(signal_name, value, rule, sig_map)
                    if text is not None:
                        facts.append(self._make_fact(text, ctx, ts, signal_name, value))
                continue

            # Special handling for media_volume=0 (→ "stopped all media playback")
            if signal_name == "media_volume" and value == 0:
                media_type = sig_map.get("media_source")
                if media_type != "off":
                    facts.append(self._make_fact(
                        "stopped all media playback", ctx, ts, "media_off", "off"))
                continue

            text = self._render_text(signal_name, value, rule, sig_map)
            if text is not None:
                facts.append(self._make_fact(text, ctx, ts, signal_name, value))

        return facts

    # ── Internal rendering helpers ───────────────────────────────────────────

    def _render_media_source(self, value: str, rule: Dict, sig_map: Dict) -> Optional[str]:
        """Render text for media_source signal, substituting content_id."""
        value_templates = rule.get("value_templates", {})
        if str(value) not in value_templates:
            return None
        template = value_templates[str(value)]
        content_id = sig_map.get("media_content_id", "unknown")
        return template.replace("{content_id}", str(content_id))

    def _render_text(self, signal_name: str, value, rule: Dict, sig_map: Dict) -> Optional[str]:
        """
        Render fact text for a signal value using the rule definition.
        Returns None if the signal should be skipped (trigger condition not met,
        out of range, or no matching template).
        """
        category = rule.get("category", "categorical")

        # ── value_range check (numeric) ──
        if category == "numeric":
            value_range = rule.get("value_range")
            if value_range and isinstance(value, (int, float)):
                lo, hi = value_range
                if not (lo <= value <= hi):
                    log.warning(
                        "Signal '%s' value %s out of range [%s, %s] — dropped",
                        signal_name, value, lo, hi,
                    )
                    return None

        # ── trigger_condition check ──
        condition = rule.get("trigger_condition")
        if condition and not self._eval_trigger(condition, value):
            return None

        # ── value_templates (categorical exact match) ──
        value_templates = rule.get("value_templates")
        if value_templates:
            key = str(value)
            if key in value_templates:
                return value_templates[key]
            # No matching template → no fact
            return None

        # ── value_templates_by_range (window_position style) ──
        by_range = rule.get("value_templates_by_range")
        if by_range:
            if isinstance(value, (int, float)):
                if value > 0:
                    tpl = by_range.get("positive", "")
                    return tpl.replace("{value}", str(value))
                else:
                    tpl = by_range.get("zero", "")
                    return tpl.replace("{value}", str(value))
            return None

        # ── generic text_template ──
        text_template = rule.get("text_template")
        if text_template:
            return text_template.replace("{value}", str(value))

        return None

    def _eval_trigger(self, condition: str, value) -> bool:
        """
        Safely evaluate a simple trigger condition like 'value > 0'.
        Only supports comparisons of 'value' against a numeric literal.
        No eval() used.
        """
        condition = condition.strip()
        for op, fn in [
            (">=", lambda a, b: a >= b),
            ("<=", lambda a, b: a <= b),
            ("!=", lambda a, b: a != b),
            ("==", lambda a, b: a == b),
            (">",  lambda a, b: a > b),
            ("<",  lambda a, b: a < b),
        ]:
            if op in condition:
                parts = condition.split(op, 1)
                if parts[0].strip() == "value":
                    try:
                        threshold = float(parts[1].strip())
                        if isinstance(value, (int, float)):
                            return fn(value, threshold)
                    except (ValueError, TypeError):
                        pass
                    return False
        # Unknown condition format → pass through
        return True

    # ── StructuredContext extraction ─────────────────────────────────────────

    def _extract_structured_context(
        self, event_data: Dict, sig_map: Dict
    ) -> StructuredContext:
        """Extract StructuredContext from event signals."""
        signals_list = event_data.get("signals", [])
        return StructuredContext(
            time_bucket=self._classify_time_bucket(signals_list),
            hour=self._extract_hour(signals_list),
            weekday=self._extract_weekday(event_data),
            vehicle_state=self._classify_vehicle_state(signals_list, sig_map),
            geofence=self._match_geofence(
                sig_map.get("gps_latitude"),
                sig_map.get("gps_longitude"),
            ),
            poi_type=self._normalize_poi_type(sig_map.get("poi_type")),
            wiper_state=self._normalize_wiper(sig_map.get("wiper_state")),
            temp_bucket=self._classify_temp_bucket(sig_map.get("outside_temp_c")),
            door_lock=self._normalize_door_lock(sig_map.get("door_lock_state")),
            window_state=self._classify_window_state(
                sig_map.get("window_state_snapshot")
            ),
            approach_unlock=self._normalize_approach_unlock(
                sig_map.get("approach_unlock_state")
            ),
        )

    # ── 新增 6 个 trigger 维度的归一化 / 分桶 helpers ──

    @staticmethod
    def _normalize_wiper(value) -> str:
        """接受字符串 (off/low/medium/high/max) 或整型 0-4。"""
        if value is None:
            return "unknown"
        if isinstance(value, (int, float)):
            return {0: "off", 1: "low", 2: "medium", 3: "high", 4: "max"}.get(
                int(value), "unknown"
            )
        v = str(value).strip().lower()
        return v if v in {"off", "low", "medium", "high", "max"} else "unknown"

    @staticmethod
    def _classify_temp_bucket(celsius) -> str:
        """<10 cold, 10-22 mild, 22-28 warm, >=28 hot。"""
        if celsius is None:
            return "unknown"
        try:
            c = float(celsius)
        except (TypeError, ValueError):
            return "unknown"
        if c < 10:
            return "cold"
        if c < 22:
            return "mild"
        if c < 28:
            return "warm"
        return "hot"

    @staticmethod
    def _normalize_door_lock(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "locked" if value else "unlocked"
        v = str(value).strip().lower()
        return v if v in {"locked", "unlocked"} else None

    @staticmethod
    def _classify_window_state(value) -> str:
        """接受 str (CLOSED/HALF/OPEN/closed/...) 或百分比数值。"""
        if value is None:
            return "unknown"
        if isinstance(value, (int, float)):
            v = float(value)
            if v <= 0:
                return "closed"
            if v < 80:
                return "partial"
            return "open"
        s = str(value).strip().lower()
        if s in {"closed", "close"}:
            return "closed"
        if s in {"half", "half open", "partial"}:
            return "partial"
        if s in {"open", "full", "full open"}:
            return "open"
        return "unknown"

    @staticmethod
    def _normalize_approach_unlock(value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, bool):
            return "enabled" if value else "disabled"
        v = str(value).strip().lower()
        return v if v in {"enabled", "disabled"} else None

    @staticmethod
    def _normalize_poi_type(value) -> Optional[str]:
        if value is None:
            return None
        v = str(value).strip().lower()
        # 允许的 POI 类型：与 place_semantic.place_type 对齐
        allowed = {"home", "work", "site_entrance", "park", "mall", "custom"}
        return v if v in allowed else None

    def _classify_time_bucket(self, signals_list: list) -> str:
        """Map first signal's hour to a named time bucket."""
        hour = self._extract_hour(signals_list)
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

    def _extract_hour(self, signals_list: list) -> int:
        """Extract hour integer from first signal's timestamp."""
        if not signals_list:
            return -1
        try:
            ts_str = signals_list[0]["t"]
            return int(ts_str.split("T")[1].split(":")[0])
        except (IndexError, ValueError, KeyError, AttributeError):
            return -1

    def _extract_weekday(self, event_data: Dict) -> Optional[bool]:
        """Return True if date is a weekday, False for weekend, None if unknown."""
        date_str = event_data.get("date", "")
        if not date_str:
            return None
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.weekday() < 5
        except ValueError:
            return None

    def _classify_vehicle_state(self, signals_list: list, sig_map: Dict) -> str:
        """Infer vehicle state from engine/gear/speed signals."""
        has_engine_on = any(
            s["signal"] == "engine_status" and s["value"] == "on"
            for s in signals_list
        )
        has_engine_off = any(
            s["signal"] == "engine_status" and s["value"] == "off"
            for s in signals_list
        )
        has_gear_park = sig_map.get("gear_position") == "P"
        speeds = [
            s["value"]
            for s in signals_list
            if s["signal"] == "vehicle_speed" and isinstance(s["value"], (int, float))
        ]

        if has_engine_off or has_gear_park:
            return "parked"
        if has_engine_on:
            if speeds and all(0 < sp < 5 for sp in speeds):
                return "crawling"
            return "engine_started"
        if speeds and all(0 < sp < 5 for sp in speeds):
            return "crawling"
        return "unknown"

    def _match_geofence(self, lat: Optional[float], lon: Optional[float]) -> Optional[str]:
        """Match GPS coordinates to a named geofence using haversine-lite distance."""
        if lat is None or lon is None:
            return None
        for name, cfg in self._geofences.items():
            glat = cfg["lat"]
            glon = cfg["lon"]
            radius_m = cfg["radius_m"]
            dlat = (lat - glat) * 111_320
            dlon = (lon - glon) * 111_320 * math.cos(math.radians(glat))
            dist = math.sqrt(dlat ** 2 + dlon ** 2)
            if dist < radius_m:
                return name
        return None

    # ── Fact construction ────────────────────────────────────────────────────

    def _parse_timestamp(self, event_data: Dict) -> datetime:
        """Parse timestamp from first signal or date field."""
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

    def _make_fact(
        self,
        text: str,
        ctx: StructuredContext,
        ts: datetime,
        signal_name: str,
        raw_value,
    ) -> Fact:
        """Construct a Fact with metadata."""
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
