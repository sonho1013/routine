"""
Tab 2: Scene Card Manager

Two-tier layout:
  - Upper tier: Accepted scene cards (user has confirmed)
  - Lower tier: Pending / Recommendation scene cards (awaiting user action)

Each scene card displays:
  - Scene name (display_name) + status badge
  - Grouped habits from card snapshot (not latest batch)
  - Per-habit control params + confidence
  - Dominant context summary

Actions:
  - Accept: pending/recommendation -> accepted (freezes snapshot)
  - Dismiss: removes pending / rejects recommendation
"""
import streamlit as st
import logging

from engine.proactive_executor import parse_habit_actions

log = logging.getLogger(__name__)

USERS = ["Mary", "Tom", "Alice", "David", "Lena"]


def render():
    """Tab 2 entry point"""

    col_user, _ = st.columns([2, 3])
    with col_user:
        username = st.selectbox("Driver Profile", USERS, key="t2_user")

    st.divider()

    status = _load_status(username)
    if status is None:
        st.info("No data available. Use Tab 1 to train the model first.")
        return

    scene_cards = status.get("scene_cards", [])
    if not scene_cards:
        st.info("No scene cards yet. Use Tab 1 to train the model first.")
        return

    accepted_cards = [c for c in scene_cards if c["status"] == "accepted"]
    pending_cards = [c for c in scene_cards if c["status"] in ("pending", "recommendation")]

    # --- Summary metrics ---
    total_habits = sum(len(c.get("snapshot_habits", [])) for c in scene_cards)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Scene Cards", len(scene_cards))
    with c2:
        st.metric("Active", len(accepted_cards))
    with c3:
        st.metric("Pending", len(pending_cards))
    with c4:
        st.metric("Total Habits", total_habits)

    # ── Upper tier: Accepted ──
    st.markdown("### Active Scene Cards")
    if accepted_cards:
        for card in accepted_cards:
            _render_accepted_card(card)
    else:
        st.caption("No accepted scene cards yet. Accept cards from the recommendations below.")

    st.divider()

    # ── Lower tier: Pending / Recommendation ──
    st.markdown("### Pending Recommendations")
    if pending_cards:
        for card in pending_cards:
            _render_pending_card(username, card)
    else:
        st.caption("No pending recommendations. Run learning in Tab 1 to generate new scene cards.")


# ═══════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════

def _load_status(username: str):
    try:
        from engine.habit_engine import HabitDemoEngine
        # Tab 2 is display + SQLite-only: _load_status reads habit_memory.db;
        # accept/dismiss are pure UPDATE statements. Drift detection runs in
        # Tab 1's ingest_signal_batch, not here. No LLM needed — pass None so
        # we don't eagerly construct an OpenAI client we never use.
        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        status = engine.get_status()
        engine.close()
        return status
    except Exception as e:
        log.debug(f"Tab2 load status: {e}")
        return None


# ═══════════════════════════════════════════════════
# Card rendering — uses snapshot_habits from card itself
# ═══════════════════════════════════════════════════

def _render_accepted_card(card: dict):
    display_name = card.get("display_name", "—")
    context_str = _format_dominant_context(card.get("dominant_context", {}))
    habits = card.get("snapshot_habits", [])
    habits_html = _build_snapshot_habits_html(habits)

    st.html(
        f'<div style="background:#1A2A42;border:1px solid #1E3A5F;'
        f'border-left:3px solid #00C896;border-radius:8px;'
        f'padding:14px 16px;margin-bottom:8px;font-family:sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:4px;">'
        f'<span style="font-weight:600;font-size:0.95rem;color:#E8EDF3;">{display_name}</span>'
        f'<span style="display:inline-block;background:rgba(0,200,150,0.15);color:#00C896;'
        f'padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:600;">ACCEPTED</span>'
        f'</div>'
        f'<div style="font-size:0.7rem;color:#8899AA;margin-bottom:8px;">{context_str}</div>'
        f'{habits_html}'
        f'</div>'
    )


def _render_pending_card(username: str, card: dict):
    card_id = card["card_id"]
    display_name = card.get("display_name", "—")
    status = card.get("status", "pending")
    context_str = _format_dominant_context(card.get("dominant_context", {}))
    habits = card.get("snapshot_habits", [])

    if status == "recommendation":
        border_color, status_text = "#F5A623", "RECOMMENDATION"
        status_color, status_bg = "#F5A623", "rgba(245,166,35,0.15)"
    else:
        border_color, status_text = "#00BFC8", "PENDING"
        status_color, status_bg = "#00BFC8", "rgba(0,191,200,0.15)"

    habits_html = _build_snapshot_habits_html(habits)

    st.html(
        f'<div style="background:#1A2A42;border:1px solid #1E3A5F;'
        f'border-left:3px solid {border_color};border-radius:8px;'
        f'padding:14px 16px;margin-bottom:4px;font-family:sans-serif;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:4px;">'
        f'<span style="font-weight:600;font-size:0.95rem;color:#E8EDF3;">{display_name}</span>'
        f'<span style="display:inline-block;background:{status_bg};color:{status_color};'
        f'padding:2px 8px;border-radius:4px;font-size:0.7rem;font-weight:600;">'
        f'{status_text}</span>'
        f'</div>'
        f'<div style="font-size:0.7rem;color:#8899AA;margin-bottom:8px;">{context_str}</div>'
        f'{habits_html}'
        f'</div>'
    )

    # Accept / Dismiss buttons
    c_accept, c_dismiss, c_spacer = st.columns([1, 1, 2])
    with c_accept:
        if st.button("Accept", key=f"t2_accept_{card_id}", type="primary",
                      use_container_width=True):
            _handle_accept(username, card_id, display_name)
    with c_dismiss:
        if st.button("Dismiss", key=f"t2_dismiss_{card_id}",
                      use_container_width=True):
            _handle_dismiss(username, card_id, display_name, status)


# ═══════════════════════════════════════════════════
# Snapshot habits HTML — renders habits from card snapshot
# ═══════════════════════════════════════════════════

def _build_snapshot_habits_html(snapshot_habits: list) -> str:
    """Build HTML for habits from a scene card's content snapshot."""
    if not snapshot_habits:
        return '<div style="font-size:0.8rem;color:#8899AA;">No habits in snapshot</div>'

    rows = []
    for h in snapshot_habits:
        text = h.get("habit_text", "")
        signal = h.get("signal", "")
        conf = h.get("confidence", 0)
        stats = h.get("raw_value_stats", {})
        count = stats.get("count", 0)

        # Prefer the aggregated raw_value_stats (numeric mean / categorical
        # dominant_value) attached at cluster time — they are the source of
        # truth for the control parameter. Fall back to regex-over-habit-text
        # for legacy habits whose stats shape is unknown.
        params_str = _format_params_from_stats(signal, stats)
        if params_str == "—":
            params_str = _format_actions(text)
        badge = _conf_badge(conf)

        params_block = ""
        if params_str != "—":
            badges = "".join(
                f'<span style="display:inline-block;background:#132035;'
                f'border:1px solid #1E3A5F;padding:1px 6px;border-radius:3px;'
                f'font-size:0.7rem;color:#E8EDF3;margin:2px 2px 0 0;">{p.strip()}</span>'
                for p in params_str.split(",")
            )
            params_block = f'<div style="margin-top:4px;">{badges}</div>'

        rows.append(
            f'<div style="padding:6px 0;border-top:1px solid rgba(30,58,95,0.4);">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="flex:1;">'
            f'<div style="font-size:0.8rem;color:#E8EDF3;margin-bottom:2px;">'
            f'{text[:80]}</div>'
            f'<div style="font-size:0.65rem;color:#667788;">{signal}</div>'
            f'</div>'
            f'<div style="text-align:right;min-width:60px;">'
            f'{badge}'
            f'<div style="font-size:0.6rem;color:#8899AA;">{count} obs</div>'
            f'</div>'
            f'</div>'
            f'{params_block}'
            f'</div>'
        )

    return "".join(rows)


# ═══════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════

def _format_dominant_context(ctx: dict) -> str:
    """Format dominant_context from snapshot into readable string."""
    if not ctx:
        return "—"
    parts = []
    tb = ctx.get("time_bucket", "")
    if tb:
        parts.append(tb.replace("_", " ").title())
    vs = ctx.get("vehicle_state", "")
    if vs:
        parts.append(vs.replace("_", " ").title())
    geo = ctx.get("geofence")
    if geo:
        parts.append(geo.title())
    wd = ctx.get("weekday")
    if wd is True:
        parts.append("Weekday")
    elif wd is False:
        parts.append("Weekend")
    return " · ".join(parts) if parts else "—"


def _conf_badge(conf: float) -> str:
    if conf >= 0.85:
        color, bg = "#00C896", "rgba(0,200,150,0.15)"
    elif conf >= 0.70:
        color, bg = "#00BFC8", "rgba(0,191,200,0.15)"
    else:
        color, bg = "#8899AA", "rgba(136,153,170,0.12)"
    return (
        f'<span style="display:inline-block;padding:2px 6px;border-radius:3px;'
        f'font-size:0.75rem;font-weight:600;color:{color};background:{bg};">'
        f'{conf:.2f}</span>'
    )


# signal_name → human label shown on the Control Params chip
_SIGNAL_LABEL = {
    "hvac_temp_target": "AC",
    "hvac_power": "AC Power",
    "seat_heating": "Seat Heat",
    "nav_destination": "Nav",
    "nav_route_pref": "Route",
    "media_source": "Media",
    "media_content_id": "Content",
    "media_volume": "Vol",
    "media_off": "Media",
    "drive_mode": "Mode",
    "acc_distance": "ACC",
    "window_position": "Window",
    "driver_window_position": "Driver Window",
    "keyless_entry": "Keyless",
    "engine_status": "Engine",
}

# signal_name → unit suffix
_SIGNAL_UNIT = {
    "hvac_temp_target": "°C",
    "media_volume": "%",
    "window_position": "%",
    "driver_window_position": "%",
}


def _format_params_from_stats(signal: str, stats: dict) -> str:
    """Render control params from the habit's aggregated raw_value_stats.

    Numeric stats use the mean (rounded sensibly); categorical stats use the
    dominant_value. Returns "—" when the signal is unknown or stats are empty.
    """
    if not signal or not stats:
        return "—"
    label = _SIGNAL_LABEL.get(signal, signal)
    unit = _SIGNAL_UNIT.get(signal, "")
    stype = stats.get("type")
    if stype == "numeric":
        mean = stats.get("mean")
        if mean is None:
            return "—"
        # Integer-like (volume, window %) render without decimal;
        # others (temperature) render one decimal.
        if unit in ("%",) or float(mean).is_integer():
            value = f"{int(round(mean))}"
        else:
            value = f"{mean:.1f}"
        return f"{label}={value}{unit}"
    if stype == "categorical":
        dv = stats.get("dominant_value")
        if not dv:
            return "—"
        return f"{label}={dv}"
    return "—"


def _format_actions(habit_text: str) -> str:
    actions = parse_habit_actions(habit_text)
    if not actions:
        return "—"
    _LABELS = {
        "hvac_temp_target": "AC",
        "hvac_power": "AC Power",
        "seat_heating": "Seat Heat",
        "nav_destination": "Nav",
        "nav_route_pref": "Route",
        "media_source": "Media",
        "media_content_id": "Content",
        "media_volume": "Vol",
        "media_off": "Media",
        "drive_mode": "Mode",
        "acc_distance": "ACC",
        "window_position": "Window",
        "keyless_entry": "Keyless",
        "engine_status": "Engine",
    }
    params = []
    for k, v in actions.items():
        label = _LABELS.get(k, k)
        if k == "hvac_temp_target":
            params.append(f"{label}={v:g}\u00b0C")
        elif k == "media_volume":
            params.append(f"{label}={v}%")
        elif k == "window_position":
            params.append(f"{label}={v}%")
        elif k == "media_off":
            params.append("Media=OFF")
        elif k == "hvac_power":
            params.append(f"{label}={v}")
        else:
            params.append(f"{label}={v}")
    return ", ".join(params)


# ═══════════════════════════════════════════════════
# Accept / Dismiss handlers
# ═══════════════════════════════════════════════════

def _handle_accept(username: str, card_id: str, display_name: str):
    try:
        from engine.habit_engine import HabitDemoEngine
        # Tab 2 is display + SQLite-only: _load_status reads habit_memory.db;
        # accept/dismiss are pure UPDATE statements. Drift detection runs in
        # Tab 1's ingest_signal_batch, not here. No LLM needed — pass None so
        # we don't eagerly construct an OpenAI client we never use.
        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        engine.scene_card_store.accept(card_id)
        engine.close()
        st.success(f"Scene card '{display_name}' accepted!")
    except Exception as e:
        st.error(f"Accept error: {e}")
        log.exception("Tab2 accept error")


def _handle_dismiss(username: str, card_id: str, display_name: str, status: str):
    try:
        from engine.habit_engine import HabitDemoEngine
        # Tab 2 is display + SQLite-only: _load_status reads habit_memory.db;
        # accept/dismiss are pure UPDATE statements. Drift detection runs in
        # Tab 1's ingest_signal_batch, not here. No LLM needed — pass None so
        # we don't eagerly construct an OpenAI client we never use.
        engine = HabitDemoEngine(username=username.lower(), llm_client=None)
        if status == "recommendation":
            engine.scene_card_store.reject_recommendation(card_id)
        else:
            engine.scene_card_store.dismiss_pending(card_id)
        engine.close()
        st.info(f"Scene card '{display_name}' dismissed.")
    except Exception as e:
        st.error(f"Dismiss error: {e}")
        log.exception("Tab2 dismiss error")
