"""Sidebar footer indicator: `live ✓` / `CACHE ⚠`."""
import streamlit as st
from panoramix_core.demo_mode import is_force_cache


def render_cache_status_footer() -> None:
    label = "CACHE ⚠" if is_force_cache() else "live ✓"
    color = "#cc8800" if is_force_cache() else "#888888"
    st.sidebar.markdown(
        f"<div style='position:fixed;bottom:8px;left:8px;"
        f"font-size:11px;color:{color};opacity:0.7'>{label}</div>",
        unsafe_allow_html=True,
    )
