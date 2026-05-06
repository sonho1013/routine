"""
Habit Memory Demo — Streamlit Simulator

Entry point: streamlit run simulator/app.py

Tab 1: User Profile Generator (学习阶段)
  - 信号输入 → signal_to_fact → ChromaDB → DBSCAN → Habit 检测

Tab 2: Smart Habit Recommender (推理阶段)
  - 上下文匹配 → ProactiveExecutor → 推荐卡片 → Accept/Reject
"""
import os
import sys
import streamlit as st

# 确保 project root 在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Page config ──
st.set_page_config(
    page_title="Habit Memory Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load custom CSS ──
CSS_PATH = os.path.join(os.path.dirname(__file__), "static", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Nav bar ──
st.html("""
<div style="
    background: linear-gradient(135deg, #0A1628 0%, #132035 60%, #0e2a3f 100%);
    padding: 0 24px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 2px solid #00BFC8;
    box-shadow: 0 2px 16px rgba(0,0,0,0.25);
    font-family: 'DM Sans', sans-serif;
">
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="
            width:32px; height:32px; border-radius:8px;
            background: linear-gradient(135deg, #00BFC8, #00636F);
            display:flex; align-items:center; justify-content:center;
            font-size:1rem;
        ">🧠</div>
        <span style="
            font-size:1.05rem; font-weight:700;
            color:#E8EDF3; letter-spacing:0.02em;
        ">Cockpit AI <span style="color:#00BFC8;">·</span> Habit Memory</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="
            background:rgba(0,191,200,0.1); border:1px solid rgba(0,191,200,0.2);
            border-radius:12px; padding:3px 10px;
            font-size:0.7rem; color:#00BFC8; font-weight:600;
        ">MS1 PoC</span>
        <span style="font-size:0.72rem; color:#8899AA;">Paranomix</span>
    </div>
</div>
""")

# ── Demo emergency switch ──
from panoramix_core.demo_mode import set_force_cache_runtime
from simulator.components.cache_status import render_cache_status_footer

_qp = st.query_params
_force = _qp.get("cache_only") == "1"
set_force_cache_runtime(_force)

# ── Tabs ──
tab1, tab2 = st.tabs([
    "📊 User Profile Generator",
    "💡 Smart Habit Recommender",
])

with tab1:
    from simulator import tab1_learning
    tab1_learning.render()

with tab2:
    from simulator import tab2_recommendation
    tab2_recommendation.render()

render_cache_status_footer()
