"""
Suggestion Card — 推荐动作卡片

展示 ProactiveExecutor 的推荐结果，含 accept/reject 按钮。
"""
import streamlit as st
from typing import Optional


def render_suggestion_card(
    habit_text: str,
    scene_name: Optional[str],
    match_confidence: float,
    combined_confidence: float,
    context_distance: float,
    parsed_actions: dict,
    habit_id: str,
    key_prefix: str = "sug",
) -> Optional[str]:
    """
    渲染单张推荐卡片。

    Returns:
        "accept" | "reject" | None
    """
    conf_pct = int(combined_confidence * 100)
    match_pct = int(match_confidence * 100)

    # 置信度颜色
    if conf_pct >= 70:
        conf_color = "#00C896"
    elif conf_pct >= 40:
        conf_color = "#F5A623"
    else:
        conf_color = "#E74C3C"

    st.markdown(f"""
    <div class="suggestion-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span class="scene-tag">{scene_name or 'Unknown Scene'}</span>
            <span style="color:{conf_color}; font-weight:600; font-size:1.1rem;">
                {conf_pct}%
            </span>
        </div>
        <div style="margin:0.5rem 0; font-size:0.95rem;">
            {habit_text}
        </div>
        <div style="font-size:0.75rem; color:#8899AA;">
            Match: {match_pct}% &bull; Distance: {context_distance:.3f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 动作详情
    if parsed_actions:
        with st.expander("Parsed Actions", expanded=False):
            for k, v in parsed_actions.items():
                st.text(f"  {k}: {v}")

    # Accept / Reject 按钮
    col_accept, col_reject, _ = st.columns([1, 1, 2])
    action = None
    with col_accept:
        if st.button("Accept", key=f"{key_prefix}_{habit_id}_accept",
                      type="primary"):
            action = "accept"
    with col_reject:
        if st.button("Reject", key=f"{key_prefix}_{habit_id}_reject"):
            action = "reject"

    return action


def render_no_recommendations(candidates_checked: int, threshold: float):
    """无推荐时的占位展示"""
    st.info(
        f"No matching habits found.\n\n"
        f"Checked {candidates_checked} candidates "
        f"(threshold = {threshold:.2f})"
    )
