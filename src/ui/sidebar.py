"""
Sidebar: mode selector, rate limit display, clear-chat button, footer.
"""
import streamlit as st

from src.ui.rate_limit import (
    MAX_QUERIES,
    get_query_count,
    get_remaining_queries,
    reset_rate_limit,
)


MODE_DESCRIPTIONS = {
    "Operator": "Historical lookups and forecasts grounded in data.",
    "Analyst": "Explanations, comparisons, and trend insights.",
    "Trader": "Supply/demand signals and price implications.",
}


def render_sidebar() -> str:
    """
    Render the sidebar and return the selected mode as a lowercase string
    (matches the keys expected by build_executor: 'operator', 'analyst', 'trader').
    """
    with st.sidebar:
        st.title("⚡ Energy Analyst")
        st.caption("RAG agent for grid operations, analytics, and trading signals.")

        st.markdown("---")

        st.subheader("Mode")
        mode = st.radio(
            "Select agent mode",
            options=["Operator", "Analyst", "Trader"],
            index=0,
            label_visibility="collapsed",
        )
        st.caption(MODE_DESCRIPTIONS[mode])

        st.markdown("---")

        st.subheader("Query Budget")
        used = get_query_count()
        remaining = get_remaining_queries()

        st.progress(
            min(used / MAX_QUERIES, 1.0),
            text=f"{used} / {MAX_QUERIES} queries used",
        )

        if remaining == 0:
            st.error("Query limit reached. Refresh the page to start a new session.")
        elif remaining <= 3:
            st.warning(f"Only {remaining} queries remaining this session.")

        st.markdown("---")

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            reset_rate_limit()
            st.rerun()

        st.markdown("---")

        st.caption(
            "Built by **Denylson Fuentes**\n\n"
            "[Portfolio](https://denylson-fuentes.netlify.app) · "
            "[GitHub](https://github.com/Denylson-Fuentes)"
        )

    return mode.lower()