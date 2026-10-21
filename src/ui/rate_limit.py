"""
Session-based query rate limiting.

Each browser session gets N queries. Prevents accidental Bedrock cost
blowup if the demo receives unexpected traffic.
"""
import streamlit as st


MAX_QUERIES = 20


def init_rate_limit():
    """Initialize the session counter if not already set."""
    if "query_count" not in st.session_state:
        st.session_state["query_count"] = 0


def get_query_count() -> int:
    """Return the number of queries used in this session."""
    return st.session_state.get("query_count", 0)


def get_remaining_queries() -> int:
    """Return how many queries remain."""
    return max(0, MAX_QUERIES - get_query_count())


def increment_query_count():
    """Increment the query counter by 1."""
    st.session_state["query_count"] = get_query_count() + 1


def is_rate_limited() -> bool:
    """True if the user has exhausted their query budget."""
    return get_query_count() >= MAX_QUERIES


def reset_rate_limit():
    """Reset the counter (used by 'clear chat' button)."""
    st.session_state["query_count"] = 0
