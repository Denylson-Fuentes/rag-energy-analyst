"""
Unit tests for session-based rate limiting.

Uses a mock session_state so tests can run without Streamlit running.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


class SessionStateMock(dict):
    """Minimal dict subclass that mimics st.session_state attribute access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def _setup():
    """Fresh mock session state for each test."""
    import streamlit as st
    st.session_state = SessionStateMock()
    from src.ui import rate_limit
    # Reset module state in case it imported a stale session_state
    rate_limit.st.session_state = st.session_state
    return rate_limit


def test_init_sets_zero():
    rl = _setup()
    rl.init_rate_limit()
    assert rl.get_query_count() == 0


def test_increment():
    rl = _setup()
    rl.init_rate_limit()
    rl.increment_query_count()
    rl.increment_query_count()
    assert rl.get_query_count() == 2
    assert rl.get_remaining_queries() == rl.MAX_QUERIES - 2


def test_rate_limited_at_max():
    rl = _setup()
    rl.init_rate_limit()
    for _ in range(rl.MAX_QUERIES):
        rl.increment_query_count()
    assert rl.is_rate_limited() is True
    assert rl.get_remaining_queries() == 0


def test_not_rate_limited_below_max():
    rl = _setup()
    rl.init_rate_limit()
    for _ in range(rl.MAX_QUERIES - 1):
        rl.increment_query_count()
    assert rl.is_rate_limited() is False
    assert rl.get_remaining_queries() == 1


def test_reset():
    rl = _setup()
    rl.init_rate_limit()
    for _ in range(5):
        rl.increment_query_count()
    rl.reset_rate_limit()
    assert rl.get_query_count() == 0
    assert rl.is_rate_limited() is False