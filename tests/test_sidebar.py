"""
Smoke test for the sidebar — run with:
  streamlit run scripts/test_sidebar.py
"""
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.ui.sidebar import render_sidebar
from src.ui.rate_limit import init_rate_limit, increment_query_count

st.set_page_config(page_title="Sidebar Test", page_icon="⚡")
init_rate_limit()

mode = render_sidebar()

st.title("Sidebar smoke test")
st.write(f"Selected mode: **{mode}**")

st.markdown("### Simulate queries")
if st.button("Add 5 queries (to test the progress bar)"):
    for _ in range(5):
        increment_query_count()
    st.rerun()

st.write(f"Current count: {st.session_state.get('query_count', 0)}")
st.write(f"Messages in state: {st.session_state.get('messages', 'not initialized')}")