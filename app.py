"""
Streamlit entry point for the Energy Analyst RAG agent.

Run with:
    streamlit run app.py
"""
import os
import sys

# Ensure project root is on the path and is the working directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

# --- Page config ---
st.set_page_config(
    page_title="Energy Analyst — RAG Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure Chroma DB exists, build it if not
from src.config import CHROMA_DIR

if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    with st.spinner("First-time setup: building vector store... (~2 min)"):
        from src.ingestion.build_vectorstore import build_vectorstore
        build_vectorstore()

from src.ui.chat import handle_chat_input, init_chat, render_chat_history
from src.ui.rate_limit import init_rate_limit
from src.ui.sidebar import render_sidebar

# --- Session state init ---
init_rate_limit()
init_chat()

# --- Sidebar (returns selected mode) ---
mode = render_sidebar()

# --- Main content ---
st.title("⚡ Energy Analyst")
st.caption(
    "Ask questions about solar forecasts, EV charging loads, or historical grid data. "
    "Every answer is grounded in your trained models and real datasets."
)

# Optional: pre-seed sample questions as clickable buttons on first load
if len(st.session_state.get("messages", [])) == 0:
    st.markdown("### Try a question")
    sample_cols = st.columns(3)

    samples = [
        "What was the total EV charging energy for GRID_LV22 on 2020-08-11?",
        "What's the solar forecast for 2023-07-15?",
        "Is solar generation higher in summer or winter?",
    ]

    for col, q in zip(sample_cols, samples, strict=False):
        with col:
            if st.button(q, use_container_width=True, key=f"sample_{hash(q)}"):
                st.session_state["messages"].append({"role": "user", "content": q})
                st.rerun()

    st.markdown("---")

# --- Chat history ---
render_chat_history()

# --- Chat input (also handles agent invocation) ---
handle_chat_input(mode)