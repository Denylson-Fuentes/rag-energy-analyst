"""
Retrieval tool: gives the agent access to the Chroma vector store.
"""
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.retrieval.smart_retrieve import smart_retrieve, format_results_for_agent


# Module-level cache — vectorstore loads once per process
_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        _vectorstore = Chroma(
            persist_directory=os.path.join(PROJECT_ROOT, "data", "chroma_db"),
            embedding_function=embeddings,
        )
    return _vectorstore


@tool
def search_energy_data(query: str) -> str:
    """
    Search historical energy data (EV charging loads and solar irradiance).

    Use this tool when the user asks about past events, historical patterns,
    specific dates, or specific grid nodes.

    Examples of good queries:
    - "GRID_LV22 on 2020-08-11"
    - "highest solar irradiance day in Palm Springs"
    - "cloudy days with low solar output"
    - "most active EV charging day for GRID_08A"

    Returns: Formatted text with source citations for grounding your answer.
    """
    vs = get_vectorstore()
    results = smart_retrieve(query, vs, k=3)
    return format_results_for_agent(results)