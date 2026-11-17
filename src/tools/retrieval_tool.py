"""
Retrieval tool: gives the agent access to the Chroma vector store.
"""
import threading

from langchain_chroma import Chroma
from langchain_core.tools import tool

from src import config
from src.retrieval.embeddings import get_embeddings
from src.retrieval.smart_retrieve import RETRIEVERS, format_results_for_agent

# Module-level cache — vectorstore loads once per process
_vectorstore = None
_lock = threading.Lock()


def get_vectorstore():
    global _vectorstore
    with _lock:  # concurrent first use (eval workers, Streamlit sessions) races chromadb's client init
        if _vectorstore is None:
            _vectorstore = Chroma(
                persist_directory=config.CHROMA_DIR,
                embedding_function=get_embeddings(),
            )
    return _vectorstore


# Original (v1) tool description, kept verbatim for the baseline agent.
_DESCRIPTION_V1 = """
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

_DESCRIPTION = """
Search historical energy data documents (one document per EV grid-node-day and
one per Palm Springs solar day).

Use this tool to look up specific days: a grid node's charging on a date, or
solar irradiance / cloud cover on a date. Include exact dates (YYYY-MM-DD) and
grid node IDs in the query; you may include several to compare them.

Examples of good queries:
- "GRID_LV22 on 2020-08-11"
- "GRID_08A on 2019-11-18 and 2019-11-19"
- "solar irradiance 2023-07-15"

Do NOT use this for totals, averages, rankings, or "which day/node had the
most" questions over a month or year; it only returns a few documents.

Returns: Formatted text with [Source N] citations.
"""


def make_search_tool(version: str = "v2", k: int = 3):
    retrieve = RETRIEVERS[version]

    description = _DESCRIPTION_V1 if version == "v1" else _DESCRIPTION

    @tool("search_energy_data", description=description)
    def search_energy_data(query: str) -> str:
        results = retrieve(query, get_vectorstore(), k=k)
        return format_results_for_agent(results)

    return search_energy_data


search_energy_data = make_search_tool("v2")
