"""
LangChain agent executor (v1 API).

Agent variants (used by the app and by the evaluation harness):

* rag_only - classic RAG: the LLM plus the v1 retriever, nothing else.
* agent_v1 - the originally shipped agent: 4 tools, v1 retriever, v1 prompt.
* agent_v2 - eval-driven version: v2 retriever, analytics tool, v2 prompt.
"""
from langchain.agents import create_agent

from src.agent.llm import get_llm
from src.agent.prompts import ANALYST_PROMPT, OPERATOR_PROMPT, OPERATOR_PROMPT_V2, TRADER_PROMPT
from src.tools.analytics_tool import query_energy_stats
from src.tools.ev_prediction_tool import list_available_grid_nodes, predict_ev_load
from src.tools.retrieval_tool import make_search_tool
from src.tools.solar_forecast_tool import forecast_solar

RAG_ONLY_PROMPT = """You are an assistant for energy grid data. Use the search_energy_data
tool to retrieve documents and answer only from them, citing [Source N]. If the documents
do not contain the answer, say you don't have the data."""

VARIANTS = ("rag_only", "agent_v1", "agent_v2")


def get_tools(variant: str):
    if variant == "rag_only":
        return [make_search_tool("v1")]
    if variant == "agent_v1":
        return [make_search_tool("v1"), forecast_solar, predict_ev_load, list_available_grid_nodes]
    if variant == "agent_v2":
        return [
            make_search_tool("v2"), query_energy_stats, forecast_solar,
            predict_ev_load, list_available_grid_nodes,
        ]
    raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")


def build_executor(mode: str = "operator", variant: str = "agent_v2", llm=None):
    """Build the agent for a given UI mode and variant."""
    if variant == "rag_only":
        system_prompt = RAG_ONLY_PROMPT
    elif mode == "operator":
        system_prompt = OPERATOR_PROMPT_V2 if variant == "agent_v2" else OPERATOR_PROMPT
    else:
        system_prompt = {"analyst": ANALYST_PROMPT, "trader": TRADER_PROMPT}.get(mode, OPERATOR_PROMPT)

    tools = get_tools(variant)
    if variant == "agent_v2" and mode != "operator":
        system_prompt += (
            "\n\nAlso available: query_energy_stats for totals, averages and rankings over "
            "months or years, and list_available_grid_nodes for data coverage."
        )

    return create_agent(llm or get_llm(), tools, system_prompt=system_prompt)
