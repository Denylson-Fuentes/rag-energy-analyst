"""
LangChain agent executor (v1 API).
"""
from langchain.agents import create_agent

from src.tools.retrieval_tool import search_energy_data
from src.tools.solar_forecast_tool import forecast_solar
from src.tools.ev_prediction_tool import predict_ev_load, list_available_grid_nodes
from src.agent.llm import get_llm
from src.agent.prompts import OPERATOR_PROMPT, ANALYST_PROMPT, TRADER_PROMPT


def build_executor(mode: str = "operator"):
    """Build the agent for a given mode using LangChain v1."""
    prompt_map = {
        "operator": OPERATOR_PROMPT,
        "analyst": ANALYST_PROMPT,
        "trader": TRADER_PROMPT,
    }
    system_prompt = prompt_map.get(mode, OPERATOR_PROMPT)

    tools = [search_energy_data, forecast_solar, predict_ev_load, list_available_grid_nodes]
    llm = get_llm()

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )
    return agent