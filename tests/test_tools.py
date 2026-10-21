"""
Smoke tests for the LangChain tools.

These verify that tools load their assets, accept valid inputs, and
return non-empty strings. They do NOT test model accuracy — that's
covered in the retraining scripts.
"""
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def test_solar_forecast_returns_string():
    from src.tools.solar_forecast_tool import forecast_solar
    result = forecast_solar.invoke({"date": "2023-07-15"})
    assert isinstance(result, str)
    assert len(result) > 100
    assert "W/m²" in result


def test_solar_forecast_handles_missing_date():
    from src.tools.solar_forecast_tool import forecast_solar
    result = forecast_solar.invoke({"date": "1990-01-01"})
    assert isinstance(result, str)
    assert "No solar data" in result


def test_ev_prediction_returns_string():
    from src.tools.ev_prediction_tool import predict_ev_load
    result = predict_ev_load.invoke({"grid_node": "GRID_08A", "date": "2019-11-18"})
    assert isinstance(result, str)
    assert "kWh" in result


def test_ev_prediction_handles_missing_date():
    from src.tools.ev_prediction_tool import predict_ev_load
    result = predict_ev_load.invoke({"grid_node": "GRID_08A", "date": "1990-01-01"})
    assert isinstance(result, str)
    assert "No charging data" in result


def test_list_nodes_returns_all():
    from src.tools.ev_prediction_tool import list_available_grid_nodes
    result = list_available_grid_nodes.invoke({})
    assert isinstance(result, str)
    assert "GRID_" in result
    assert "GRID_LV22" in result
    assert "GRID_08A" in result


def test_retrieval_returns_sources():
    from src.tools.retrieval_tool import search_energy_data
    result = search_energy_data.invoke({"query": "GRID_LV22 on 2020-08-11"})
    assert isinstance(result, str)
    assert "[Source 1]" in result