"""
EV charging prediction tool: runs the trained Random Forest model.
"""
import sys
import os
import joblib
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.tools import tool


_model = None
_ev_data = None

FEATURES = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'day_of_week', 'month', 'week_of_year',
    'active_chargers', 'unique_vehicles',
]


def load_assets():
    global _model, _ev_data
    if _model is None:
        _model = joblib.load(os.path.join(PROJECT_ROOT, "models", "ev_grid_rf.pkl"))
        _ev_data = pd.read_csv(
            os.path.join(PROJECT_ROOT, "data", "raw", "ev_grid_features.csv"),
            parse_dates=["timestamp"],
        )


@tool
def predict_ev_load(grid_node: str, date: str) -> str:
    """
    Predict hourly EV charging energy for a specific grid node on a given date.

    Use this tool when the user asks about EV charging predictions, expected
    loads, or forecasts for a specific node and date.

    Args:
        grid_node: Grid node identifier (e.g., "GRID_LV22", "GRID_08A")
        date: Date in YYYY-MM-DD format (e.g., "2020-08-11")

    Returns: Predicted vs actual energy for each charging hour.
    """
    load_assets()

    grid_node = grid_node.upper()
    target_date = pd.to_datetime(date).date()

    day_df = _ev_data[
        (_ev_data["grid_node"] == grid_node) &
        (_ev_data["timestamp"].dt.date == target_date)
    ].copy()

    if len(day_df) == 0:
        nodes = _ev_data["grid_node"].unique().tolist()
        return (
            f"No charging data for {grid_node} on {date}. "
            f"Available grid nodes: {', '.join(nodes)}"
        )

    day_df["predicted"] = _model.predict(day_df[FEATURES].values)

    total_actual = day_df["total_energy_kwh"].sum()
    total_pred = day_df["predicted"].sum()
    mae = (day_df["predicted"] - day_df["total_energy_kwh"]).abs().mean()

    return (
        f"EV charging prediction for {grid_node} on {date}:\n"
        f"- Total actual energy: {total_actual:.1f} kWh\n"
        f"- Total predicted energy: {total_pred:.1f} kWh\n"
        f"- MAE: {mae:.2f} kWh\n"
        f"- Charging hours: {len(day_df)}\n\n"
        f"Hourly breakdown:\n" +
        "\n".join(
            f"  {row['timestamp'].strftime('%H:%M')}: "
            f"actual {row['total_energy_kwh']:.1f} kWh, "
            f"predicted {row['predicted']:.1f} kWh"
            for _, row in day_df.iterrows()
        )
    )
    
@tool
def list_available_grid_nodes() -> str:
    """
    List all grid nodes available in the system, with their date ranges and data coverage.

    Use this tool when the user asks what grids are available, what data exists,
    or which nodes can be queried.
    """
    load_assets()

    lines = []
    for node in sorted(_ev_data['grid_node'].unique()):
        node_df = _ev_data[_ev_data['grid_node'] == node]
        min_date = node_df['timestamp'].min().date()
        max_date = node_df['timestamp'].max().date()
        n_days = node_df['timestamp'].dt.date.nunique()
        total_energy = node_df['total_energy_kwh'].sum()
        lines.append(
            f"- {node}: {n_days} days with data, "
            f"{min_date} to {max_date}, "
            f"total energy {total_energy:.1f} kWh"
        )

    return "Available grid nodes:\n" + "\n".join(lines)