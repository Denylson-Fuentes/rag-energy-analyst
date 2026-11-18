"""
Daily summary tables built from the raw CSVs. Shared by the analytics tool.
"""
from functools import lru_cache

import pandas as pd

from src import config


@lru_cache(maxsize=1)
def ev_daily() -> pd.DataFrame:
    """One row per (grid_node, date)."""
    df = pd.read_csv(config.EV_CSV, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.strftime("%H:%M")
    g = df.groupby(["grid_node", "date"])
    out = pd.DataFrame({
        "energy_kwh": g["total_energy_kwh"].sum(),
        "charging_hours": g.size(),
        "peak_hour_energy_kwh": g["total_energy_kwh"].max(),
        "max_active_chargers": g["active_chargers"].max(),
    }).reset_index()
    peak_idx = df.loc[g["total_energy_kwh"].idxmax(), ["grid_node", "date", "hour"]]
    out = out.merge(peak_idx.rename(columns={"hour": "peak_hour"}), on=["grid_node", "date"])
    out["month"] = out["date"].str[:7]
    out["year"] = out["date"].str[:4]
    return out


@lru_cache(maxsize=1)
def solar_daily() -> pd.DataFrame:
    """One row per date (Palm Springs)."""
    df = pd.read_csv(config.SOLAR_CSV, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.strftime("%H:%M")
    g = df.groupby("date")
    out = pd.DataFrame({
        "peak_ghi": g["ghi"].max(),
        "mean_cloud_cover": g["total_cloud_cover"].mean(),
        # 30-minute readings: W/m² * 0.5 h / 1000 = kWh/m²
        "daily_irradiation_kwh_m2": g["ghi"].sum() * 0.5 / 1000,
    }).reset_index()
    peak_idx = df.loc[g["ghi"].idxmax(), ["date", "hour"]]
    out = out.merge(peak_idx.rename(columns={"hour": "peak_time"}), on="date")
    out["month"] = out["date"].str[:7]
    out["year"] = out["date"].str[:4]
    return out
