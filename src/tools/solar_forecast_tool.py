"""
Solar forecast tool: runs the trained Random Forest model.
"""
import sys
import os
import joblib
import numpy as np
import pandas as pd
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from langchain_core.tools import tool

_model = None
_scaler = None
_solar_data = None

FEATURES = [
    'cos_zenith', 'solar_elevation', 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'total_cloud_cover', 'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h',
    'ghi_rolling_mean_1h', 'ghi_rolling_std_1h', 'csi_lag_1h', 'csi_lag_2h',
    'cloud_type', 'cloud_0', 'cloud_1', 'cloud_2', 'cloud_3', 'cloud_4',
    'cloud_5', 'cloud_6', 'cloud_7', 'cloud_8', 'cloud_9',
]

HORIZON_ROWS = 12  # 6 hours ahead in 30-min data


def load_assets():
    global _model, _scaler, _solar_data
    if _model is None:
        _model = joblib.load(os.path.join(PROJECT_ROOT, "models", "solar_p50_v2.pkl"))
        _scaler = joblib.load(os.path.join(PROJECT_ROOT, "models", "solar_scaler_v2.pkl"))
        _solar_data = pd.read_csv(
            os.path.join(PROJECT_ROOT, "data", "raw", "palm_springs_features.csv"),
            parse_dates=["timestamp"],
        ).sort_values("timestamp").reset_index(drop=True)


@tool
def forecast_solar(date: str) -> str:
    """
    Forecast solar irradiance (GHI) for Palm Springs, CA 6 hours ahead using the trained model.

    Use this tool when the user asks about solar forecasts, expected generation,
    or future solar output.

    Args:
        date: Date in YYYY-MM-DD format (e.g., "2023-07-15")

    Returns: Forecast summary with peak GHI and hourly predictions (6h horizon).
    """
    load_assets()

    target_date = pd.to_datetime(date).date()
    day_mask = _solar_data["timestamp"].dt.date == target_date
    day_df = _solar_data[day_mask].copy()

    if len(day_df) == 0:
        return (
            f"No solar data available for {date}. "
            f"Available range: {_solar_data['timestamp'].min().date()} to {_solar_data['timestamp'].max().date()}"
        )

    # Make predictions for each row in the day
    X = _scaler.transform(day_df[FEATURES].values)
    day_df["predicted"] = _model.predict(X)

    # The model predicts GHI 6 hours ahead (12 rows = 6h in 30-min data).
    # For display: prediction at row i corresponds to actual at row i+12.
    # Add a "target" column = actual GHI 6 hours later.
    day_df = day_df.reset_index(drop=True)
    day_df["actual_6h_ahead"] = day_df["ghi"].shift(-HORIZON_ROWS)
    valid = day_df.dropna(subset=["actual_6h_ahead"])

    if len(valid) == 0:
        return (
            f"Not enough future data to evaluate 6-hour-ahead forecasts for {date}. "
            f"This usually means {date} is near the end of the available data."
        )

    # Report peak prediction
    peak_idx = valid["predicted"].idxmax()
    peak_time = valid.loc[peak_idx, "timestamp"].strftime("%H:%M")
    peak_value = valid.loc[peak_idx, "predicted"]

    # Report peak actual (6h ahead)
    actual_peak_idx = valid["actual_6h_ahead"].idxmax()
    actual_peak_time = valid.loc[actual_peak_idx, "timestamp"].strftime("%H:%M")
    actual_peak_value = valid.loc[actual_peak_idx, "actual_6h_ahead"]

    # MAE
    mae = np.mean(np.abs(valid["predicted"] - valid["actual_6h_ahead"]))

    return (
        f"Solar forecast for {date} (Palm Springs, 6-hour-ahead):\n"
        f"- Peak predicted GHI: {peak_value:.0f} W/m² (forecast issued at {peak_time})\n"
        f"- Peak actual GHI 6h ahead: {actual_peak_value:.0f} W/m² (at {actual_peak_time})\n"
        f"- MAE: {mae:.1f} W/m²\n"
        f"- Forecast points: {len(valid)}\n\n"
        f"Sample predictions (issued at time → predicted 6h ahead → actual 6h ahead):\n" +
        "\n".join(
            f"  {row['timestamp'].strftime('%H:%M')} → {row['predicted']:.0f} → {row['actual_6h_ahead']:.0f} W/m²"
            for _, row in valid.iloc[::4].iterrows()
        )
    )