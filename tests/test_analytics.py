"""The analytics tool must agree with an independent pandas computation."""
import pandas as pd
import pytest

from src import config
from src.tools.analytics_tool import compute_stats, query_energy_stats


@pytest.fixture(scope="module")
def ev():
    df = pd.read_csv(config.EV_CSV, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def test_monthly_total_matches_raw(ev):
    raw = ev[(ev["date"] >= "2020-08-01") & (ev["date"] <= "2020-08-31")]["total_energy_kwh"].sum()
    r = compute_stats("ev", "energy_kwh", "sum", start_date="2020-08-01", end_date="2020-08-31")
    assert r["overall"] == pytest.approx(raw)


def test_ranking_by_node(ev):
    sub = ev[ev["date"].str.startswith("2020-08")]
    expected = sub.groupby("grid_node")["total_energy_kwh"].sum().idxmax()
    r = compute_stats("ev", "energy_kwh", "sum", group_by="grid_node",
                      start_date="2020-08-01", end_date="2020-08-31", top_n=1)
    assert r["top"][0][0] == expected


def test_day_count(ev):
    expected = ev[(ev["grid_node"] == "GRID_08A") & ev["date"].str.startswith("2020")]["date"].nunique()
    r = compute_stats("ev", "energy_kwh", "count", grid_node="grid_08a",
                      start_date="2020-01-01", end_date="2020-12-31")
    assert r["overall"] == expected


def test_solar_argmax_day():
    s = pd.read_csv(config.SOLAR_CSV, parse_dates=["timestamp"])
    s = s[s["timestamp"].dt.month == 7]
    expected = s.loc[s["ghi"].idxmax(), "timestamp"].strftime("%Y-%m-%d")
    r = compute_stats("solar", "peak_ghi", "max", group_by="date",
                      start_date="2023-07-01", end_date="2023-07-31", top_n=1)
    assert r["top"][0][0] == expected


@pytest.mark.parametrize("kwargs,msg", [
    ({"dataset": "wind", "metric": "x"}, "Unknown dataset"),
    ({"dataset": "ev", "metric": "peak_ghi"}, "not valid"),
    ({"dataset": "solar", "metric": "peak_ghi", "grid_node": "GRID_LV22"}, "only applies"),
    ({"dataset": "ev", "metric": "energy_kwh", "start_date": "2030-01-01"}, "No data"),
])
def test_errors_are_explicit(kwargs, msg):
    assert msg in compute_stats(**kwargs)["error"]


def test_tool_output_is_text():
    out = query_energy_stats.invoke({"dataset": "solar", "metric": "peak_ghi", "aggregation": "mean",
                                     "start_date": "2023-07-01", "end_date": "2023-07-31"})
    assert out.startswith("Analytics result") and "Overall mean" in out
