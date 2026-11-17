"""
Analytics tool: exact aggregations over the full dataset.

Retrieval returns a handful of documents, so it cannot answer "which node used
the most energy in August 2020?" or "average peak GHI in July". This tool
computes those answers directly with pandas over daily summary tables.
"""
from typing import Literal

from langchain_core.tools import tool

from src.analytics.daily import ev_daily, solar_daily

EV_METRICS = ("energy_kwh", "charging_hours", "peak_hour_energy_kwh", "max_active_chargers")
SOLAR_METRICS = ("peak_ghi", "mean_cloud_cover", "daily_irradiation_kwh_m2")


def compute_stats(
    dataset: str,
    metric: str,
    aggregation: str = "sum",
    group_by: str = "none",
    start_date: str | None = None,
    end_date: str | None = None,
    grid_node: str | None = None,
    top_n: int = 5,
    ascending: bool = False,
) -> dict:
    """Pure function behind the tool. Returns a dict so it can be unit tested."""
    if dataset == "ev":
        df, valid = ev_daily(), EV_METRICS
    elif dataset == "solar":
        df, valid = solar_daily(), SOLAR_METRICS
    else:
        return {"error": f"Unknown dataset {dataset!r}. Use 'ev' or 'solar'."}
    if metric not in valid:
        return {"error": f"Metric {metric!r} not valid for {dataset}. Choose from {list(valid)}."}
    if group_by == "grid_node" and dataset != "ev":
        return {"error": "group_by='grid_node' only applies to the ev dataset."}

    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    if grid_node:
        if dataset != "ev":
            return {"error": "grid_node filter only applies to the ev dataset."}
        df = df[df["grid_node"] == grid_node.upper()]

    if df.empty:
        return {"error": "No data matches these filters.", "n_rows": 0}

    agg = {"count": "count"}.get(aggregation, aggregation)
    if agg not in ("sum", "mean", "max", "min", "count", "median"):
        return {"error": f"Unknown aggregation {aggregation!r}."}

    result = {
        "dataset": dataset, "metric": metric, "aggregation": aggregation,
        "filters": {"start_date": start_date, "end_date": end_date, "grid_node": grid_node},
        "n_days": int(df["date"].nunique()), "n_rows": int(len(df)),
        "overall": float(df[metric].agg(agg)),
    }
    if group_by != "none":
        series = df.groupby(group_by)[metric].agg(agg).sort_values(ascending=ascending)
        result["group_by"] = group_by
        result["n_groups"] = int(len(series))
        result["top"] = [(str(k), float(v)) for k, v in series.head(top_n).items()]
    return result


def format_stats(r: dict) -> str:
    if "error" in r:
        return f"Analytics error: {r['error']}"
    f = {k: v for k, v in r["filters"].items() if v}
    lines = [
        f"Analytics result ({r['dataset']}, {r['aggregation']} of daily {r['metric']}"
        + (f", filters {f}" if f else "") + "):",
        f"- Rows (node-days or days): {r['n_rows']}, distinct dates: {r['n_days']}",
        f"- Overall {r['aggregation']}: {r['overall']:.2f}",
    ]
    if "top" in r:
        lines.append(f"- Ranked by {r['group_by']} ({r['n_groups']} groups):")
        lines += [f"  {i}. {k}: {v:.2f}" for i, (k, v) in enumerate(r["top"], 1)]
    return "\n".join(lines)


@tool
def query_energy_stats(
    dataset: Literal["ev", "solar"],
    metric: Literal[
        "energy_kwh", "charging_hours", "peak_hour_energy_kwh", "max_active_chargers",
        "peak_ghi", "mean_cloud_cover", "daily_irradiation_kwh_m2",
    ],
    aggregation: Literal["sum", "mean", "max", "min", "count", "median"] = "sum",
    group_by: Literal["none", "grid_node", "date", "month", "year"] = "none",
    start_date: str | None = None,
    end_date: str | None = None,
    grid_node: str | None = None,
    top_n: int = 5,
    ascending: bool = False,
) -> str:
    """
    Compute exact statistics over the FULL dataset: totals, averages, counts,
    rankings, and "which day / node / month had the highest or lowest X".

    Use this instead of search_energy_data for any question spanning more than
    a couple of days (a month, a year, all nodes).

    Datasets and metrics (all metrics are per-day values first):
      ev    (EV charging, 10 grid nodes, 2019-04 to 2021-10): energy_kwh,
            charging_hours, peak_hour_energy_kwh, max_active_chargers
      solar (Palm Springs, every day of 2023): peak_ghi (W/m²),
            mean_cloud_cover (%), daily_irradiation_kwh_m2

    Args:
        dataset: "ev" or "solar".
        metric: one of the metrics above for that dataset.
        aggregation: sum | mean | max | min | count | median of the daily values.
        group_by: none | grid_node (ev only) | date | month | year. Results are
            ranked; use top_n=1 for "which X had the most".
        start_date / end_date: inclusive YYYY-MM-DD bounds (e.g. a month:
            2020-08-01 to 2020-08-31).
        grid_node: restrict to one node, e.g. "GRID_LV22" (ev only).
        top_n: how many ranked groups to return.
        ascending: True to rank lowest first.

    Example: which node used the most energy in Aug 2020 ->
        dataset="ev", metric="energy_kwh", aggregation="sum", group_by="grid_node",
        start_date="2020-08-01", end_date="2020-08-31", top_n=3
    """
    return format_stats(compute_stats(
        dataset, metric, aggregation, group_by, start_date, end_date, grid_node, top_n, ascending,
    ))

