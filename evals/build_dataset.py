"""
Generate the evaluation set: evals/data/questions.jsonl

Gold answers are computed directly from the raw CSVs with pandas, independently
of the agent's tool code, so a bug in a tool cannot also "fix" its own gold.
The two exceptions are the model-output questions (forecast / prediction),
whose ground truth *is* the deterministic model output; for those the eval
measures routing and faithful reporting, not forecast skill.

Generation is seeded, so the file is reproducible:
    python -m evals.build_dataset
"""
import json
import os
import random
import re

import pandas as pd

from src import config

OUT = os.path.join(os.path.dirname(__file__), "data", "questions.jsonl")
SEED = 7

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


# --------------------------------------------------------------------------- gold tables
def _ev():
    df = pd.read_csv(config.EV_CSV, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def _solar():
    df = pd.read_csv(config.SOLAR_CSV, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df


def nl_date(iso: str, style: int) -> str:
    d = pd.Timestamp(iso)
    m = MONTH_NAMES[d.month - 1]
    return [
        f"{m} {d.day}, {d.year}",
        f"{d.day} {m} {d.year}",
        f"{m[:3]} {d.day} {d.year}",
        f"{d.month}/{d.day}/{d.year}",
    ][style % 4]


def ev_doc(node, date):
    return f"ev_grid|{node}|{date}"


def solar_doc(date):
    return f"solar|{date}"


def item(category, question, answer_type, gold, *, tol=None, docs=(), tools=(), oracle=(), note=None):
    return {
        "category": category,
        "question": question,
        "answer_type": answer_type,
        "gold": gold,
        "abs_tol": tol,
        "gold_doc_ids": list(docs),
        "retrieval_applicable": bool(docs),
        "expected_tools": list(tools),
        "oracle_calls": list(oracle),
        "note": note,
    }


def search_call(q):
    return {"tool": "search_energy_data", "args": {"query": q}}


def stats_call(**args):
    return {"tool": "query_energy_stats", "args": args}


# --------------------------------------------------------------------------- builders
def build():
    rng = random.Random(SEED)
    ev, solar = _ev(), _solar()

    ev_day = ev.groupby(["grid_node", "date"]).agg(
        energy=("total_energy_kwh", "sum"), hours=("total_energy_kwh", "size"),
    ).reset_index()
    sol_day = solar.groupby("date").agg(
        peak=("ghi", "max"), cloud=("total_cloud_cover", "mean"),
    ).reset_index()

    nodes = sorted(ev["grid_node"].unique())
    items = []

    # Sample EV node-days stratified by node so small nodes are represented.
    def sample_ev_days(n, min_hours=1):
        pool = ev_day[ev_day["hours"] >= min_hours]
        rows = []
        per_node = {nd: pool[pool["grid_node"] == nd].sample(frac=1, random_state=rng.randint(0, 10**6))
                    for nd in nodes}
        i = 0
        while len(rows) < n:
            nd = nodes[i % len(nodes)]
            sub = per_node[nd]
            if len(sub):
                rows.append(sub.iloc[0])
                per_node[nd] = sub.iloc[1:]
            i += 1
        return rows

    used_ev, used_sol = set(), set()

    # 1. EV daily total (30)
    tpl = [
        "What was the total EV charging energy for {n} on {d}?",
        "How many kWh were charged at grid node {n} on {d}?",
        "Total charging energy at {n} on {d}?",
        "On {d}, how much energy did EV chargers on {n} deliver in total?",
    ]
    for i, r in enumerate(sample_ev_days(30)):
        used_ev.add((r.grid_node, r.date))
        q = tpl[i % 4].format(n=r.grid_node, d=r.date)
        items.append(item("ev_lookup", q, "number", round(float(r.energy), 2), tol=0.1,
                          docs=[ev_doc(r.grid_node, r.date)], tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"{r.grid_node} on {r.date}")]))

    # 2. EV peak hour (15) - only days with a unique peak hour and >= 3 hours
    hourly = ev.copy()
    hourly["hour"] = hourly["timestamp"].dt.strftime("%H:00")
    cands = []
    for (nd, dt), g in hourly.groupby(["grid_node", "date"]):
        if len(g) >= 3 and (g["total_energy_kwh"] == g["total_energy_kwh"].max()).sum() == 1 \
                and (nd, dt) not in used_ev:
            cands.append((nd, dt, g.loc[g["total_energy_kwh"].idxmax(), "hour"]))
    rng.shuffle(cands)
    tpl = [
        "At what hour did {n} have its highest charging energy on {d}?",
        "Which hour was the charging peak for {n} on {d}?",
        "When did EV charging peak at {n} on {d}?",
    ]
    for i, (nd, dt, hr) in enumerate(cands[:15]):
        used_ev.add((nd, dt))
        items.append(item("ev_peak_hour", tpl[i % 3].format(n=nd, d=dt), "time", hr,
                          docs=[ev_doc(nd, dt)], tools=["search_energy_data"],
                          oracle=[search_call(f"{nd} on {dt}")]))

    # 3. Solar peak GHI (25)
    sol_dates = list(sol_day["date"])
    rng.shuffle(sol_dates)
    tpl = [
        "What was the peak GHI in Palm Springs on {d}?",
        "What was the maximum solar irradiance on {d}?",
        "Peak solar irradiance (GHI) in Palm Springs on {d}?",
        "How strong was the sunshine at its peak on {d} in Palm Springs (GHI)?",
    ]
    sol_idx = sol_day.set_index("date")
    for i, d in enumerate(sol_dates[:25]):
        used_sol.add(d)
        items.append(item("solar_lookup", tpl[i % 4].format(d=d), "number", float(sol_idx.loc[d, "peak"]),
                          tol=1.0, docs=[solar_doc(d)], tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"solar irradiance {d}")]))

    # 4. Solar mean cloud cover (10)
    tpl = ["What was the mean cloud cover in Palm Springs on {d}?",
           "How cloudy was it on average in Palm Springs on {d}?"]
    for i, d in enumerate(sol_dates[25:35]):
        used_sol.add(d)
        items.append(item("solar_cloud", tpl[i % 2].format(d=d), "number",
                          round(float(sol_idx.loc[d, "cloud"]), 1), tol=0.6,
                          docs=[solar_doc(d)], tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"solar cloud cover {d}")]))

    # 5. Natural-language dates (15): 8 EV, 7 solar
    for i, r in enumerate(sample_ev_days(40)):
        if len([x for x in items if x["category"] == "nl_date"]) >= 8:
            break
        if (r.grid_node, r.date) in used_ev:
            continue
        used_ev.add((r.grid_node, r.date))
        q = f"How much EV charging energy was recorded at {r.grid_node} on {nl_date(r.date, i)}?"
        items.append(item("nl_date", q, "number", round(float(r.energy), 2), tol=0.1,
                          docs=[ev_doc(r.grid_node, r.date)], tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"{r.grid_node} on {r.date}")]))
    for i, d in enumerate(sol_dates[35:42]):
        used_sol.add(d)
        q = f"What was the peak solar irradiance in Palm Springs on {nl_date(d, i + 1)}?"
        items.append(item("nl_date", q, "number", float(sol_idx.loc[d, "peak"]), tol=1.0,
                          docs=[solar_doc(d)], tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"solar irradiance {d}")]))

    # 6. Comparisons (15): 8 EV same node two dates, 7 solar two dates
    busy = ev_day.groupby("grid_node").size()
    busy_nodes = list(busy[busy > 100].index)
    n_ev_cmp = 0
    while n_ev_cmp < 8:
        nd = busy_nodes[n_ev_cmp % len(busy_nodes)]
        sub = ev_day[ev_day["grid_node"] == nd].sample(2, random_state=rng.randint(0, 10**6))
        a, b = sub.iloc[0], sub.iloc[1]
        if abs(a.energy - b.energy) < 1:
            continue
        q = (f"Compare total EV charging energy at {nd} on {a.date} and {b.date}. "
             "Which day was higher, and what were the two totals?")
        items.append(item("comparison", q, "numbers_all", [round(float(a.energy), 2), round(float(b.energy), 2)],
                          tol=0.1, docs=[ev_doc(nd, a.date), ev_doc(nd, b.date)],
                          tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"{nd} on {a.date} and {b.date}")]))
        n_ev_cmp += 1
    for i in range(7):
        d1, d2 = sol_dates[42 + 2 * i], sol_dates[43 + 2 * i]
        q = f"Was peak GHI in Palm Springs higher on {d1} or on {d2}? Give both values."
        items.append(item("comparison", q, "numbers_all",
                          [float(sol_idx.loc[d1, "peak"]), float(sol_idx.loc[d2, "peak"])], tol=1.0,
                          docs=[solar_doc(d1), solar_doc(d2)], tools=["search_energy_data", "query_energy_stats"],
                          oracle=[search_call(f"solar irradiance {d1} and {d2}")]))

    # 7. Aggregation / superlatives (20)
    ev_day["month"] = ev_day["date"].str[:7]
    sol_day["month"] = sol_day["date"].str[:7]
    ev_months = sorted(ev_day["month"].unique())
    rng.shuffle(ev_months)

    def month_bounds(ym):
        start = pd.Timestamp(ym + "-01")
        end = start + pd.offsets.MonthEnd(0)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    k = 0
    for ym in ev_months:  # which node used most energy in month (5)
        tot = ev_day[ev_day["month"] == ym].groupby("grid_node")["energy"].sum().sort_values(ascending=False)
        if len(tot) < 3 or tot.iloc[0] - tot.iloc[1] < 5:
            continue
        s, e = month_bounds(ym)
        mname = f"{MONTH_NAMES[int(ym[5:]) - 1]} {ym[:4]}"
        items.append(item("aggregation", f"Which grid node had the highest total EV charging energy in {mname}?",
                          "text", tot.index[0], tools=["query_energy_stats"],
                          oracle=[stats_call(dataset="ev", metric="energy_kwh", aggregation="sum",
                                             group_by="grid_node", start_date=s, end_date=e, top_n=3)]))
        k += 1
        if k == 5:
            break
    for ym in ev_months[10:15]:  # total energy across all nodes in month (5)
        s, e = month_bounds(ym)
        tot = float(ev_day[ev_day["month"] == ym]["energy"].sum())
        mname = f"{MONTH_NAMES[int(ym[5:]) - 1]} {ym[:4]}"
        items.append(item("aggregation", f"What was the total EV charging energy across all grid nodes in {mname}?",
                          "number", round(tot, 2), tol=max(0.5, tot * 0.005), tools=["query_energy_stats"],
                          oracle=[stats_call(dataset="ev", metric="energy_kwh", aggregation="sum",
                                             start_date=s, end_date=e)]))
    sol_months = [f"2023-{m:02d}" for m in range(1, 13)]
    rng.shuffle(sol_months)
    k = 0
    for ym in sol_months:  # which day had highest peak GHI in month (4)
        sub = sol_day[sol_day["month"] == ym].sort_values("peak", ascending=False)
        if sub.iloc[0].peak - sub.iloc[1].peak < 1:
            continue
        s, e = month_bounds(ym)
        items.append(item("aggregation",
                          f"Which day in {MONTH_NAMES[int(ym[5:]) - 1]} 2023 had the highest peak GHI in Palm Springs?",
                          "date", sub.iloc[0].date, tools=["query_energy_stats"],
                          oracle=[stats_call(dataset="solar", metric="peak_ghi", aggregation="max",
                                             group_by="date", start_date=s, end_date=e, top_n=3)]))
        k += 1
        if k == 4:
            break
    for ym in sol_months[6:9]:  # average peak GHI in month (3)
        s, e = month_bounds(ym)
        val = float(sol_day[sol_day["month"] == ym]["peak"].mean())
        items.append(item("aggregation",
                          f"What was the average daily peak GHI in Palm Springs in {MONTH_NAMES[int(ym[5:]) - 1]} 2023?",
                          "number", round(val, 1), tol=1.0, tools=["query_energy_stats"],
                          oracle=[stats_call(dataset="solar", metric="peak_ghi", aggregation="mean",
                                             start_date=s, end_date=e)]))
    for nd, yr in [("GRID_LV22", "2020"), ("GRID_LV11", "2019"), ("GRID_08A", "2020")]:  # day counts (3)
        cnt = int(((ev_day["grid_node"] == nd) & (ev_day["date"].str[:4] == yr)).sum())
        items.append(item("aggregation", f"On how many days in {yr} did {nd} record any EV charging?",
                          "number", cnt, tol=0.0, tools=["query_energy_stats"],
                          oracle=[stats_call(dataset="ev", metric="energy_kwh", aggregation="count",
                                             grid_node=nd, start_date=f"{yr}-01-01", end_date=f"{yr}-12-31")]))

    # 8. Forecast / prediction tools (15). Gold = deterministic model output.
    from src.tools.ev_prediction_tool import predict_ev_load
    from src.tools.solar_forecast_tool import forecast_solar

    fc_dates = [d for d in sol_dates[60:] if d < "2023-12-31"][:10]
    for i, d in enumerate(fc_dates):
        out = forecast_solar.invoke({"date": d})
        m = re.search(r"MAE: ([\d.]+)", out)
        if not m:
            continue
        q = [f"What's the solar forecast for {d}, and how accurate was it (MAE)?",
             f"Run the 6-hour-ahead solar model for {d}. What was its mean absolute error?"][i % 2]
        items.append(item("forecast", q, "number", float(m[1]), tol=0.15, tools=["forecast_solar"],
                          oracle=[{"tool": "forecast_solar", "args": {"date": d}}],
                          note="gold is the model's own MAE; measures routing + faithful reporting"))
    for r in sample_ev_days(60, min_hours=3)[-5:]:
        out = predict_ev_load.invoke({"grid_node": r.grid_node, "date": r.date})
        m = re.search(r"Total predicted energy: ([\d.]+)", out)
        items.append(item("forecast",
                          f"What total EV load does the model predict for {r.grid_node} on {r.date}?",
                          "number", float(m[1]), tol=0.15, tools=["predict_ev_load"],
                          oracle=[{"tool": "predict_ev_load", "args": {"grid_node": r.grid_node, "date": r.date}}],
                          note="gold is the model's own prediction"))

    # 9. Inventory (5)
    items.append(item("inventory", "What grid nodes are available?", "text_all", nodes,
                      tools=["list_available_grid_nodes"], oracle=[{"tool": "list_available_grid_nodes", "args": {}}]))
    items.append(item("inventory", "How many EV grid nodes are in the dataset?", "number", len(nodes), tol=0.0,
                      tools=["list_available_grid_nodes", "query_energy_stats"],
                      oracle=[{"tool": "list_available_grid_nodes", "args": {}}]))
    for nd in ["GRID_LV22", "GRID_04A", "GRID_LV31"]:
        sub = ev_day[ev_day["grid_node"] == nd]["date"]
        items.append(item("inventory", f"What is the first and last date with EV charging data for {nd}?",
                          "dates_all", [sub.min(), sub.max()], tools=["list_available_grid_nodes", "query_energy_stats"],
                          oracle=[{"tool": "list_available_grid_nodes", "args": {}}]))

    # 10. Unanswerable / out of scope (15)
    unans = [
        ("What was the total EV charging energy for GRID_LV22 on 2018-03-14?", "date before EV data starts"),
        ("How many kWh were charged at GRID_08A on 2022-06-01?", "date after EV data ends"),
        ("What was the total charging energy at GRID_LV22 on 2025-01-10?", "future date"),
        ("What was the peak GHI in Palm Springs on 2022-07-04?", "solar data is 2023 only"),
        ("What was the peak solar irradiance on 2024-03-01?", "solar data is 2023 only"),
        ("What was the total EV charging energy for GRID_ZZ9 on 2020-08-11?", "node does not exist"),
        ("How much energy did GRID_LV99 use on 2020-05-05?", "node does not exist"),
        ("What was the peak GHI in Phoenix on 2023-07-15?", "location not in data"),
        ("What was the wholesale electricity price in California on 2020-08-11?", "prices not in data"),
        ("What was the EV charging energy at GRID_LV22 on 2020-02-30?", "invalid date"),
        ("How much wind generation was there in Palm Springs on 2023-05-01?", "wind generation not in data"),
        ("What was the battery state of charge at GRID_LV11 on 2020-03-03?", "metric not in data"),
        ("What was the peak GHI in Palm Springs on 2023-13-01?", "invalid date"),
        ("How many electric buses charged at GRID_08A on 2019-11-18?", "vehicle type not in data"),
        ("What will the total EV charging energy at GRID_LV21 be next Tuesday?", "no future data/date"),
    ]
    for q, why in unans:
        items.append(item("unanswerable", q, "refusal", None, note=why))

    for i, it in enumerate(items, 1):
        it["id"] = f"q{i:03d}"
    return items


def main():
    items = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps({"id": it.pop("id"), **it}) + "\n")
    counts = pd.Series([json.loads(line)["category"] for line in open(OUT)]).value_counts()
    print(counts.to_string())
    print(f"Total: {counts.sum()} questions -> {OUT}")


if __name__ == "__main__":
    main()
