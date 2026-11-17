"""
Tool-layer answerability ceiling (no LLM required).

For each question, run the tool calls a given system *could* make and check
whether the gold answer is present in what the tools return. This is an upper
bound on that system's accuracy: if the evidence never reaches the LLM, no
prompt or model can recover it. For unanswerable questions the check is
whether the tools return an explicit "no data" signal instead of plausible
but unrelated data.

    rag_only  : v1 search with the question text
    agent_v1  : v1 search, plus the model/inventory tools for their categories
    agent_v2  : the oracle tool calls stored in the eval set (v2 tools)

    python -m evals.run_ceiling
"""
from collections import defaultdict

from evals.common import load_questions, write_json
from evals.scoring import _gold_in_evidence
from src import config
from src.tools.analytics_tool import query_energy_stats
from src.tools.ev_prediction_tool import list_available_grid_nodes, predict_ev_load
from src.tools.retrieval_tool import make_search_tool
from src.tools.solar_forecast_tool import forecast_solar

_NO_DATA_MARKERS = ("No documents match", "No charging data", "No solar data", "Analytics error",
                    "No relevant documents", "Not enough future data")

SEARCH_V1, SEARCH_V2 = make_search_tool("v1"), make_search_tool("v2")
V2_TOOLS = {
    "search_energy_data": SEARCH_V2, "query_energy_stats": query_energy_stats,
    "forecast_solar": forecast_solar, "predict_ev_load": predict_ev_load,
    "list_available_grid_nodes": list_available_grid_nodes,
}
V1_DIRECT = {"forecast_solar": forecast_solar, "predict_ev_load": predict_ev_load,
             "list_available_grid_nodes": list_available_grid_nodes}


def _safe(tool, args):
    try:
        return tool.invoke(args)
    except Exception as e:  # tools raising on bad input is itself a finding
        return f"TOOL ERROR: {type(e).__name__}: {e}"


def _v2_unanswerable_outputs(q):
    """For out-of-scope questions there is no oracle call; use v2 search on the question."""
    return [_safe(SEARCH_V2, {"query": q})]


def outputs_for(system: str, it: dict) -> list[str]:
    q = it["question"]
    if system == "rag_only":
        return [_safe(SEARCH_V1, {"query": q})]
    if system == "agent_v1":
        calls = [c for c in it["oracle_calls"] if c["tool"] in V1_DIRECT]
        if calls:
            return [_safe(V1_DIRECT[c["tool"]], c["args"]) for c in calls]
        return [_safe(SEARCH_V1, {"query": q})]
    if system == "agent_v2":
        if not it["oracle_calls"]:
            return _v2_unanswerable_outputs(q)
        return [_safe(V2_TOOLS[c["tool"]], c["args"]) for c in it["oracle_calls"]]
    raise ValueError(system)


def evidence_ok(it, outs, system) -> bool:
    if it["answer_type"] == "refusal":
        return any(m in o for o in outs for m in _NO_DATA_MARKERS)
    # Superlatives/totals over a month cannot be established from 3 sampled
    # documents even if a matching string happens to appear in them.
    if it["category"] == "aggregation" and not any(o.startswith("Analytics result") for o in outs):
        return False
    return _gold_in_evidence(it, outs)


def main():
    qs = load_questions()
    systems = ("rag_only", "agent_v1", "agent_v2")
    rows = []
    for s in systems:
        for it in qs:
            outs = outputs_for(s, it)
            rows.append({"system": s, "id": it["id"], "category": it["category"],
                         "evidence_ok": evidence_ok(it, outs, s)})

    summary = {"overall": {}, "by_category": defaultdict(dict)}
    for s in systems:
        rs = [r for r in rows if r["system"] == s]
        summary["overall"][s] = round(sum(r["evidence_ok"] for r in rs) / len(rs), 4)
        cats = sorted({r["category"] for r in rs})
        for c in cats:
            rc = [r for r in rs if r["category"] == c]
            summary["by_category"][s][c] = round(sum(r["evidence_ok"] for r in rc) / len(rc), 4)

    write_json(f"ceiling_{config.EMBEDDINGS_BACKEND}.json", {"summary": summary, "rows": rows})
    cats = sorted({r["category"] for r in rows})
    print(f"Answerability ceiling (embeddings={config.EMBEDDINGS_BACKEND}, n={len(qs)})")
    print(f"{'category':<14}" + "".join(f"{s:>10}" for s in systems))
    for c in cats:
        print(f"{c:<14}" + "".join(f"{summary['by_category'][s][c]:>10.2f}" for s in systems))
    print(f"{'OVERALL':<14}" + "".join(f"{summary['overall'][s]:>10.2f}" for s in systems))


if __name__ == "__main__":
    main()
