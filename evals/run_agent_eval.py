"""
End-to-end agent evaluation.

Runs each agent variant over the eval set, stores full traces (answer, tool
calls, tool outputs, tokens, latency), then scores them. Scoring is a separate
step so scorers can be improved without paying for another LLM run:

    # run + score (needs LLM credentials)
    LLM_PROVIDER=bedrock   python -m evals.run_agent_eval --variants rag_only agent_v1 agent_v2
    LLM_PROVIDER=anthropic python -m evals.run_agent_eval --variants agent_v2 --limit 20

    # re-score existing traces
    python -m evals.run_agent_eval --score-only --provider bedrock

    # harness smoke test with the scripted model (no LLM, not a model result)
    python -m evals.run_agent_eval --provider scripted --variants agent_v2
"""
import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, ToolMessage

from evals.common import HOLDOUT, QUESTIONS, load_questions, read_jsonl, results_path, write_json, write_jsonl
from evals.scoring import FAILURE_LABELS, answer_correct, classify, groundedness, is_refusal
from src import config
from src.agent.executor import VARIANTS, build_executor

# USD per million tokens (input, output). Claude Haiku 4.5 list price.
PRICES = {"bedrock": (1.0, 5.0), "anthropic": (1.0, 5.0), "scripted": (0.0, 0.0)}


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def run_one(agent, it: dict) -> dict:
    rec = {"id": it["id"], "question": it["question"]}
    t0 = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": it["question"]}]},
            config={"recursion_limit": 16},
        )
        msgs = result["messages"]
        rec["answer"] = _text(msgs[-1].content)
        rec["tool_calls"] = [
            {"name": c["name"], "args": c["args"]}
            for m in msgs if isinstance(m, AIMessage) for c in (m.tool_calls or [])
        ]
        rec["tool_outputs"] = [_text(m.content) for m in msgs if isinstance(m, ToolMessage)]
        usage = [m.usage_metadata for m in msgs if isinstance(m, AIMessage) and m.usage_metadata]
        rec["input_tokens"] = sum(u.get("input_tokens", 0) for u in usage)
        rec["output_tokens"] = sum(u.get("output_tokens", 0) for u in usage)
        rec["llm_calls"] = sum(isinstance(m, AIMessage) for m in msgs)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    return rec


def run_variant(variant: str, questions: list[dict], provider: str, workers: int) -> list[dict]:
    if provider == "scripted":
        from evals.fake_llm import ScriptedToolModel

        llm = ScriptedToolModel(script={q["question"]: q["oracle_calls"] for q in questions})
    else:
        from src.agent.llm import get_llm

        llm = get_llm(provider)
    agent = build_executor("operator", variant, llm=llm)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(lambda it: run_one(agent, it), questions))
    for r in recs:
        r["variant"], r["provider"] = variant, provider
    return recs


def run_judge(questions: list[dict], recs: list[dict], workers: int) -> None:
    from evals.judge import get_judge, judge_record

    judge = get_judge()
    by_id = {q["id"]: q for q in questions}

    def one(r):
        if r.get("error") or "judge" in r:
            return
        try:
            r["judge"] = judge_record(judge, by_id[r["id"]]["question"], r.get("answer", ""), r.get("tool_outputs", []))
        except Exception as e:
            r["judge"] = {"error": f"{type(e).__name__}: {e}"[:300]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, recs))


def score(questions: list[dict], recs: list[dict], provider: str) -> dict:
    by_id = {q["id"]: q for q in questions}
    for r in recs:
        it = by_id[r["id"]]
        r["category"] = it["category"]
        r["correct"] = (not r.get("error")) and answer_correct(it, r.get("answer", ""))
        r["failure"] = classify(it, r)
        g = groundedness(r.get("answer", ""), r.get("tool_outputs", []), it["question"])
        r["n_claims"], r["n_supported"], r["unsupported"] = g["n_claims"], g["n_supported"], g["unsupported"]
        called = {c["name"] for c in r.get("tool_calls", [])}
        r["tool_ok"] = (not it["expected_tools"]) or bool(called & set(it["expected_tools"]))
        r["refused"] = is_refusal(r.get("answer", ""))

    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None

    with_claims = [r for r in recs if r["n_claims"]]
    lat = sorted(r["latency_s"] for r in recs)
    pin, pout = PRICES.get(provider, (0, 0))
    tin = [r.get("input_tokens", 0) for r in recs]
    tout = [r.get("output_tokens", 0) for r in recs]
    answerable = [r for r in recs if by_id[r["id"]]["answer_type"] != "refusal"]
    unanswerable = [r for r in recs if by_id[r["id"]]["answer_type"] == "refusal"]
    cats = defaultdict(list)
    for r in recs:
        cats[r["category"]].append(r)
    judged = [r["judge"] for r in recs if isinstance(r.get("judge"), dict) and "n_claims" in r["judge"]]
    judge_stats = {}
    if judged:
        jc = [j for j in judged if j["n_claims"]]
        judge_stats = {
            "judge_groundedness_claim_level": round(
                sum(j["n_supported"] for j in jc) / max(1, sum(j["n_claims"] for j in jc)), 4),
            "judge_fully_grounded_rate": round(sum(j["n_supported"] == j["n_claims"] for j in jc) / max(1, len(jc)), 4),
            "judge_n": len(judged),
        }
    return {
        **judge_stats,
        "n": len(recs),
        "accuracy": acc(recs),
        "accuracy_answerable": acc(answerable),
        "refusal_accuracy": acc(unanswerable),
        "false_refusal_rate": round(sum(r["refused"] and not r["correct"] for r in answerable) / max(1, len(answerable)), 4),
        "groundedness_claim_level": round(
            sum(r["n_supported"] for r in with_claims) / max(1, sum(r["n_claims"] for r in with_claims)), 4),
        "fully_grounded_rate": round(sum(r["n_supported"] == r["n_claims"] for r in with_claims) / max(1, len(with_claims)), 4),
        "tool_selection_accuracy": round(sum(r["tool_ok"] for r in recs) / len(recs), 4),
        "errors": sum(bool(r.get("error")) for r in recs),
        "latency_p50_s": lat[len(lat) // 2],
        "latency_p95_s": lat[int(len(lat) * 0.95) - 1],
        "mean_llm_calls": round(statistics.mean(r.get("llm_calls", 0) for r in recs), 2),
        "mean_input_tokens": round(statistics.mean(tin)),
        "mean_output_tokens": round(statistics.mean(tout)),
        "cost_per_query_usd": round((statistics.mean(tin) * pin + statistics.mean(tout) * pout) / 1e6, 5),
        "by_category": {c: acc(rs) for c, rs in sorted(cats.items())},
        "failures": dict(Counter(r["failure"] for r in recs if r["failure"] != "ok").most_common()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--provider", default=config.LLM_PROVIDER, choices=["bedrock", "anthropic", "scripted"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--question-set", default="main", choices=["main", "holdout"],
                    help="main = evals/data/questions.jsonl; holdout = questions_holdout.jsonl")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=int(os.getenv("EVAL_WORKERS", "4")))
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--print-failures", action="store_true", help="print summary JSON and every failure case")
    ap.add_argument("--judge", action="store_true", help="also run the LLM groundedness judge (evals/judge.py)")
    args = ap.parse_args()

    qs = load_questions(HOLDOUT if args.question_set == "holdout" else QUESTIONS)
    suffix = "" if args.question_set == "main" else f"_{args.question_set}"
    if args.categories:
        qs = [q for q in qs if q["category"] in args.categories]
    if args.limit:
        qs = qs[: args.limit]

    summaries = {}
    for v in args.variants:
        trace_name = f"agent_{v}_{args.provider}{suffix}.jsonl"
        if args.score_only:
            recs = read_jsonl(results_path(trace_name))
        else:
            print(f"Running {v} on {len(qs)} questions ({args.provider})...", flush=True)
            recs = run_variant(v, qs, args.provider, args.workers)
        if args.judge:
            print(f"Judging {v}...", flush=True)
            run_judge(qs, recs, args.workers)
        summaries[v] = score(qs, recs, args.provider)
        write_jsonl(trace_name, recs)

    out = write_json(f"agent_summary_{args.provider}{suffix}.json", {
        "question_set": args.question_set,
        "provider": args.provider,
        "model": config.BEDROCK_MODEL_ID if args.provider == "bedrock" else config.ANTHROPIC_MODEL_ID,
        "embeddings_backend": config.EMBEDDINGS_BACKEND,
        "summaries": summaries,
    })
    keys = ["accuracy", "accuracy_answerable", "refusal_accuracy", "groundedness_claim_level",
            "fully_grounded_rate", "tool_selection_accuracy", "latency_p50_s", "cost_per_query_usd"]
    def cell(x):
        return f"{'n/a' if x is None else x:>12}"

    print(f"\n{'metric':<26}" + "".join(f"{v:>12}" for v in summaries))
    for k in keys:
        print(f"{k:<26}" + "".join(cell(summaries[v].get(k)) for v in summaries))
    print(f"\n{'accuracy by category':<26}" + "".join(f"{v:>12}" for v in summaries))
    for c in next(iter(summaries.values()))["by_category"]:
        print(f"{c:<26}" + "".join(cell(summaries[v]["by_category"].get(c)) for v in summaries))
    for v, s in summaries.items():
        print(f"\n{v} failures: " + ", ".join(f"{FAILURE_LABELS[k]}={n}" for k, n in s["failures"].items()))

    if args.print_failures:
        # Printed so results can be read from CI logs without downloading the artifact.
        print("\n=== SUMMARY JSON ===")
        print(json.dumps(summaries, default=str))
        print("=== FAILURE CASES ===")
        for v in summaries:
            for r in read_jsonl(results_path(f"agent_{v}_{args.provider}{suffix}.jsonl")):
                if r.get("failure") not in (None, "ok"):
                    calls = ",".join(c["name"] for c in r.get("tool_calls", []))
                    ans = " ".join((r.get("answer") or r.get("error") or "").split())[:240]
                    print(f"{v}|{r['id']}|{r['category']}|{r['failure']}|{calls}|{ans}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
