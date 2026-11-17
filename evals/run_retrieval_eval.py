"""
Retrieval evaluation: recall@k and MRR for every question that has gold
documents, comparing three retrievers on the same Chroma index:

    semantic   - pure vector similarity (no filters)
    hybrid_v1  - the originally shipped hybrid retriever
    hybrid_v2  - the eval-driven retriever

The question text is used as the query (the agent may rewrite it; that effect
is captured separately in the end-to-end agent run).

    python -m evals.run_retrieval_eval [--gate]
"""
import argparse
import time
from collections import defaultdict

from evals.common import load_questions, write_json
from src import config
from src.retrieval.smart_retrieve import smart_retrieve, smart_retrieve_v2
from src.tools.retrieval_tool import get_vectorstore

KS = (1, 3, 5, 10)


def _semantic(q, vs, k):
    return vs.similarity_search(q, k=k)


RETRIEVERS = {"semantic": _semantic, "hybrid_v1": smart_retrieve, "hybrid_v2": smart_retrieve_v2}

# Regression gate used in CI (production retriever, production k=3).
GATE = {"retriever": "hybrid_v2", "metric": "recall@3", "min": 0.95}


def evaluate(questions, vs, retrievers=RETRIEVERS):
    rows = []
    for name, fn in retrievers.items():
        for it in questions:
            if not it["retrieval_applicable"]:
                continue
            t0 = time.perf_counter()
            docs = fn(it["question"], vs, max(KS))
            ms = (time.perf_counter() - t0) * 1000
            ranked = [d.metadata.get("doc_id") for d in docs]
            gold = set(it["gold_doc_ids"])
            row = {"retriever": name, "id": it["id"], "category": it["category"], "ms": ms, "ranked": ranked[:10]}
            for k in KS:
                row[f"recall@{k}"] = len(gold & set(ranked[:k])) / len(gold)
            rr = 0.0
            for i, d in enumerate(ranked, 1):
                if d in gold:
                    rr = 1 / i
                    break
            row["mrr"] = rr
            rows.append(row)
    return rows


def summarize(rows):
    metrics = [f"recall@{k}" for k in KS] + ["mrr"]
    overall, by_cat = {}, defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["retriever"], None)].append(r)
        groups[(r["retriever"], r["category"])].append(r)
    for (ret, cat), rs in groups.items():
        agg = {m: round(sum(r[m] for r in rs) / len(rs), 4) for m in metrics}
        agg["n"] = len(rs)
        agg["p50_ms"] = round(sorted(r["ms"] for r in rs)[len(rs) // 2], 1)
        if cat is None:
            overall[ret] = agg
        else:
            by_cat[ret][cat] = agg
    return {"embeddings_backend": config.EMBEDDINGS_BACKEND, "overall": overall, "by_category": by_cat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="exit non-zero if the regression gate fails")
    args = ap.parse_args()

    qs = load_questions()
    rows = evaluate(qs, get_vectorstore())
    summary = summarize(rows)
    tag = config.EMBEDDINGS_BACKEND
    write_json(f"retrieval_{tag}.json", {"summary": summary, "rows": rows})

    print(f"Retrieval eval (embeddings={tag}, n={summary['overall']['hybrid_v2']['n']} questions)")
    print(f"{'retriever':<11} " + " ".join(f"{m:>9}" for m in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]))
    for ret, agg in summary["overall"].items():
        print(f"{ret:<11} " + " ".join(f"{agg[m]:>9.3f}" for m in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]))

    if args.gate:
        val = summary["overall"][GATE["retriever"]][GATE["metric"]]
        ok = val >= GATE["min"]
        print(f"\nGate: {GATE['retriever']} {GATE['metric']} = {val:.3f} (min {GATE['min']}) -> {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
