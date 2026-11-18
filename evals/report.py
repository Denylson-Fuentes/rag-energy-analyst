"""
Turn evaluation results into figures and markdown tables for the client report.

    python -m evals.report

Reads reports/results/*.json, writes reports/figures/*.png and
reports/metrics.md (auto-generated; the narrative report links to it).
Any result file that is missing is skipped, so the report can be produced
before the paid LLM run.
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evals.common import RESULTS_DIR  # noqa: E402
from evals.scoring import FAILURE_LABELS  # noqa: E402

REPORT_DIR = os.path.abspath(os.path.join(RESULTS_DIR, ".."))
FIG_DIR = os.path.join(REPORT_DIR, "figures")

# Reference categorical palette, slots 1-3 (validated adjacent + all-pairs, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
NEUTRAL = "#b9b8b2"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e1", "#fcfcfb"

CATEGORY_ORDER = ["ev_lookup", "solar_lookup", "solar_cloud", "ev_peak_hour", "nl_date",
                  "comparison", "aggregation", "forecast", "inventory", "unanswerable"]
CATEGORY_LABEL = {
    "ev_lookup": "EV daily lookup", "solar_lookup": "Solar peak lookup", "solar_cloud": "Solar cloud lookup",
    "ev_peak_hour": "EV peak hour", "nl_date": "Natural-language dates", "comparison": "Two-day comparison",
    "aggregation": "Aggregation / ranking", "forecast": "Model forecast", "inventory": "Data inventory",
    "unanswerable": "Out of scope (should refuse)",
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def grouped_hbar(path, title, subtitle, categories, series: dict, xlabel, xmax=1.0, pct=True):
    """categories top-to-bottom; series: name -> list of values aligned to categories."""
    n_s = len(series)
    h = 0.8 / n_s
    fig, ax = plt.subplots(figsize=(8, 0.22 * len(categories) * n_s + 1.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ys = list(range(len(categories)))[::-1]
    for i, (name, vals) in enumerate(series.items()):
        off = (n_s - 1) / 2 * h - i * h
        for y, v in zip(ys, vals, strict=True):
            if v is None:
                continue
            ax.barh(y + off, v, height=h * 0.86, color=SERIES[i], label=name if y == ys[0] else None)
            ax.text(v + xmax * 0.01, y + off, f"{v:.0%}" if pct else f"{v:.2f}",
                    va="center", ha="left", fontsize=7, color=INK_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([CATEGORY_LABEL.get(c, c) for c in categories], color=INK)
    ax.set_xlim(0, xmax * 1.1)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_title(subtitle, loc="left", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(loc="upper right", ncol=n_s, frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def failure_stack(path, summaries: dict):
    labels = [k for k in FAILURE_LABELS if k != "ok"]
    present = [k for k in labels if any(s["failures"].get(k) for s in summaries.values())]
    fig, axes = plt.subplots(1, len(summaries), figsize=(3.2 * len(summaries) + 1, 0.42 * len(present) + 1.6),
                             dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = [axes] if len(summaries) == 1 else axes
    xmax = max(max(s["failures"].values(), default=0) for s in summaries.values()) or 1
    for ax, (variant, s) in zip(axes, summaries.items(), strict=True):
        _style(ax)
        ys = list(range(len(present)))[::-1]
        vals = [s["failures"].get(k, 0) for k in present]
        ax.barh(ys, vals, height=0.6, color=SERIES[0])
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + xmax * 0.02, y, str(v), va="center", fontsize=8, color=INK_2)
        ax.set_xlim(0, xmax * 1.2)
        ax.set_title(f"{variant}  (n={s['n']}, errors={s['n'] - round(s['accuracy'] * s['n'])})",
                     loc="left", fontsize=9, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([FAILURE_LABELS[k] for k in present], color=INK, fontsize=8)
    fig.suptitle("Failure modes by agent variant (count of questions)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _load(pattern):
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    return {os.path.basename(f): json.load(open(f)) for f in files}


def _table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    md = ["# Evaluation metrics (auto-generated by `python -m evals.report`)", ""]

    for name, d in _load("retrieval_*.json").items():
        tag = name[len("retrieval_"):-5]
        s = d["summary"]
        md += [f"## Retrieval (embeddings: `{tag}`, n={s['overall']['hybrid_v2']['n']} questions with gold documents)", ""]
        md.append(_table(["Retriever", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR", "p50 ms"],
                         [[r, f"{a['recall@1']:.3f}", f"{a['recall@3']:.3f}", f"{a['recall@5']:.3f}",
                           f"{a['recall@10']:.3f}", f"{a['mrr']:.3f}", a["p50_ms"]] for r, a in s["overall"].items()]))
        cats = [c for c in CATEGORY_ORDER if c in s["by_category"]["hybrid_v2"]]
        md += ["", "Recall@3 by category:", ""]
        md.append(_table(["Category"] + list(s["by_category"]),
                         [[CATEGORY_LABEL[c]] + [f"{s['by_category'][r][c]['recall@3']:.2f}" for r in s["by_category"]]
                          for c in cats]))
        md.append("")
        grouped_hbar(os.path.join(FIG_DIR, f"retrieval_recall3_{tag}.png"),
                     "Retrieval recall@3 by question type",
                     f"Share of gold documents in the top 3 results · embeddings: {tag}",
                     cats, {r: [s["by_category"][r][c]["recall@3"] for c in cats] for r in s["by_category"]},
                     "Recall@3")

    for name, d in _load("ceiling_*.json").items():
        tag = name[len("ceiling_"):-5]
        s = d["summary"]
        cats = [c for c in CATEGORY_ORDER if c in s["by_category"]["agent_v2"]]
        md += [f"## Tool-layer answerability ceiling (embeddings: `{tag}`, n=165)", ""]
        md.append(_table(["Category"] + list(s["overall"]),
                         [[CATEGORY_LABEL[c]] + [f"{s['by_category'][v][c]:.2f}" for v in s["overall"]] for c in cats]
                         + [["**Overall**"] + [f"**{s['overall'][v]:.2f}**" for v in s["overall"]]]))
        md.append("")
        grouped_hbar(os.path.join(FIG_DIR, f"ceiling_{tag}.png"),
                     "Can the tools even surface the answer?",
                     f"Share of questions whose gold answer reaches the LLM · embeddings: {tag}",
                     cats, {v: [s["by_category"][v][c] for c in cats] for v in s["overall"]}, "Answerability ceiling")

    for d in _load("agent_summary_*.json").values():
        if d["provider"] == "scripted":
            continue
        sm = d["summaries"]
        md += [f"## End-to-end agent ({d['provider']} · {d['model']} · embeddings `{d['embeddings_backend']}`)", ""]
        keys = [("accuracy", "Answer accuracy (all)"), ("accuracy_answerable", "Accuracy, answerable"),
                ("refusal_accuracy", "Correct refusals (out of scope)"), ("false_refusal_rate", "False refusals"),
                ("groundedness_claim_level", "Groundedness (numeric claims)"),
                ("fully_grounded_rate", "Answers fully grounded"),
                ("judge_groundedness_claim_level", "Groundedness (LLM judge)"),
                ("tool_selection_accuracy", "Tool selection"), ("latency_p50_s", "Latency p50 (s)"),
                ("latency_p95_s", "Latency p95 (s)"), ("mean_llm_calls", "LLM calls / query"),
                ("cost_per_query_usd", "Cost / query (USD)")]
        rows = []
        for k, label in keys:
            if any(k in s for s in sm.values()):
                rows.append([label] + [s.get(k, "–") for s in sm.values()])
        md.append(_table(["Metric"] + list(sm), rows))
        cats = [c for c in CATEGORY_ORDER if c in next(iter(sm.values()))["by_category"]]
        md += ["", "Accuracy by category:", ""]
        md.append(_table(["Category"] + list(sm),
                         [[CATEGORY_LABEL[c]] + [f"{s['by_category'][c]:.2f}" for s in sm.values()] for c in cats]))
        md += ["", "Failure modes (count of questions):", ""]
        fl = [k for k in FAILURE_LABELS if k != "ok" and any(s["failures"].get(k) for s in sm.values())]
        md.append(_table(["Failure mode"] + list(sm),
                         [[FAILURE_LABELS[k]] + [s["failures"].get(k, 0) for s in sm.values()] for k in fl]))
        md.append("")
        grouped_hbar(os.path.join(FIG_DIR, f"agent_accuracy_{d['provider']}.png"),
                     "End-to-end answer accuracy by question type", f"{d['model']} · 165 questions",
                     cats, {v: [s["by_category"][c] for c in cats] for v, s in sm.items()}, "Answer accuracy")
        failure_stack(os.path.join(FIG_DIR, f"failure_modes_{d['provider']}.png"), sm)

    with open(os.path.join(REPORT_DIR, "metrics.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {os.path.join(REPORT_DIR, 'metrics.md')} and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
