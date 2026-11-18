# Energy Analyst Assistant: Evaluation and Hardening Report

**Prepared for:** Grid Operations & Analytics stakeholders
**System:** Energy Analyst, an agentic RAG assistant over EV charging and solar irradiance data
**Version evaluated:** v1 (as deployed) and v2 (this engagement)
**Status of measurements:** see the box below

> **What is measured and what is pending.**
> Measured: retrieval quality and the tool-layer answerability ceiling, over the full
> 165-question set, with the **production MiniLM embeddings** in
> [CI run #1](https://github.com/Denylson-Fuentes/rag-energy-analyst/actions/runs/44233104401)
> (verbatim output: `reports/results/ci_minilm_run1.txt`). CI re-runs these on every push.
> The per-category charts were produced locally with the offline hashing embedder, which
> needs no model download (`make eval-offline`). Charts are labelled with their embedder.
> Pending: end-to-end LLM answer accuracy and groundedness (Section 4.3). These need cloud
> LLM credentials, and running them is one command (`make eval-agent`) or the manual GitHub
> workflow *Agent eval (LLM, manual)*. `reports/metrics.md` regenerates with the results.

---

## 1. Executive summary

**The question.** Can operators trust the assistant's answers enough to use it for routine
lookups, and where does it break?

**What we did.** We built a 165-question evaluation set from the underlying data. Every
question has a machine-checkable gold answer. We then measured the system layer by layer:
does retrieval find the right records, can the tools surface the answer at all, and does
the LLM report it faithfully. We fixed the failure modes we found and wrapped the system in
automated tests, a container, and CI so that regressions are caught before deployment.

**Key findings (measured):**

1. **The deployed system cannot answer about a third of realistic questions, whatever the
   LLM does.** For only **64%** of questions does the deployed agent's tool layer return the
   correct answer to the model (53% for plain RAG). The model cannot report what it never sees.
2. **Pure vector search does not work on this data.** With the production MiniLM embeddings,
   semantic search alone finds the right record in the top 3 for **1.8%** of lookup questions.
   Dates and node IDs carry the meaning, and small embedding models don't encode them. The
   metadata filter is what makes the system work, so it must be tested as carefully as the model.
3. **Four specific defects explain most of the gap:**
   - *Month and year questions* ("which node used the most energy in August 2020?") are
     impossible with a top-3 retriever: **0 of 20** answerable.
   - *Comparisons* retrieve only the first date mentioned: **3 of 15** answerable.
   - *Natural-language dates* ("August 11, 2020") are ignored by the date filter, so only
     **3 of 15** such questions can be answered, against **100%** for ISO dates.
   - *Silent fallback.* When a request is out of range (for example, a date before the data
     starts), the retriever quietly returns **other dates' records**. This happened for
     **15 of 15** out-of-scope questions. It is the single largest hallucination risk,
     because the model receives real-looking numbers for the wrong day.
4. **After the v2 changes, the tool-layer ceiling rises from 64% to 96%.** Retrieval recall@3
   rises from 0.80 to 1.00 on the same production index. The remaining 4% (6 of 165) are out-of-scope
   requests, such as electricity prices or another city on a valid date. Only the model's own
   judgment can catch these, and the v2 prompt now states the data's scope explicitly.
5. **We also found and fixed a production concurrency bug.** Under concurrent use, model and
   vector-store loading raced and crashed requests (5 of the first 165 harness calls failed).
   This would have surfaced as intermittent errors in the multi-user Streamlit app.

**Recommendation.** Deploy v2. It is behind the same UI and costs the same per query. Before
widening access, run the end-to-end LLM evaluation once with production credentials. Gate
releases on two numbers: retrieval recall@3 ≥ 0.95, which CI already enforces, and the
correct-refusal rate on out-of-scope questions, which needs the LLM run.

---

## 2. System under evaluation

| Component | v1 (deployed) | v2 (this engagement) |
|---|---|---|
| LLM | Claude Haiku 4.5 on Amazon Bedrock, temperature 0 | unchanged |
| Retrieval | Chroma + MiniLM, hybrid metadata filter (first date/node only) | multi-date/node filters, natural-language dates, explicit "no match" instead of fallback |
| Tools | search, solar forecast, EV prediction, node inventory | + `query_energy_stats` (exact totals, averages, rankings over any span) |
| Prompt | role + rules | + explicit data scope, routing guidance, "never substitute another date" |
| Reliability | unguarded lazy loading | thread-safe loading |
| Delivery | Streamlit Cloud | + Docker image, GitHub Actions CI, regression gate |

Three variants were compared throughout:

- **`rag_only`**: classic RAG (retriever + LLM, no other tools). This is the baseline that
  most "chat with your data" systems ship.
- **`agent_v1`**: the deployed agent.
- **`agent_v2`**: the hardened agent.

---

## 3. Evaluation design

### 3.1 Question set (165 questions, 10 types)

| Type | n | Example | Gold answer |
|---|---|---|---|
| EV daily lookup | 30 | *Total charging energy at GRID_07A on 2020-02-24?* | kWh (±0.1) |
| Solar peak lookup | 25 | *Peak GHI in Palm Springs on 2023-05-09?* | W/m² (±1) |
| Solar cloud lookup | 10 | *How cloudy was it on average on …?* | % (±0.6) |
| EV peak hour | 15 | *Which hour was the charging peak for GRID_LV21 on …?* | hour |
| Natural-language dates | 15 | *… on 11 August 2020?* | kWh / W/m² |
| Two-day comparison | 15 | *Was peak GHI higher on … or …? Give both values.* | both values |
| Aggregation / ranking | 20 | *Which grid node had the highest total EV charging energy in August 2020?* | node / date / total |
| Model forecast | 15 | *Run the 6-hour-ahead solar model for … What was its MAE?* | model output |
| Data inventory | 5 | *What grid nodes are available?* | node list / dates |
| Out of scope | 15 | *Peak GHI in Phoenix …*, *wholesale price …*, *GRID_LV22 on 2020-02-30* | must refuse |

**How the gold answers were built.** Questions are generated from the raw CSVs by a seeded
script (`evals/build_dataset.py`). Gold values come from independent pandas code, not the
agent's tool code, so a tool bug cannot also correct its own answer key. Sampling is
stratified across all 10 grid nodes and the full year of solar data. Phrasings vary within
each type. Lookup and comparison questions each record the IDs of the documents that contain
the answer. These IDs feed the retrieval metrics.

### 3.2 Metrics

| Metric | Definition | Why it matters to operators |
|---|---|---|
| **Retrieval recall@k** | Share of gold documents among the top *k* retrieved (k = 1, 3, 5, 10); production uses k = 3 | If the record isn't retrieved, the answer can't be right |
| **MRR** | Mean reciprocal rank of the first gold document | Ranking quality |
| **Answerability ceiling** | Share of questions where the tools return the gold answer to the model; runs without an LLM | Upper bound on accuracy; separates tool problems from model problems |
| **Answer accuracy** | Gold value present in the answer within tolerance; correct refusal for out-of-scope questions | The headline number |
| **Groundedness** | Share of the numbers the agent states that trace to a tool output (exact, rounded, or a simple difference, sum, or percent change of tool values) | Measures fabrication directly |
| **Fully grounded rate** | Share of answers with zero untraceable numbers | What an operator experiences |
| **Tool-selection accuracy** | Agent called at least one tool able to answer | Routing quality |
| **Failure mode** | Each wrong answer gets one root cause (Section 4.4) | Tells you what to fix |

**Grading.** Scoring is deterministic and unit-tested (`tests/test_scoring.py`, 30+ cases).
Every gold answer in this domain is a number, ID, date, or "no data", so exact checks are
cheaper and more reproducible than an LLM judge. An optional LLM judge (`--judge`) cross-checks
groundedness at the claim level, including non-numeric claims. It uses a different model from
the agent, so the agent never grades itself.

---

## 4. Results

### 4.1 Retrieval

*110 questions with gold documents. Same Chroma index for all three retrievers.*

**Production embeddings (MiniLM, CI run #1):**

| Retriever | Recall@1 | **Recall@3** | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| Pure semantic | 0.00 | **0.02** | 0.05 | 0.10 | 0.02 |
| Hybrid v1 (deployed) | 0.80 | **0.80** | 0.80 | 0.80 | 0.87 |
| **Hybrid v2** | 0.93 | **1.00** | 1.00 | 1.00 | 1.00 |

**Pure semantic search fails almost completely with MiniLM.** Every daily document looks the
same to a sentence embedder ("Grid Node … Date … Total Energy …"), and the model does not
distinguish 2020-08-11 from 2020-11-08. The metadata filter does the real work. Using
a bigger embedding model would not fix this; the fix is to treat identifiers as structured data.

For comparison, a lexical (hashing) embedder that matches date strings character by character
reaches 0.82 recall@3 on its own. The two hybrid retrievers score the same with either embedder,
because the filter decides the result.

Per-category view (offline hashing embedder; the hybrid columns match production):

![Retrieval recall@3 by question type](figures/retrieval_recall3_hashing.png)

Reading the chart:

- On ISO-dated single lookups all three retrievers are perfect. The deployed system works
  well for the questions it was demoed on.
- **Natural-language dates** fail almost completely for v1 (7%). The date regex only
  recognises `YYYY-MM-DD`, so the date filter is never applied.
- **On comparisons, v1 retrieves at most half the evidence (50%).** v1 applies a filter for
  the *first* date only, which excludes the second day by construction. With a lexical
  embedder, unfiltered search actually beats v1 here (73%).
- Recall@1 for v2 is 0.93 rather than 1.00 only because comparisons have two gold documents.
- Pure-semantic figures move by about ±0.02 between index rebuilds, because Chroma's HNSW index
  is approximate and not bit-for-bit reproducible. The filtered (hybrid) figures are stable.

### 4.2 Tool-layer answerability ceiling (no LLM)

*Production MiniLM embeddings, CI run #1:*

| | rag_only | agent_v1 | **agent_v2** |
|---|---|---|---|
| **Overall ceiling** | 53% | 64% | **96%** |
| Lookups: EV, solar peak, cloud, peak hour (80) | 100% | 100% | 100% |
| Aggregation / ranking (20) | 0% | 0% | 100% |
| Two-day comparison (15) | 20% | 20% | 100% |
| Natural-language dates (15) | 20%* | 20%* | 100% |
| Model forecast (15) | 7% | 100% | 100% |
| Data inventory (5) | 0% | 100% | 100% |
| Out of scope, "no data" signalled (15) | 0% | 0% | 60% |

![Answerability ceiling](figures/ceiling_hashing.png)
*(Chart from the offline embedder: identical except natural-language dates for v1, 27%.)*

\* Slightly above retrieval recall because, by coincidence, a gold value sometimes appears
in a wrong day's document. The ceiling is therefore an *upper* bound for v1.

**Interpretation.** For a third of realistic questions, the deployed agent's failure is
decided before the LLM runs. More prompt engineering or a bigger model cannot fix that. The
analytics tool and the retriever fixes move the ceiling from 64% to 96%.

### 4.3 End-to-end agent accuracy and groundedness (pending LLM run)

| Metric | rag_only | agent_v1 | agent_v2 |
|---|---|---|---|
| Answer accuracy (165) | *pending* | *pending* | *pending* |
| Correct refusals (15 out of scope) | *pending* | *pending* | *pending* |
| Groundedness, numeric claims | *pending* | *pending* | *pending* |
| Fully grounded answers | *pending* | *pending* | *pending* |
| Tool-selection accuracy | *pending* | *pending* | *pending* |
| Latency p50 / p95 · cost per query | *pending* | *pending* | *pending* |

To produce these figures:
`LLM_PROVIDER=bedrock make eval-agent`, or run the GitHub workflow *Agent eval (LLM, manual)*
with AWS secrets. Expected spend is roughly $1–3 per variant on Haiku 4.5. The runner stores
full traces in `reports/results/`: every tool call, tool output, token count and latency. All
scoring can be re-run on stored traces at no cost (`--score-only`).

**What to expect, and how to read it.** The ceiling bounds accuracy: about 0.53 / 0.64 / 0.96.
The interesting quantity is the **gap between ceiling and accuracy**. That gap is the model's
own error: misreading evidence, choosing the wrong tool, or failing to refuse. The failure
taxonomy below splits it into actionable causes.

### 4.4 Failure-mode breakdown

Root causes found so far (tool layer, measured), with the fix applied in v2:

| Failure mode | Affected (v1) | Root cause | v2 fix | Ceiling after |
|---|---|---|---|---|
| **Silent fallback to wrong records** | 15 / 15 out-of-scope | A filter matching nothing fell back to unfiltered search | Explicit "no documents match", and the prompt forbids substitution | 9 / 15 caught by tools; 6 rely on the LLM |
| **Aggregation impossible** | 20 / 20 | Top-3 retrieval cannot sum or rank 30–900 days | `query_energy_stats` analytics tool | 20 / 20 |
| **Second entity dropped** | 12 / 15 comparisons | Constraint parser kept the first date/node only | `$in` filters over every date/node; k widened | 15 / 15 |
| **Natural-language dates ignored** | 12 / 15 | Regex recognised ISO dates only | Date parser for 5 formats | 15 / 15 |
| **Wrong source filter** | latent | Substring match: "ev" in "every", "level", "seven" | Word-boundary keyword match | (unit-tested) |
| **Impossible dates** | 2 / 2 | 2020-02-30 silently dropped, so other days were returned | Flagged as invalid | 2 / 2 |
| **Concurrency crash** | 5 / 165 harness calls | Unlocked lazy init of Chroma client and model globals | Locks | 0 / 165 |

The end-to-end run adds model-level causes, each with its own label in the trace files:
*answered without a tool*, *wrong tool*, *evidence retrieved but misreported*, *wrong answer
with untraceable numbers* (hallucination), *false refusal*, and *answered an unanswerable
question*.

---

## 5. Engineering deliverables

| Deliverable | Where | What it guarantees |
|---|---|---|
| Evaluation set (165) + generator | `evals/data/questions.jsonl`, `evals/build_dataset.py` | Reproducible; a test asserts the file regenerates identically |
| Scorers + taxonomy | `evals/scoring.py` | Deterministic, unit-tested |
| Runners | `evals/run_retrieval_eval.py`, `run_ceiling.py`, `run_agent_eval.py` | Layered diagnosis; traces stored for re-scoring |
| Report generator | `evals/report.py` → `reports/metrics.md`, `reports/figures/` | Numbers in this report regenerate from results files |
| Test suite (77 tests) | `tests/` | Runs offline in ~10 s, including an end-to-end agent-loop test with a scripted model |
| Container | `Dockerfile` | Index baked in; `--target test` fails the build if lint or tests fail; health check |
| CI | `.github/workflows/ci.yml` | Lint + tests → MiniLM retrieval eval with **recall@3 ≥ 0.95 gate** → Docker build and health check. All green on [run #1](https://github.com/Denylson-Fuentes/rag-energy-analyst/actions/runs/44233104401) |
| Paid eval on demand | `.github/workflows/eval.yml` | One-click LLM evaluation with uploaded artifacts |

---

## 6. Limitations and risks

- **v2 was tuned on this evaluation set.** The defects were found on these 165 questions, so
  v2's gains are optimistic. Before quoting v2 figures externally, generate a held-out set
  with a different seed (`SEED` in `build_dataset.py`) and report on that.
- **Templated questions.** Phrasings are varied but synthetic. Real operator questions will
  be messier (typos, relative dates such as "last Tuesday", follow-ups). Sample 50 real
  queries from logs and add them as a second set.
- **Forecast questions test routing and reporting, not forecast skill.** Their gold is the
  model's own output. Forecast accuracy belongs to the model-training pipeline.
- **The answer matcher is lenient.** "Any number in the answer within tolerance" can credit an
  answer that lists many numbers. Groundedness and the per-question traces guard against this.
- **Unfiltered questions depend on a weak embedder.** MiniLM gives 2% recall without filters.
  Any question that carries no date or node ID, such as "show me a cloudy day", is served
  poorly. That category isn't in this set yet, and a better embedder or BM25 hybrid search
  should be evaluated if it matters to users.
- **The ML models are pickles from an older scikit-learn.** They load with version warnings.
  Retrain under the pinned version (`scripts/retrain_*.py`) before the next release.

---

## 7. Recommended next steps

1. **Run the end-to-end evaluation once** (about 15 minutes, a few dollars) and fill in
   Section 4.3. Adopt `agent_v2` if its accuracy is at least `agent_v1`'s on every category.
   The ceiling suggests a large margin on aggregation and comparison questions.
2. **Add a refusal gate to CI.** Correct refusals on the 15 out-of-scope questions should be
   ≥ 90%, run nightly or before release, because a wrong confident number costs more than a
   refusal.
3. **Build a held-out and a real-query set**, as in Section 6.
4. **Add observability.** Log tool calls and the groundedness score per production answer,
   and alert on a falling grounded rate.

---

### Appendix A: Reproduce

```bash
make install            # CPU torch + dev requirements
make test               # 77 tests, offline (~10 s)
make eval-offline       # retrieval + ceiling + figures, no LLM, no model download
make eval-agent         # end-to-end, needs LLM_PROVIDER + credentials
make docker-test        # builds the image and runs the suite inside it
```

### Appendix B: File map

```
evals/
  build_dataset.py      seeded generator, independent gold answers
  data/questions.jsonl  165 questions
  scoring.py            accuracy, groundedness, failure taxonomy
  run_retrieval_eval.py recall@k / MRR, 3 retrievers, CI gate
  run_ceiling.py        tool-layer answerability ceiling
  run_agent_eval.py     end-to-end runner + scorer (traces → results)
  judge.py              optional LLM groundedness judge
  report.py             metrics.md + figures
reports/
  client_report.md      this document
  metrics.md            auto-generated tables
  figures/, results/
```
