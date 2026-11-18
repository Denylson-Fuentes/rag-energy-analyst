# ⚡ Energy Analyst — RAG Agent for Grid Operations

An **agentic RAG system** that answers natural-language questions about energy grid data. It retrieves historical records, calls trained ML models on demand, and produces grounded answers with source citations.

> **New in v2:** a 165-question evaluation suite (retrieval recall@k, groundedness, answer accuracy,
> failure-mode breakdown), an analytics tool, a hardened retriever, Docker, and GitHub Actions CI.
> Read the [evaluation report](reports/client_report.md).

Built by [Denylson Fuentes](https://denylson-fuentes.netlify.app) · [Live Demo](https://rag-energy-analyst-ney5edk97ndsqgekwwpdsw.streamlit.app)

---

## What It Does

Ask it a question, and the agent decides what to do:

| Question | What the agent does |
|----------|---------------------|
| *"What was the total EV charging energy for GRID_LV22 on 2020-08-11?"* | Retrieves the exact document from a Chroma vector store using hybrid search (date + grid node filters) |
| *"What's the solar forecast for 2023-07-15?"* | Calls the trained Random Forest model, runs 6-hour-ahead inference, and reports the MAE |
| *"What grid nodes are available?"* | Calls the node inventory tool and formats the response as a table |
| *"Is solar generation higher in summer or winter?"* | Retrieves multiple solar documents and synthesizes a comparison with citations |
| *"Which grid node used the most energy in August 2020?"* | Calls the analytics tool, which aggregates all 3,216 node-days exactly (retrieval alone can't rank a month) |

---

## Architecture

User Question (Streamlit)
↓
LangChain Agent (Claude Haiku 4.5 via Bedrock)
↓
┌───┴────────┬──────────┬────────────────┐
↓ ↓ ↓ ↓
Chroma DB Solar Model EV Model Node Inventory Analytics (v2)
(3,581 docs) (RandomForest) (RandomForest) (Tool) (pandas aggregations)


**Key design decisions:**
- **Hybrid retrieval** — dates and node IDs don't embed well in small models, so we extract them (ISO and natural-language dates, every date/node in the question) and apply them as Chroma metadata filters, then rank the filtered subset semantically. If an explicit date/node matches nothing, the tool says so instead of falling back to unrelated documents.
- **Retrieval for lookups, analytics for aggregations** — totals, averages and rankings go to an exact pandas tool; a top-3 retriever cannot answer them.
- **Grounding over creativity** — `temperature=0` on the LLM. Every number in the response comes from a tool call, not from the model's imagination.
- **Honest refusal** — when tools return no data, the agent says "I don't have this" rather than hallucinating a plausible answer.
- **Rate limiting** — 20 queries per session to protect Bedrock cost ($0.004/query).

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Claude Haiku 4.5 (Amazon Bedrock, cross-region inference profile) |
| Agent framework | LangChain v1 (`create_agent`) |
| Vector store | Chroma (3,581 documents) |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| ML models | Random Forest (solar 6h-ahead, EV hourly load) |
| UI | Streamlit |
| Evaluation | 165-question set, deterministic scorers, optional LLM judge |
| Packaging / CI | Docker (with test stage), GitHub Actions |
| Deployment | Streamlit Cloud |

---

## Data Sources

- **EV charging**: NREL Workplace Charging Dataset (14,588 sessions → 3,216 grid-node-day documents)
- **Solar**: NSRDB Palm Springs 2023 (13,168 30-minute readings → 365 daily documents)

---

## Example Interactions

**Q:** *What was the total EV charging energy for GRID_LV22 on 2020-08-11?*
> Total EV charging energy for GRID_LV22 on 2020-08-11: **97.17 kWh** [Source 1]. Charging occurred across 5 hours (06:00–10:00), with peak demand at 09:00 (43.65 kWh) when 3 chargers were active.

**Q:** *What's the solar forecast for 2023-07-15?*
> Peak predicted GHI: **988 W/m²** (forecast issued at 02:00 for conditions at 08:00). Actual 6h-ahead peak: 981 W/m². **MAE: 6.3 W/m²**.

**Q:** *What grid nodes are available?*
> (Formats a table with all 10 grid nodes, their date ranges, and total energy.)

---

## Evaluation

The full write-up is in [`reports/client_report.md`](reports/client_report.md), and the auto-generated
tables are in [`reports/metrics.md`](reports/metrics.md).

| Layer | What's measured | Script |
|---|---|---|
| Retrieval | recall@1/3/5/10, MRR for semantic vs hybrid v1 vs hybrid v2 | `evals/run_retrieval_eval.py` |
| Tool layer | answerability ceiling: does the gold answer even reach the LLM? (no LLM needed) | `evals/run_ceiling.py` |
| End to end | answer accuracy, numeric groundedness, tool selection, refusals, latency, cost, failure modes | `evals/run_agent_eval.py` |

Headline (measured end to end with Claude Haiku 4.5 on Bedrock, 165 questions per variant):

| | Plain RAG | Deployed agent (v1) | **Hardened agent (v2)** |
|---|---|---|---|
| Answer accuracy | 71.5% | 83.0% | **98.8%** |
| Accuracy, answerable questions | 69.3% | 82.0% | **100%** |
| Correct refusals (out of scope) | 93.3% | 93.3% | 86.7% |
| Numeric claims traceable to a tool | 98.3% | 98.2% | 97.7% |
| Latency p50 · cost per query | 1.8 s · $0.004 | 1.8 s · $0.006 | 1.5 s · $0.006 |

Pure vector search finds the right record only **2%** of the time on this data (MiniLM), so the
metadata filter is what makes the system work. v2 cuts wrong answers from 28 to 2. Most of v1's
errors came from missing tools, not from the model.

```bash
make test           # 79 tests, offline, ~10 s
make eval-offline   # retrieval + ceiling + figures, no LLM / no model download
make eval-agent     # end-to-end; LLM_PROVIDER=bedrock|anthropic + credentials
```

## Docker

```bash
docker build -t energy-analyst .                      # bakes the vector index into the image
docker run -p 8501:8501 --env-file .env energy-analyst
docker build --target test -t energy-analyst:test .   # build fails if lint/tests fail
```

## CI

`.github/workflows/ci.yml` runs on every push: lint + tests → retrieval eval with production MiniLM
embeddings and a **recall@3 ≥ 0.95 regression gate** → Docker build, in-image tests, and a container
health check. `.github/workflows/eval.yml` runs the paid end-to-end LLM evaluation on demand.

---

## Running Locally

```bash
git clone https://github.com/Denylson-Fuentes/rag-energy-analyst.git
cd rag-energy-analyst

# Install dependencies (CPU torch first keeps the install small)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Set AWS credentials (Claude via Bedrock)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Build the vector store (one-time, ~2 min)
python scripts/ingest_all.py

# Run the app
streamlit run app.py
