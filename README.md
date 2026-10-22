# ⚡ Energy Analyst — RAG Agent for Grid Operations

An **agentic RAG system** that answers natural-language questions about energy grid data. It retrieves historical records, calls trained ML models on demand, and produces grounded answers with source citations.

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

---

## Architecture

User Question (Streamlit)
↓
LangChain Agent (Claude Haiku 4.5 via Bedrock)
↓
┌───┴────────┬──────────┬────────────────┐
↓ ↓ ↓ ↓
Chroma DB Solar Model EV Model Node Inventory
(3,581 docs) (RandomForest) (RandomForest) (Tool)


**Key design decisions:**
- **Hybrid retrieval** — dates and node IDs don't embed well in small models, so we extract them via regex and apply as Chroma metadata filters, then rank the filtered subset semantically.
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

## Running Locally

```bash
git clone https://github.com/Denylson-Fuentes/rag-energy-analyst.git
cd rag-energy-analyst

# Install dependencies
pip install -r requirements.txt

# Set AWS credentials (Claude via Bedrock)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Build the vector store (one-time, ~2 min)
python scripts/ingest_all.py

# Run the app
streamlit run app.py

Testing
pytest tests/ -v