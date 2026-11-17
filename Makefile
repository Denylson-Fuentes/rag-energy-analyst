.PHONY: install test lint index eval-offline eval-retrieval eval-agent report docker docker-test run

PY ?= python

install:
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	$(PY) -m pytest -q

index:
	$(PY) scripts/ingest_all.py

# Retrieval + ceiling + figures with the offline embedder (no downloads, no LLM).
eval-offline:
	EMBEDDINGS_BACKEND=hashing $(PY) -c "from src.ingestion.build_vectorstore import build_vectorstore as b; import os; from src import config; os.path.isdir(config.CHROMA_DIR) or b()"
	EMBEDDINGS_BACKEND=hashing $(PY) -m evals.run_retrieval_eval
	EMBEDDINGS_BACKEND=hashing $(PY) -m evals.run_ceiling
	$(PY) -m evals.report

# Retrieval eval with production MiniLM embeddings (downloads the model once).
eval-retrieval: index
	$(PY) -m evals.run_retrieval_eval --gate
	$(PY) -m evals.run_ceiling
	$(PY) -m evals.report

# End-to-end LLM eval. Set LLM_PROVIDER=bedrock (AWS creds) or anthropic (ANTHROPIC_API_KEY).
eval-agent: index
	$(PY) -m evals.run_agent_eval --variants rag_only agent_v1 agent_v2
	$(PY) -m evals.report

report:
	$(PY) -m evals.report

docker:
	docker build -t energy-analyst .

docker-test:
	docker build --target test -t energy-analyst:test .

run:
	streamlit run app.py
