# syntax=docker/dockerfile:1
#
# Energy Analyst: Streamlit app + agent + evaluation harness.
#
#   docker build -t energy-analyst .                       # runtime image
#   docker build --target test -t energy-analyst:test .    # runs the test suite during build
#   docker run -p 8501:8501 --env-file .env energy-analyst
#
# The vector store is built at image build time so containers start instantly.
# EMBEDDINGS_BACKEND=hashing builds an image without downloading the MiniLM model
# (air-gapped builds); production uses the default, minilm.

ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE} AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG EMBEDDINGS_BACKEND=minilm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    EMBEDDINGS_BACKEND=${EMBEDDINGS_BACKEND} \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build (~4 GB).
RUN pip install torch --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY data/raw ./data/raw
COPY models ./models
COPY app.py ./
COPY .streamlit ./.streamlit

# Bake the embedding model and vector index into the image.
RUN python scripts/ingest_all.py

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]


# ---------------------------------------------------------------------------
# Test stage: same image plus dev tools; the build fails if lint or tests fail.
FROM runtime AS test
USER root
COPY requirements-dev.txt pyproject.toml ./
RUN pip install -r requirements-dev.txt
COPY tests ./tests
COPY evals ./evals
RUN chown -R app:app /app
USER app
RUN ruff check . && python -m pytest -q
