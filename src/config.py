"""
Central configuration. Everything is overridable with environment variables so
the same code runs in Streamlit Cloud, Docker, CI, and the eval harness.
"""
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
EV_CSV = os.path.join(RAW_DIR, "ev_grid_features.csv")
SOLAR_CSV = os.path.join(RAW_DIR, "palm_springs_features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Embedding backend:
#   "minilm"  - sentence-transformers all-MiniLM-L6-v2 (production default)
#   "hashing" - dependency-free lexical hashing embedder (offline / air-gapped CI)
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "minilm")

# Each backend gets its own persisted index so they never get mixed up.
CHROMA_DIR = os.getenv(
    "CHROMA_DIR",
    os.path.join(DATA_DIR, "chroma_db" if EMBEDDINGS_BACKEND == "minilm" else f"chroma_db_{EMBEDDINGS_BACKEND}"),
)

# LLM provider: "bedrock" (production) or "anthropic" (Claude API, handy for evals)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
ANTHROPIC_MODEL_ID = os.getenv("ANTHROPIC_MODEL_ID", "claude-haiku-4-5")
