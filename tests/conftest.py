"""
Test configuration.

Tests run fully offline: the vector store uses the deterministic `hashing`
embedder unless EMBEDDINGS_BACKEND is set explicitly (CI's retrieval job sets
it to `minilm`). The index is built once per session if missing (~10 s).
"""
import os
import sys

os.environ.setdefault("EMBEDDINGS_BACKEND", "hashing")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def vectorstore():
    from src import config
    from src.ingestion.build_vectorstore import build_vectorstore
    from src.tools.retrieval_tool import get_vectorstore

    if not os.path.isdir(config.CHROMA_DIR) or not os.listdir(config.CHROMA_DIR):
        build_vectorstore()
    return get_vectorstore()


@pytest.fixture(scope="session")
def questions():
    from evals.common import load_questions

    return load_questions()
