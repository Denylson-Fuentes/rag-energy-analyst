"""
Embedding backends.

`minilm` is what production uses. `hashing` is a deterministic lexical
embedder (sklearn HashingVectorizer, L2-normalised) that needs no model
download. It exists so the retrieval pipeline can be built and tested in
air-gapped environments; it is NOT a substitute when reporting production
retrieval quality.
"""
from langchain_core.embeddings import Embeddings

from src import config


class HashingEmbeddings(Embeddings):
    """Character/word n-gram hashing embedder. Deterministic and offline."""

    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=n_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            norm="l2",
            alternate_sign=False,
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Embed only the document header (first 6 lines). The hourly readings
        # are long, repetitive and drown the identifying fields.
        heads = ["\n".join(t.splitlines()[:6]) for t in texts]
        return self._vec.transform(heads).toarray().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


_cache: dict[str, Embeddings] = {}


def get_embeddings(backend: str | None = None) -> Embeddings:
    backend = backend or config.EMBEDDINGS_BACKEND
    if backend not in _cache:
        if backend == "minilm":
            from langchain_huggingface import HuggingFaceEmbeddings

            _cache[backend] = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        elif backend == "hashing":
            _cache[backend] = HashingEmbeddings()
        else:
            raise ValueError(f"Unknown EMBEDDINGS_BACKEND: {backend!r}")
    return _cache[backend]
