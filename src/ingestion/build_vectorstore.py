"""
Build the Chroma vector store from the loaded documents.
"""
import os
import shutil

from langchain_chroma import Chroma

from src import config
from src.ingestion.loaders import load_ev_grid_documents, load_solar_documents
from src.retrieval.embeddings import get_embeddings


def build_vectorstore(persist_dir: str | None = None, backend: str | None = None, rebuild: bool = False):
    """Load CSVs, build embeddings, persist to disk."""
    backend = backend or config.EMBEDDINGS_BACKEND
    persist_dir = persist_dir or config.CHROMA_DIR
    if rebuild and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    print("Loading documents...")
    ev_docs = load_ev_grid_documents(config.EV_CSV)
    print(f"  EV documents: {len(ev_docs)}")
    solar_docs = load_solar_documents(config.SOLAR_CSV)
    print(f"  Solar documents: {len(solar_docs)}")
    all_docs = ev_docs + solar_docs
    print(f"Total: {len(all_docs)} documents")

    print(f"\nEmbedding backend: {backend}")
    embeddings = get_embeddings(backend)

    print("Building Chroma index...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        ids=[d.metadata["doc_id"] for d in all_docs],
        persist_directory=persist_dir,
    )

    print(f"\n✅ Vector store persisted to {persist_dir}")
    print(f"   Total documents indexed: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()
