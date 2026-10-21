"""
Build the Chroma vector store from the loaded documents.
"""
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.ingestion.loaders import load_ev_grid_documents, load_solar_documents


def build_vectorstore(data_dir: str = "data", persist_dir: str = "data/chroma_db"):
    """Load CSVs, build embeddings, persist to disk."""
    print("Loading documents...")
    ev_docs = load_ev_grid_documents(os.path.join(data_dir, "raw", "ev_grid_features.csv"))
    print(f"  EV documents: {len(ev_docs)}")

    solar_docs = load_solar_documents(os.path.join(data_dir, "raw", "palm_springs_features.csv"))
    print(f"  Solar documents: {len(solar_docs)}")

    all_docs = ev_docs + solar_docs
    print(f"Total: {len(all_docs)} documents")

    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print("Building Chroma index (this takes 2-5 minutes on first run)...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"\n✅ Vector store persisted to {persist_dir}")
    print(f"   Total documents indexed: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    build_vectorstore()