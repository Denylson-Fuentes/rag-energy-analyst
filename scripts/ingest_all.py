"""Entry point: build the vector store from scratch."""
import sys
import os

# Get the absolute path of the project root (parent of scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Change working directory to project root
os.chdir(PROJECT_ROOT)

# Add project root to Python path so `src` is importable
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.build_vectorstore import build_vectorstore

if __name__ == "__main__":
    build_vectorstore()