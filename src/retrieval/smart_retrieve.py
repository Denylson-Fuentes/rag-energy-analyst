"""
Hybrid retrieval: metadata filtering + semantic search.

Pure semantic search fails on dates and IDs because small embedding models
treat "2020-08-11" and "2020-11-08" as nearly identical. This module
extracts structured constraints via regex, applies them as Chroma filters,
then ranks the filtered subset semantically.
"""
import re
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def extract_constraints(query: str) -> dict:
    """
    Extract structured constraints from natural language query.

    Returns dict with optional keys: 'date', 'grid_node', 'source'.
    """
    constraints = {}

    # Date: YYYY-MM-DD
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
    if date_match:
        constraints['date'] = date_match.group()

    # Grid node: GRID_XXX
    node_match = re.search(r'GRID_[A-Z0-9]+', query, re.IGNORECASE)
    if node_match:
        constraints['grid_node'] = node_match.group().upper()

    # Source keyword
    query_lower = query.lower()
    if any(k in query_lower for k in ['solar', 'irradiance', 'ghi', 'palm springs']):
        constraints['source'] = 'solar'
    elif any(k in query_lower for k in ['ev', 'charging', 'charger', 'grid node']):
        constraints['source'] = 'ev_grid'

    return constraints


def build_chroma_filter(constraints: dict) -> dict | None:
    """Convert constraints dict into Chroma 'where' filter format."""
    conditions = []
    for key, value in constraints.items():
        conditions.append({key: value})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def smart_retrieve(query: str, vectorstore: Chroma, k: int = 3) -> list[Document]:
    """
    Hybrid retrieval:
      1. Extract structured constraints from the query
      2. Apply them as metadata filters
      3. Semantic search within the filtered subset
      4. If no constraints found, fall back to pure semantic search
    """
    constraints = extract_constraints(query)
    where_filter = build_chroma_filter(constraints)

    if where_filter:
        # Try filtered retrieval first
        try:
            results = vectorstore.similarity_search(query, k=k, filter=where_filter)
            if results:
                return results
        except Exception as e:
            print(f"Filtered search failed: {e}. Falling back to unfiltered.")

    # Fallback: pure semantic
    return vectorstore.similarity_search(query, k=k)


def format_results_for_agent(results: list[Document]) -> str:
    """
    Format retrieved documents into a string for the LLM context.

    Each document is numbered so the LLM can cite it (e.g., "source [1]").
    """
    if not results:
        return "No relevant documents found."

    formatted = []
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata
        header = f"[Source {i}] {meta.get('source', 'unknown')} | "
        if 'grid_node' in meta:
            header += f"Grid Node: {meta['grid_node']} | "
        if 'location' in meta:
            header += f"Location: {meta['location']} | "
        header += f"Date: {meta.get('date', 'unknown')}"

        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)