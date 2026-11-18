"""
Hybrid retrieval: metadata filtering + semantic search.

Pure semantic search fails on dates and IDs because small embedding models
treat "2020-08-11" and "2020-11-08" as nearly identical. This module
extracts structured constraints via regex, applies them as Chroma filters,
then ranks the filtered subset semantically.

Two versions live here on purpose so the evaluation can compare them:

* v1 (`smart_retrieve`) - the original shipped retriever. Kept unchanged.
* v2 (`smart_retrieve_v2`) - fixes the failure modes found by the eval set:
    - only the first date / node in a question was used (broke comparisons)
    - natural-language dates ("August 11, 2020") were ignored
    - substring keyword matching ("ev" in "every", "level", "seven")
      applied the wrong source filter
    - when a filter matched nothing, it silently fell back to unfiltered
      search and handed the LLM documents for the WRONG date/node
"""
import re
from datetime import date as _date

from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# v1 (original)
# ---------------------------------------------------------------------------


def extract_constraints(query: str) -> dict:
    """
    Extract structured constraints from natural language query (v1).

    Returns dict with optional keys: 'date', 'grid_node', 'source'.
    """
    constraints = {}

    date_match = re.search(r'\d{4}-\d{2}-\d{2}', query)
    if date_match:
        constraints['date'] = date_match.group()

    node_match = re.search(r'GRID_[A-Z0-9]+', query, re.IGNORECASE)
    if node_match:
        constraints['grid_node'] = node_match.group().upper()

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
        if isinstance(value, (list, tuple)):
            conditions.append({key: value[0]} if len(value) == 1 else {key: {"$in": list(value)}})
        else:
            conditions.append({key: value})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def smart_retrieve(query: str, vectorstore, k: int = 3) -> list[Document]:
    """
    Hybrid retrieval (v1):
      1. Extract structured constraints from the query
      2. Apply them as metadata filters
      3. Semantic search within the filtered subset
      4. If no constraints found (or nothing matched), fall back to pure semantic search
    """
    constraints = extract_constraints(query)
    where_filter = build_chroma_filter(constraints)

    if where_filter:
        try:
            results = vectorstore.similarity_search(query, k=k, filter=where_filter)
            if results:
                return results
        except Exception as e:
            print(f"Filtered search failed: {e}. Falling back to unfiltered.")

    return vectorstore.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# v2
# ---------------------------------------------------------------------------

_MONTHS = {
    m: i
    for i, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"),
            ("may",), ("jun", "june"), ("jul", "july"), ("aug", "august"),
            ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"), ("dec", "december"),
        ],
        start=1,
    )
    for m in names
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_RE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_US_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MDY_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", re.I)
_YM_RE = re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)")
_NODE_RE = re.compile(r"\bGRID_[A-Z0-9]+\b", re.I)

_SOLAR_KW = re.compile(r"\b(solar|irradiance|ghi|palm springs|sun|sunny|cloud\w*|pv)\b", re.I)
_EV_KW = re.compile(r"\b(ev|evs|charging|charger\w*|grid node\w*|vehicle\w*|kwh)\b", re.I)


def _safe_date(y: int, m: int, d: int) -> str | None:
    try:
        return _date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def parse_dates(text: str) -> list[str]:
    """Return every calendar date mentioned in `text`, ISO formatted, in order, de-duplicated."""
    found: list[tuple[int, str]] = []
    for m in _ISO_RE.finditer(text):
        found.append((m.start(), _safe_date(m[1], m[2], m[3])))
    for m in _US_RE.finditer(text):
        found.append((m.start(), _safe_date(m[3], m[1], m[2])))
    for m in _MDY_RE.finditer(text):
        found.append((m.start(), _safe_date(m[3], _MONTHS[m[1].lower()], m[2])))
    for m in _DMY_RE.finditer(text):
        found.append((m.start(), _safe_date(m[3], _MONTHS[m[2].lower()], m[1])))
    out: list[str] = []
    for _, d in sorted(found):
        if d and d not in out:
            out.append(d)
    return out


def parse_months(text: str) -> list[str]:
    """Return 'YYYY-MM' months mentioned without a specific day (e.g. 'July 2023', '2020-08')."""
    stripped = _MDY_RE.sub(" ", _ISO_RE.sub(" ", text))
    stripped = _DMY_RE.sub(" ", stripped)
    out = []
    for m in _MONTH_YEAR_RE.finditer(stripped):
        out.append(f"{m[2]}-{_MONTHS[m[1].lower()]:02d}")
    for m in _YM_RE.finditer(stripped):
        out.append(f"{m[1]}-{m[2]}")
    return list(dict.fromkeys(out))


def extract_constraints_v2(query: str) -> dict:
    """
    Extract structured constraints (v2). Values may be lists.

    Keys: 'date' (list), 'month' (list), 'grid_node' (list), 'source' (str).
    """
    constraints: dict = {}
    invalid = [m.group() for m in _ISO_RE.finditer(query) if not _safe_date(m[1], m[2], m[3])]
    if invalid:
        constraints["invalid_date"] = invalid
    dates = parse_dates(query)
    if dates:
        constraints["date"] = dates
    else:
        months = parse_months(query)
        if months:
            constraints["month"] = months

    nodes = list(dict.fromkeys(n.upper() for n in _NODE_RE.findall(query)))
    if nodes:
        constraints["grid_node"] = nodes

    if nodes or (_EV_KW.search(query) and not _SOLAR_KW.search(query)):
        constraints["source"] = "ev_grid"
    elif _SOLAR_KW.search(query) and not _EV_KW.search(query):
        constraints["source"] = "solar"
    return constraints


class NoMatchingDocuments(Document):
    """Sentinel document returned when explicit constraints match nothing."""


def smart_retrieve_v2(query: str, vectorstore, k: int = 3) -> list[Document]:
    """
    Hybrid retrieval (v2).

    * Uses every date / node in the question (`$in` filters), so comparisons
      retrieve both sides; k is raised to cover every requested entity.
    * If the question names an explicit date or node and nothing matches,
      returns a single NoMatchingDocuments sentinel instead of unrelated
      documents. Falling back to unrelated documents is how the v1 agent
      produced confident answers for dates it has no data for.
    """
    c = extract_constraints_v2(query)
    if "invalid_date" in c:
        return [NoMatchingDocuments(
            page_content=f"{', '.join(c['invalid_date'])} is not a valid calendar date, so no documents match.",
            metadata={"doc_id": "NO_MATCH", "source": "none"},
        )]
    where = build_chroma_filter(c)
    if where is None:
        return vectorstore.similarity_search(query, k=k)

    n_entities = max(1, len(c.get("date", []))) * max(1, len(c.get("grid_node", [])))
    k_eff = max(k, n_entities) if "date" in c else k
    results = vectorstore.similarity_search(query, k=k_eff, filter=where)
    if results:
        return results

    # Explicit identifiers that match nothing: say so instead of guessing.
    if "date" in c or "grid_node" in c or "month" in c:
        desc = ", ".join(f"{key}={val}" for key, val in c.items())
        return [NoMatchingDocuments(
            page_content=f"No documents match the requested constraints ({desc}).",
            metadata={"doc_id": "NO_MATCH", "source": "none"},
        )]
    return vectorstore.similarity_search(query, k=k)


def format_results_for_agent(results: list[Document]) -> str:
    """
    Format retrieved documents into a string for the LLM context.

    Each document is numbered so the LLM can cite it (e.g., "source [1]").
    """
    if not results:
        return "No relevant documents found."
    if len(results) == 1 and isinstance(results[0], NoMatchingDocuments):
        return (
            "No documents match. " + results[0].page_content
            + " The dataset has no record for this request; do not substitute other dates or nodes."
        )

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


RETRIEVERS = {"v1": smart_retrieve, "v2": smart_retrieve_v2}
