"""
Deterministic scorers for the agent evaluation.

Why deterministic: every gold answer in this domain is a number, a node ID, a
date, or "no data". Exact checks are cheaper, reproducible, and easier to
defend to a client than an LLM judge. An optional LLM judge (evals/judge.py)
is available as a cross-check for groundedness.

Exposed:
    answer_correct(item, answer)          -> bool
    is_refusal(text)                      -> bool
    groundedness(answer, tool_outputs, q) -> dict(n_claims, n_supported, unsupported)
    retrieved_doc_ids(tool_outputs)       -> list[str]
    classify(item, record)                -> failure label ("ok" when correct)
"""
import re
from itertools import combinations

from src.retrieval.smart_retrieve import _MONTH_RE, _MONTHS, parse_dates

# ----------------------------------------------------------------------------- text cleanup
_NODE = re.compile(r"\bGRID_[A-Z0-9]+\b", re.I)
_CITE = re.compile(r"\[(?:Source\s*)?\d+(?:\s*,\s*(?:Source\s*)?\d+)*\]", re.I)
_ISO = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")
_US = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
_NL = re.compile(rf"\b(?:{_MONTH_RE})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b", re.I)
_DMY = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:{_MONTH_RE})\.?(?:,?\s+\d{{4}})?\b", re.I)
_MY = re.compile(rf"\b(?:{_MONTH_RE})\.?\s+\d{{4}}\b", re.I)
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_LIST_MARK = re.compile(r"(?m)^\s*\d+[.)]\s")
_YEAR = re.compile(r"(?<![\d.,])(?:19|20)\d{2}(?![\d]|[.,]\d)")
_HORIZON = re.compile(r"\b\d+\s*-?\s*(?:hours?|h)\s*-?\s*ahead\b|\b\d+-hour\b", re.I)
_NUM = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])[-+]?\d+(?:\.\d+)?")


def _strip_non_claims(text: str) -> str:
    for rx in (_NODE, _CITE, _ISO, _US, _NL, _DMY, _MY, _TIME, _HORIZON, _LIST_MARK, _YEAR):
        text = rx.sub(" ", text)
    return text.replace("W/m²", " ").replace("m²", " ")


def extract_numbers(text: str) -> list[float]:
    """Numeric values in `text`, excluding dates, times, node IDs, years and citation markers."""
    out = []
    for m in _NUM.finditer(_strip_non_claims(text or "")):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def raw_numbers(text: str) -> list[float]:
    """Every number in a tool output (tool outputs are the evidence, so keep everything)."""
    text = _NODE.sub(" ", _ISO.sub(" ", _TIME.sub(" ", text or "")))
    return [float(m.group().replace(",", "")) for m in _NUM.finditer(text)]


def _decimals(x: float, literal: str | None = None) -> int:
    s = literal if literal is not None else repr(x)
    return len(s.split(".")[1]) if "." in s else 0


def num_match(value: float, gold: float, abs_tol: float | None) -> bool:
    tol = max(abs_tol or 0.0, abs(gold) * 0.005)
    return abs(value - gold) <= tol + 1e-9


# ----------------------------------------------------------------------------- refusals
_REFUSAL = re.compile(
    r"\b(no (?:data|record|records|information|documents?|charging data|solar data|matching)"
    r"|not available|unavailable|don't have|do not have|doesn't have|does not have"
    r"|no such|not in (?:the|my|our) (?:data|dataset)|outside (?:the|of the|my)"
    r"|(?:does|do)(?: not|n't) (?:exist|appear|include|contain|cover)|cannot (?:find|provide|answer)"
    r"|can't (?:find|provide|answer)|unable to|isn't (?:available|in)|is not (?:available|in|a valid)"
    r"|not (?:a )?valid date|invalid date|not found|no .{0,30}available"
    r"|(?:is|are) invalid|(?:does|do)(?: not|n't) (?:specify|record|track|distinguish))\b",
    re.I,
)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text or ""))


# ----------------------------------------------------------------------------- correctness
def _hours_in(text: str) -> set[int]:
    hours = {int(h) for h in re.findall(r"\b(\d{1,2}):\d{2}\b", text)}
    for h, ampm in re.findall(r"\b(\d{1,2})\s*([ap])\.?m\.?\b", text, re.I):
        h = int(h) % 12 + (12 if ampm.lower() == "p" else 0)
        hours.add(h)
    return hours


def _dates_in(text: str, year_hint: str | None = None) -> set[str]:
    found = set(parse_dates(text))
    if year_hint:  # "July 15" with the year implied by the question
        for mon, day in re.findall(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?!,?\s*\d{{4}})", text, re.I):
            try:
                found.add(f"{year_hint}-{_MONTHS[mon.lower()]:02d}-{int(day):02d}")
            except ValueError:
                pass
    return found


def answer_correct(item: dict, answer: str) -> bool:
    answer = answer or ""
    t, gold, tol = item["answer_type"], item["gold"], item.get("abs_tol")
    if t == "refusal":
        return is_refusal(answer)
    if t == "number":
        return any(num_match(v, gold, tol) for v in extract_numbers(answer))
    if t == "numbers_all":
        nums = extract_numbers(answer)
        return all(any(num_match(v, g, tol) for v in nums) for g in gold)
    if t == "text":
        return gold.lower() in answer.lower()
    if t == "text_all":
        return all(g.lower() in answer.lower() for g in gold)
    if t in ("date", "dates_all"):
        golds = [gold] if t == "date" else gold
        found = _dates_in(answer, year_hint=golds[0][:4])
        return all(g in found for g in golds)
    if t == "time":
        return int(gold[:2]) in _hours_in(answer)
    raise ValueError(f"unknown answer_type {t}")


# ----------------------------------------------------------------------------- groundedness
def groundedness(answer: str, tool_outputs: list[str], question: str = "") -> dict:
    """
    Numeric-claim groundedness: every number the agent states must be traceable
    to a tool output (exactly, as a rounding of it, or as a simple difference /
    sum / percent change of two tool numbers). Numbers that appear in the
    question are treated as given.
    """
    claims = [c for c in extract_numbers(answer) if c not in (0.0,)]
    evidence = sorted(set(raw_numbers("\n".join(tool_outputs)) + raw_numbers(question)))
    ev_small = evidence[:400]
    derived = set()
    for a, b in combinations(ev_small[:200], 2):
        derived.update({round(abs(a - b), 2), round(a + b, 2)})
        if b:
            derived.add(round(abs(a - b) / abs(b) * 100, 1))
        if a:
            derived.add(round(abs(a - b) / abs(a) * 100, 1))
    unsupported = []
    for c in claims:
        tol = 0.5 * 10 ** -_decimals(c) + 1e-9
        if any(abs(c - e) <= max(tol, abs(e) * 0.005) for e in evidence):
            continue
        if any(abs(c - d) <= max(tol, abs(d) * 0.01) for d in derived):
            continue
        unsupported.append(c)
    return {"n_claims": len(claims), "n_supported": len(claims) - len(unsupported), "unsupported": unsupported}


# ----------------------------------------------------------------------------- retrieval trace
_HEADER = re.compile(
    r"\[Source \d+\] (ev_grid|solar) \|(?: Grid Node: (GRID_[A-Z0-9]+) \|)?(?: Location: [^|]+\|)? Date: (\d{4}-\d{2}-\d{2})"
)


def retrieved_doc_ids(tool_outputs: list[str]) -> list[str]:
    ids = []
    for out in tool_outputs:
        for src, node, d in _HEADER.findall(out or ""):
            ids.append(f"ev_grid|{node}|{d}" if src == "ev_grid" else f"solar|{d}")
    return ids


def _gold_in_evidence(item: dict, tool_outputs: list[str]) -> bool:
    if item["answer_type"] in ("refusal",):
        return False
    text = "\n".join(tool_outputs)
    if item["answer_type"] in ("number", "numbers_all"):
        golds = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
        nums = raw_numbers(text)
        return all(any(num_match(v, g, item.get("abs_tol")) for v in nums) for g in golds)
    return answer_correct(item, text)


# ----------------------------------------------------------------------------- taxonomy
FAILURE_LABELS = {
    "ok": "Correct",
    "error": "Run error (exception / timeout)",
    "missed_refusal": "Answered an unanswerable question",
    "false_refusal": "Refused an answerable question",
    "no_tool_call": "Answered without calling any tool",
    "wrong_tool": "Did not call a tool that can answer this",
    "retrieval_miss": "Retriever did not return the gold document(s)",
    "hallucination": "Wrong answer containing ungrounded numbers",
    "extraction_error": "Evidence was retrieved but the answer misreported it",
    "other": "Other / unclassified",
}


def classify(item: dict, rec: dict) -> str:
    if rec.get("error"):
        return "error"
    answer, outs = rec.get("answer") or "", rec.get("tool_outputs") or []
    called = {c["name"] for c in rec.get("tool_calls") or []}
    if answer_correct(item, answer):
        return "ok"
    if item["answer_type"] == "refusal":
        return "missed_refusal"
    if not called:
        return "no_tool_call"
    if item["expected_tools"] and not called & set(item["expected_tools"]):
        return "wrong_tool"
    if item["retrieval_applicable"] and "search_energy_data" in called and not _gold_in_evidence(item, outs):
        if not set(item["gold_doc_ids"]) <= set(retrieved_doc_ids(outs)):
            return "retrieval_miss"
    if _gold_in_evidence(item, outs):
        if is_refusal(answer):
            return "false_refusal"
        return "extraction_error"
    if is_refusal(answer):
        return "false_refusal"
    if groundedness(answer, outs, item["question"])["unsupported"]:
        return "hallucination"
    return "other"
