"""Constraint extraction (pure) and retrieval behaviour (against the offline index)."""
import pytest

from src.retrieval.smart_retrieve import (
    NoMatchingDocuments,
    extract_constraints,
    extract_constraints_v2,
    format_results_for_agent,
    parse_dates,
    parse_months,
    smart_retrieve,
    smart_retrieve_v2,
)


# ----------------------------------------------------------------------------- v1 regressions
def test_v1_substring_bug_is_documented():
    """'ev' inside 'every' made v1 apply the EV source filter. v2 must not."""
    q = "Was every day in July 2023 sunny?"
    assert extract_constraints(q).get("source") == "ev_grid"
    assert extract_constraints_v2(q).get("source") == "solar"


def test_v1_uses_only_first_date():
    q = "Compare GRID_LV22 on 2020-08-11 and 2020-08-12"
    assert extract_constraints(q)["date"] == "2020-08-11"
    assert extract_constraints_v2(q)["date"] == ["2020-08-11", "2020-08-12"]


# ----------------------------------------------------------------------------- v2 parsing
@pytest.mark.parametrize("text,expected", [
    ("on 2020-08-11", ["2020-08-11"]),
    ("on August 11, 2020", ["2020-08-11"]),
    ("on 11 August 2020", ["2020-08-11"]),
    ("on Aug 11 2020", ["2020-08-11"]),
    ("on 8/11/2020", ["2020-08-11"]),
    ("on Sept. 3rd, 2023", ["2023-09-03"]),
    ("between 2023-01-02 and January 5, 2023", ["2023-01-02", "2023-01-05"]),
    ("no date here", []),
])
def test_parse_dates(text, expected):
    assert parse_dates(text) == expected


def test_parse_months_ignores_full_dates():
    assert parse_months("in July 2023") == ["2023-07"]
    assert parse_months("in 2020-08") == ["2020-08"]
    assert parse_months("on July 15, 2023") == []


def test_v2_multiple_nodes_and_source():
    c = extract_constraints_v2("grid_lv22 vs GRID_08A on 2020-01-02")
    assert c["grid_node"] == ["GRID_LV22", "GRID_08A"]
    assert c["source"] == "ev_grid"


def test_v2_invalid_date_flagged():
    assert extract_constraints_v2("GRID_LV22 on 2020-02-30")["invalid_date"] == ["2020-02-30"]


# ----------------------------------------------------------------------------- retrieval (index)
def test_v2_exact_lookup(vectorstore):
    docs = smart_retrieve_v2("GRID_LV22 on 2020-08-11", vectorstore)
    assert docs[0].metadata["doc_id"] == "ev_grid|GRID_LV22|2020-08-11"


def test_v2_natural_language_date(vectorstore):
    docs = smart_retrieve_v2("peak solar irradiance on August 3, 2023", vectorstore)
    assert docs[0].metadata["doc_id"] == "solar|2023-08-03"


def test_v2_comparison_returns_both(vectorstore):
    docs = smart_retrieve_v2("GRID_LV22 on 2020-08-11 and 2020-08-12", vectorstore)
    ids = {d.metadata["doc_id"] for d in docs}
    assert {"ev_grid|GRID_LV22|2020-08-11", "ev_grid|GRID_LV22|2020-08-12"} <= ids


@pytest.mark.parametrize("q", [
    "GRID_LV22 on 2018-01-01",        # before data
    "GRID_ZZ9 on 2020-08-11",         # unknown node
    "GRID_LV22 on 2020-02-30",        # impossible date
])
def test_v2_no_silent_fallback(vectorstore, q):
    docs = smart_retrieve_v2(q, vectorstore)
    assert len(docs) == 1 and isinstance(docs[0], NoMatchingDocuments)
    assert format_results_for_agent(docs).startswith("No documents match")


def test_v1_silent_fallback_still_present(vectorstore):
    """Baseline behaviour kept for the eval: v1 returns other dates' documents."""
    docs = smart_retrieve("GRID_LV22 on 2018-01-01", vectorstore)
    assert docs and all(d.metadata["date"] != "2018-01-01" for d in docs)
