"""The scorers are the measuring instrument; they get the most tests."""
import pytest

from evals.scoring import (
    answer_correct,
    classify,
    extract_numbers,
    groundedness,
    is_refusal,
    retrieved_doc_ids,
)


def test_extract_numbers_skips_ids_dates_times_citations():
    text = ("GRID_LV22 on 2020-08-11: **97.17 kWh** [Source 1]. Peak at 09:00 (43.65 kWh). "
            "August 2020 total 2,005.30 kWh; July 15, 2023 peaked at 988 W/m².")
    assert extract_numbers(text) == [97.17, 43.65, 2005.30, 988.0]


@pytest.mark.parametrize("text,expected", [
    ("I don't have data for GRID_ZZ9.", True),
    ("There is no data available for 2018-03-14.", True),
    ("2020-02-30 is not a valid date.", True),
    ("The dataset does not include electricity prices.", True),
    ("Total energy was 97.17 kWh [Source 1].", False),
])
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected


def _item(t, gold, tol=None, **kw):
    return {"answer_type": t, "gold": gold, "abs_tol": tol, "question": "", "expected_tools": [],
            "retrieval_applicable": False, "gold_doc_ids": [], "category": "x", **kw}


@pytest.mark.parametrize("item,answer,ok", [
    (_item("number", 97.17, 0.1), "It was **97.2 kWh**.", True),
    (_item("number", 97.17, 0.1), "It was 96.5 kWh.", False),
    (_item("number", 1586.36, 0.5), "GRID_LV22 used 1,586.4 kWh", True),
    (_item("numbers_all", [10.0, 20.5], 0.1), "10.0 vs 20.5 kWh", True),
    (_item("numbers_all", [10.0, 20.5], 0.1), "only 10.0 kWh", False),
    (_item("text", "GRID_LV22"), "The top node was grid_lv22.", True),
    (_item("date", "2023-07-15"), "July 15 had the highest peak.", True),
    (_item("date", "2023-07-15"), "July 16, 2023 was highest.", False),
    (_item("dates_all", ["2019-04-01", "2021-10-11"]), "From 2019-04-01 to October 11, 2021", True),
    (_item("time", "09:00"), "Charging peaked at 9 AM.", True),
    (_item("time", "13:00"), "The peak was at 1 pm.", True),
    (_item("time", "13:00"), "The peak was at 11:00.", False),
    (_item("refusal", None), "No data is available for that date.", True),
    (_item("refusal", None), "It was 45.2 kWh.", False),
])
def test_answer_correct(item, answer, ok):
    assert answer_correct(item, answer) is ok


def test_groundedness_allows_rounding_and_arithmetic():
    g = groundedness("A is 97.2 kWh, B is 43.65, so A is 53.52 kWh higher (122.6%). Also 999 kWh.",
                     ["Total Energy: 97.17 kWh", "Total Energy: 43.65 kWh"])
    assert g["n_claims"] == 5
    assert g["unsupported"] == [999.0]


def test_retrieved_doc_ids():
    outs = ["[Source 1] ev_grid | Grid Node: GRID_LV22 | Date: 2020-08-11\n...",
            "[Source 1] solar | Location: Palm Springs | Date: 2023-07-15\n..."]
    assert retrieved_doc_ids(outs) == ["ev_grid|GRID_LV22|2020-08-11", "solar|2023-07-15"]


def _rec(answer, calls=(), outs=(), error=None):
    return {"answer": answer, "tool_calls": [{"name": c, "args": {}} for c in calls],
            "tool_outputs": list(outs), "error": error}


LOOKUP = _item("number", 97.17, 0.1, retrieval_applicable=True, expected_tools=["search_energy_data"],
               gold_doc_ids=["ev_grid|GRID_LV22|2020-08-11"], question="GRID_LV22 on 2020-08-11?")
GOOD_OUT = "[Source 1] ev_grid | Grid Node: GRID_LV22 | Date: 2020-08-11\nTotal Energy: 97.17 kWh"
WRONG_OUT = "[Source 1] ev_grid | Grid Node: GRID_LV22 | Date: 2021-01-01\nTotal Energy: 12.00 kWh"


@pytest.mark.parametrize("item,rec,label", [
    (LOOKUP, _rec("97.17 kWh", ["search_energy_data"], [GOOD_OUT]), "ok"),
    (LOOKUP, _rec("", error="Timeout"), "error"),
    (LOOKUP, _rec("About 80 kWh"), "no_tool_call"),
    (LOOKUP, _rec("12.00 kWh", ["search_energy_data"], [WRONG_OUT]), "retrieval_miss"),
    (LOOKUP, _rec("It was 91.17 kWh", ["search_energy_data"], [GOOD_OUT]), "extraction_error"),
    (LOOKUP, _rec("No data is available.", ["search_energy_data"], [GOOD_OUT]), "false_refusal"),
    (LOOKUP, _rec("x", ["forecast_solar"], ["MAE 3"]), "wrong_tool"),
    (_item("refusal", None), _rec("It was 45 kWh", ["search_energy_data"], [WRONG_OUT]), "missed_refusal"),
])
def test_classify(item, rec, label):
    assert classify(item, rec) == label
