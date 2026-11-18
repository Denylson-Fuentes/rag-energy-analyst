"""Integrity checks for the evaluation set and the harness."""
from collections import Counter

from src import config
from src.ingestion.loaders import load_ev_grid_documents, load_solar_documents

ANSWER_TYPES = {"number", "numbers_all", "text", "text_all", "date", "dates_all", "time", "refusal"}


def test_size_and_ids(questions):
    assert len(questions) >= 150
    assert len({q["id"] for q in questions}) == len(questions)
    assert len({q["question"] for q in questions}) == len(questions)


def test_category_coverage(questions):
    counts = Counter(q["category"] for q in questions)
    assert set(counts) == {"ev_lookup", "ev_peak_hour", "solar_lookup", "solar_cloud", "nl_date",
                           "comparison", "aggregation", "forecast", "inventory", "unanswerable"}
    assert min(counts.values()) >= 5


def test_schema(questions):
    for q in questions:
        assert q["answer_type"] in ANSWER_TYPES, q["id"]
        assert (q["answer_type"] == "refusal") == (q["gold"] is None), q["id"]
        assert q["retrieval_applicable"] == bool(q["gold_doc_ids"]), q["id"]


def test_gold_docs_exist(questions):
    ids = {d.metadata["doc_id"] for d in load_ev_grid_documents(config.EV_CSV)}
    ids |= {d.metadata["doc_id"] for d in load_solar_documents(config.SOLAR_CSV)}
    for q in questions:
        for d in q["gold_doc_ids"]:
            assert d in ids, (q["id"], d)


def test_dataset_is_reproducible(questions):
    from evals.build_dataset import build

    rebuilt = build()
    assert [q["question"] for q in rebuilt] == [q["question"] for q in questions]
    assert [q["gold"] for q in rebuilt] == [q["gold"] for q in questions]


def test_harness_end_to_end_with_scripted_model(questions, vectorstore):
    """Agent loop -> trace capture -> scoring, with a scripted model (no LLM)."""
    from evals.run_agent_eval import run_variant, score

    sample = [q for q in questions if q["category"] in ("ev_lookup", "aggregation", "forecast")][::6]
    recs = run_variant("agent_v2", sample, "scripted", workers=4)
    assert not [r for r in recs if r.get("error")]
    s = score(sample, recs, "scripted")
    assert s["accuracy"] == 1.0
    assert s["groundedness_claim_level"] == 1.0
