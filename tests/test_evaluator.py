import json

from sportsscience_rag.evaluator import (
    EvalReport,
    GoldQuery,
    build_run_dict,
    evaluate_run,
    load_gold,
)
from sportsscience_rag.retriever import RetrievedChunk


def _chunk(source, score, idx=0):
    return RetrievedChunk(
        rank=idx + 1, score=score, source=source, chunk_index=idx,
        page_numbers=(), section_path="", text="",
    )


def test_build_run_dict_keeps_max_score_per_filename():
    hits = {
        "q1": [
            _chunk("s3://b/gabbett.pdf", 0.5, 0),
            _chunk("s3://b/gabbett.pdf", 0.9, 1),   # same paper, higher score wins
            _chunk("s3://b/disalvo.pdf", 0.7, 2),
        ]
    }
    run = build_run_dict(hits)
    assert run == {"q1": {"gabbett.pdf": 0.9, "disalvo.pdf": 0.7}}


def test_build_run_dict_handles_empty_hits():
    assert build_run_dict({"q1": []}) == {"q1": {}}


def test_load_gold_parses_graded_records(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text(
        json.dumps({"query": "training load", "relevant": {"gabbett.pdf": 3}, "notes": "n"})
        + "\n"
        + json.dumps({"query": "hir soccer", "relevant": {"disalvo.pdf": 3, "bradley.pdf": 2}})
        + "\n",
        encoding="utf-8",
    )
    queries, qrels = load_gold(p)
    assert queries[0] == GoldQuery(query="training load", relevant={"gabbett.pdf": 3}, notes="n")
    assert queries[1].notes == ""  # missing notes defaults to empty string
    assert qrels == {
        "training load": {"gabbett.pdf": 3},
        "hir soccer": {"disalvo.pdf": 3, "bradley.pdf": 2},
    }


def test_load_gold_skips_blank_lines(tmp_path):
    p = tmp_path / "gold.jsonl"
    p.write_text(
        json.dumps({"query": "q", "relevant": {"a.pdf": 1}}) + "\n\n  \n",
        encoding="utf-8",
    )
    queries, qrels = load_gold(p)
    assert len(queries) == 1


def test_evaluate_run_computes_metrics_and_per_query():
    queries = [
        GoldQuery(query="q1", relevant={"a.pdf": 3}, notes="known-item"),
        GoldQuery(query="q2", relevant={"b.pdf": 3, "c.pdf": 2}, notes=""),
    ]
    qrels = {"q1": {"a.pdf": 3}, "q2": {"b.pdf": 3, "c.pdf": 2}}
    # q1: a.pdf ranked 1 (perfect). q2: b.pdf ranked 2, x.pdf ranked 1 (irrelevant).
    run = {
        "q1": {"a.pdf": 0.9, "z.pdf": 0.4},
        "q2": {"x.pdf": 0.8, "b.pdf": 0.7, "c.pdf": 0.3},
    }
    report = evaluate_run(qrels, run, queries)

    assert isinstance(report, EvalReport)
    assert report.n_queries == 2
    # every configured metric is present and in [0, 1]
    for name in ["mrr", "recall@5", "recall@10", "hit_rate@1", "ndcg@10"]:
        assert name in report.metrics
        assert 0.0 <= report.metrics[name] <= 1.0
    # q1 got its relevant paper at rank 1; q2's best relevant paper is at rank 2
    by_query = {row["query"]: row for row in report.per_query}
    assert by_query["q1"]["best_rank"] == 1
    assert by_query["q2"]["best_rank"] == 2
    assert by_query["q1"]["notes"] == "known-item"


def test_evaluate_run_reports_miss_as_none_rank():
    queries = [GoldQuery(query="q1", relevant={"a.pdf": 3})]
    qrels = {"q1": {"a.pdf": 3}}
    run = {"q1": {"other.pdf": 0.9}}  # relevant paper not retrieved at all
    report = evaluate_run(qrels, run, queries)
    row = report.per_query[0]
    assert row["best_rank"] is None
    assert report.metrics["hit_rate@1"] == 0.0
