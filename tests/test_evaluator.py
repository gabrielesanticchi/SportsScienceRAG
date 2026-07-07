import json

from sportsscience_rag.evaluator import GoldQuery, build_run_dict, load_gold
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
