from types import SimpleNamespace
from unittest.mock import MagicMock

from sportsscience_rag.retriever import RetrievedChunk, Retriever


def _point(score, source, idx, pages, section, text):
    return SimpleNamespace(
        score=score,
        payload={
            "source": source,
            "chunk_index": idx,
            "page_numbers": pages,
            "section_path": section,
            "text": text,
        },
    )


def _embedder():
    emb = MagicMock()
    emb.embed.return_value = [[0.1] * 384]
    return emb


def test_search_embeds_query_and_calls_query_points_with_named_vector():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[])
    emb = _embedder()
    r = Retriever(client, emb, "coll")

    r.search("training load", limit=7)

    emb.embed.assert_called_once_with(["training load"])
    kwargs = client.query_points.call_args.kwargs
    assert kwargs["collection_name"] == "coll"
    assert kwargs["using"] == "text_embedding"
    assert kwargs["limit"] == 7
    assert kwargs["query"] == [0.1] * 384
    assert kwargs["with_payload"] is True


def test_search_maps_points_to_ranked_chunks():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[
        _point(0.82, "s3://b/gabbett.pdf", 3, [2], "# Intro", "load and injury"),
        _point(0.71, "s3://b/disalvo.pdf", 0, [1, 2], "# Methods", "high-intensity running"),
    ])
    r = Retriever(client, _embedder(), "coll")

    hits = r.search("q", limit=10)

    assert [h.rank for h in hits] == [1, 2]
    assert hits[0] == RetrievedChunk(
        rank=1, score=0.82, source="s3://b/gabbett.pdf", chunk_index=3,
        page_numbers=(2,), section_path="# Intro", text="load and injury",
    )
    assert hits[1].page_numbers == (1, 2)
    assert hits[1].source == "s3://b/disalvo.pdf"


def test_search_handles_missing_optional_payload_fields():
    client = MagicMock()
    client.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(score=0.5, payload={"source": "s3://b/x.pdf", "chunk_index": 0, "text": "t"}),
    ])
    r = Retriever(client, _embedder(), "coll")

    hits = r.search("q")

    assert hits[0].page_numbers == ()
    assert hits[0].section_path == ""
    assert hits[0].text == "t"
