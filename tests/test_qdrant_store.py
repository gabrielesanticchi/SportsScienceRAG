from unittest.mock import MagicMock

from sportsscience_rag.models import Chunk
from sportsscience_rag.qdrant_store import QdrantStore, point_id


def test_point_id_is_deterministic_uuid():
    a = point_id("hash123", 0)
    b = point_id("hash123", 0)
    c = point_id("hash123", 1)
    assert a == b
    assert a != c
    assert len(a) == 36  # uuid string


def test_ensure_collection_creates_when_absent():
    client = MagicMock()
    client.collection_exists.return_value = False
    QdrantStore(client, "coll").ensure_collection()
    client.create_collection.assert_called_once()


def test_ensure_collection_skips_when_present():
    client = MagicMock()
    client.collection_exists.return_value = True
    QdrantStore(client, "coll").ensure_collection()
    client.create_collection.assert_not_called()


def test_already_ingested_true_when_points_found():
    client = MagicMock()
    client.count.return_value = MagicMock(count=3)
    assert QdrantStore(client, "coll").already_ingested("h", "pv", "cc") is True


def test_already_ingested_false_when_zero():
    client = MagicMock()
    client.count.return_value = MagicMock(count=0)
    assert QdrantStore(client, "coll").already_ingested("h", "pv", "cc") is False


def test_upsert_empty_chunks_returns_zero():
    client = MagicMock()
    n = QdrantStore(client, "coll").upsert("h", "s", "pv", "cc", [], [])
    assert n == 0
    client.upsert.assert_not_called()


def test_upsert_builds_points_with_payload():
    client = MagicMock()
    store = QdrantStore(client, "coll")
    chunks = [Chunk(index=0, text="t0", section_path="Intro", page_numbers=(1,))]
    n = store.upsert("h", "s3://b/k.pdf", "pv", "cc", chunks, [[0.1] * 384])
    assert n == 1
    args, kwargs = client.upsert.call_args
    points = kwargs["points"]
    assert points[0].payload["content_hash"] == "h"
    assert points[0].payload["section_path"] == "Intro"
    assert points[0].payload["text"] == "t0"
    assert "text_embedding" in points[0].vector
