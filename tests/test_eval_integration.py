"""End-to-end retrieval + eval smoke test against the live collection.

Requires real AWS/Qdrant credentials in the project-root .env and network
access; deselected by default (`pytest -m "not integration"`).
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from sportsscience_rag.embedder import TextEmbedder
from sportsscience_rag.evaluator import build_run_dict, evaluate_run, load_gold
from sportsscience_rag.retriever import Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_baseline_mrr_above_floor():
    load_dotenv(PROJECT_ROOT / ".env")
    collection = os.getenv("QDRANT_COLLECTION", "sport-science-documents")
    client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    retriever = Retriever(client, TextEmbedder(), collection)

    queries, qrels = load_gold(PROJECT_ROOT / "eval" / "gold.jsonl")
    hits_by_query = {g.query: retriever.search(g.query, limit=10) for g in queries}
    run_dict = build_run_dict(hits_by_query)
    report = evaluate_run(qrels, run_dict, queries)

    # A correctly wired baseline should comfortably beat random on known-item
    # queries. This is a wiring smoke test, not a quality gate — keep the floor
    # loose so it does not flake on scoring noise.
    assert report.n_queries >= 20
    assert report.metrics["mrr"] > 0.5
    assert report.metrics["recall@10"] > 0.6
