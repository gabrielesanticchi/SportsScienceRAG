"""Graded IR evaluation of the retriever against a gold set (ranx-based)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sportsscience_rag.retriever import RetrievedChunk

METRICS = ["mrr", "recall@5", "recall@10", "hit_rate@1", "ndcg@10"]


@dataclass(frozen=True)
class GoldQuery:
    """A single graded gold record.

    Attributes:
        query: The query string.
        relevant: Map of bare filename -> integer relevance grade (1-3).
        notes: Free-text annotation (e.g., "known-item primary = di Salvo").
    """

    query: str
    relevant: dict[str, int]
    notes: str = ""


def load_gold(path: str | Path) -> tuple[list[GoldQuery], dict[str, dict[str, int]]]:
    """Load a graded gold set from a JSONL file.

    Each non-blank line is a JSON object with keys ``query`` (str),
    ``relevant`` (map of bare filename -> grade), and optional ``notes``.

    Args:
        path: Path to the ``gold.jsonl`` file.

    Returns:
        A tuple of (list of ``GoldQuery``, ranx qrels dict keyed by query).
    """
    queries: list[GoldQuery] = []
    qrels: dict[str, dict[str, int]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        gold = GoldQuery(
            query=record["query"],
            relevant={k: int(v) for k, v in record["relevant"].items()},
            notes=record.get("notes", ""),
        )
        queries.append(gold)
        qrels[gold.query] = dict(gold.relevant)
    return queries, qrels


def build_run_dict(
    hits_by_query: dict[str, list[RetrievedChunk]],
) -> dict[str, dict[str, float]]:
    """Collapse chunk-level hits to a paper-level ranx run dict.

    A paper is identified by the bare filename of its ``source``. When several
    chunks of the same paper are retrieved for a query, the paper's score is the
    **maximum** chunk score (its best-matching chunk).

    Args:
        hits_by_query: Map of query string -> ranked ``RetrievedChunk`` list.

    Returns:
        Map of query -> {bare_filename: max_score} suitable for ``ranx.Run``.
    """
    run: dict[str, dict[str, float]] = {}
    for query, hits in hits_by_query.items():
        best: dict[str, float] = {}
        for hit in hits:
            filename = hit.source.rsplit("/", 1)[-1]
            if filename not in best or hit.score > best[filename]:
                best[filename] = hit.score
        run[query] = best
    return run
