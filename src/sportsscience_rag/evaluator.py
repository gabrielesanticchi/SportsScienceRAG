"""Graded IR evaluation of the retriever against a gold set (ranx-based)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ranx import Qrels, Run, evaluate

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


@dataclass(frozen=True)
class EvalReport:
    """Aggregated evaluation results.

    Attributes:
        metrics: Map of ranx metric name -> macro-averaged score.
        per_query: One row per query with best-relevant-paper rank diagnostics.
        n_queries: Number of queries evaluated.
    """

    metrics: dict[str, float]
    per_query: list[dict]
    n_queries: int


def _best_relevant_rank(
    ranked_filenames: list[str], relevant: dict[str, int]
) -> int | None:
    """Return the 1-indexed rank of the highest-ranked relevant paper, or None.

    Args:
        ranked_filenames: Retrieved filenames ordered by descending score.
        relevant: Map of relevant filename -> grade for this query.

    Returns:
        The 1-indexed rank of the first relevant paper, or ``None`` if none of
        the relevant papers were retrieved.
    """
    for rank, filename in enumerate(ranked_filenames, start=1):
        if filename in relevant:
            return rank
    return None


def evaluate_run(
    qrels_dict: dict[str, dict[str, int]],
    run_dict: dict[str, dict[str, float]],
    queries: list[GoldQuery],
) -> EvalReport:
    """Score a run dict against graded qrels using ranx.

    Args:
        qrels_dict: Graded relevance judgments {query: {filename: grade}}.
        run_dict: Retrieval results {query: {filename: score}}.
        queries: Gold queries (used for per-query notes and rank diagnostics).

    Returns:
        An ``EvalReport`` with macro-averaged metrics and per-query rows.
    """
    qrels = Qrels(qrels_dict)
    run = Run(run_dict, name="baseline")
    scores = evaluate(qrels, run, METRICS)
    metrics = {name: float(scores[name]) for name in METRICS}

    per_query: list[dict] = []
    for gold in queries:
        ranked = sorted(
            run_dict.get(gold.query, {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        ranked_filenames = [filename for filename, _ in ranked]
        best_rank = _best_relevant_rank(ranked_filenames, gold.relevant)
        per_query.append(
            {
                "query": gold.query,
                "best_rank": best_rank,
                "notes": gold.notes,
            }
        )
    return EvalReport(metrics=metrics, per_query=per_query, n_queries=len(queries))
