# Retrieval Inference & Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a document-level retrieval path (`Retriever` + `search` CLI) and a graded IR evaluation harness (`ranx`-based `evaluator` + `eval` CLI + committed gold set) over the existing `sport-science-documents` Qdrant collection.

**Architecture:** A thin `Retriever` class owns query embedding → `query_points(using="text_embedding")` → payload-to-`RetrievedChunk` mapping, returning **chunk-level** ranked results. The `evaluator` sits on top: it dedups a retriever's chunk hits to **paper-level** (max score per `source`), builds `ranx` `Qrels`/`Run` objects, and reports graded IR metrics. The existing ingest CLI moves under an `ingest` subcommand; `search` and `eval` are new subcommands on the same entry point.

**Tech Stack:** Python 3.12+, `qdrant-client`, `fastembed` (MiniLM 384-d, already used for ingestion), `ranx` 0.3.21 (graded IR metrics), argparse subcommands, pytest.

## Global Constraints

- Python `>=3.12` (per `pyproject.toml`); code is type-hinted, PEP8, Google-style docstrings, frozen dataclasses for data structures.
- Retrieval MUST use the existing named vector: `VECTOR_NAME = "text_embedding"` (from `sportsscience_rag.qdrant_store`), 384-dim, cosine.
- Embedding model MUST match ingestion: FastEmbed `sentence-transformers/all-MiniLM-L6-v2` via the existing `sportsscience_rag.embedder.TextEmbedder`. Do NOT introduce a second embedding model.
- Scoring unit is **document (paper)-level**; paper identity is the payload `source` field (one paper per source — the duplicate has already been removed from the live collection, total = 2097 points, 27 sources).
- Gold set is graded: `relevant` maps `{bare_filename: grade}` where grade ∈ {1,2,3} (3 = primary source, 2 = substantive, 1 = peripheral; unlabeled = 0/omitted).
- Metrics reported: `mrr`, `recall@5`, `recall@10`, `hit_rate@1`, `ndcg@10` — computed by `ranx`. No hand-rolled metric math.
- No `numpy`/`pandas` added as direct project deps for metric math (ranx pulls its own transitively; that is fine — do not import them in our code).
- The run dict passed to `ranx.Run` maps `{query: {source: max_score}}` — dedup chunks to **max** score per source, not last-write.
- `search` CLI output is **chunk-level** (raw retriever output). Paper-dedup happens ONLY inside `eval`.
- Unit tests mock the Qdrant client and embedder (no network). Integration-touching tests are marked `@pytest.mark.integration` (existing convention in `pyproject.toml`).
- Commit after every green task. Never use `--no-verify`. Run `git add`/`git commit` as separate Bash calls.
- Environment (AWS + Qdrant creds) is loaded from the project-root `.env` via `IngestionConfig.from_env(env_path=PROJECT_ROOT / ".env")` — never hardcode secrets.

---

## File Structure

- **Create** `src/sportsscience_rag/retriever.py` — `RetrievedChunk` (frozen dataclass) + `Retriever` class. Chunk-level semantic search over the collection.
- **Create** `src/sportsscience_rag/evaluator.py` — pure functions: `build_run_dict()` (retriever hits → `{query: {source: max_score}}`), `load_gold()` (parse `gold.jsonl` → qrels dict + query list), `evaluate_run()` (call ranx). Plus `EvalReport` frozen dataclass.
- **Create** `eval/gold.jsonl` — committed graded gold set (~30 queries).
- **Create** `eval/README.md` — the grading rubric (so future grading is consistent).
- **Modify** `src/sportsscience_rag/cli.py` — refactor flat parser into `ingest`/`search`/`eval` subcommands; wire `Retriever` and `evaluator`.
- **Modify** `tests/test_cli.py` — update for subcommand parser.
- **Modify** `.vscode/launch.json` — update the ingest debug config to the `ingest` subcommand; add a `search` debug config.
- **Modify** `pyproject.toml` — add `ranx>=0.3.21` to an `eval` optional-dependency group.
- **Modify** `README.md` — document `search`/`eval` subcommands and the eval harness (final task).
- **Create** `tests/test_retriever.py`, `tests/test_evaluator.py` — unit tests (mocked).
- **Create** `tests/test_eval_integration.py` — one `@pytest.mark.integration` end-to-end smoke test.

Bare-filename ↔ full-source resolution: the collection stores `source` as `s3://<bucket>/<filename>`. The evaluator resolves a gold bare filename to the payload source by **suffix match** (`source.rsplit("/", 1)[-1] == filename`), so gold stays bucket-agnostic.

---

### Task 1: Add `ranx` eval dependency group

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable `eval` extra so `ranx` is declared, not just ad-hoc installed.

- [ ] **Step 1: Add the optional-dependency group**

In `pyproject.toml`, under `[project.optional-dependencies]`, add an `eval` group alongside the existing `dev` group so the block reads:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]
eval = ["ranx>=0.3.21"]
```

- [ ] **Step 2: Verify ranx imports in the venv**

Run: `source .venv/bin/activate && python -c "from ranx import Qrels, Run, evaluate, compare; print('ok')"`
Expected: prints `ok` (ranx 0.3.21 is already installed in the venv; this only confirms the declared floor matches reality). SyntaxWarnings from ranx's own LaTeX strings on Python 3.14 are harmless and may be ignored.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: declare ranx as an eval optional-dependency"
```

---

### Task 2: `RetrievedChunk` model + `Retriever.search()`

**Files:**
- Create: `src/sportsscience_rag/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `sportsscience_rag.qdrant_store.VECTOR_NAME` (`"text_embedding"`); an embedder with `.embed(list[str]) -> list[list[float]]` (satisfied by `TextEmbedder`); a Qdrant client with `.query_points(collection_name, query, using, limit, with_payload) -> object with .points` where each point has `.score` and `.payload`.
- Produces:
  - `RetrievedChunk` frozen dataclass with fields: `rank: int` (1-indexed), `score: float`, `source: str`, `chunk_index: int`, `page_numbers: tuple[int, ...]`, `section_path: str`, `text: str`.
  - `Retriever(client, embedder, collection: str)` with method `search(query: str, limit: int = 10) -> list[RetrievedChunk]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retriever.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sportsscience_rag.retriever'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sportsscience_rag/retriever.py`:

```python
"""Chunk-level semantic retrieval over the Qdrant collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sportsscience_rag.qdrant_store import VECTOR_NAME


@dataclass(frozen=True)
class RetrievedChunk:
    """A single ranked chunk returned by a retrieval query.

    Attributes:
        rank: 1-indexed position in the returned ranking.
        score: Similarity score from Qdrant (cosine).
        source: Source identifier of the document (e.g., S3 URI).
        chunk_index: Sequence number of the chunk within its document.
        page_numbers: 1-indexed page numbers the chunk spans; empty if unknown.
        section_path: Hierarchical heading path of the chunk.
        text: Chunk content text.
    """

    rank: int
    score: float
    source: str
    chunk_index: int
    page_numbers: tuple[int, ...]
    section_path: str
    text: str


class Retriever:
    """Embeds a query and returns chunk-level ranked results from Qdrant.

    Attributes:
        _client: Qdrant client used to run the vector query.
        _embedder: Embedder exposing ``embed(list[str]) -> list[list[float]]``.
        _collection: Name of the collection to query.
    """

    def __init__(self, client: Any, embedder: Any, collection: str) -> None:
        """Initialize the retriever.

        Args:
            client: Qdrant client instance.
            embedder: Embedder with an ``embed`` method matching ``TextEmbedder``.
            collection: Target Qdrant collection name.
        """
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def search(self, query: str, limit: int = 10) -> list[RetrievedChunk]:
        """Embed ``query`` and return the top ``limit`` chunks, ranked.

        Args:
            query: Natural-language query string.
            limit: Maximum number of chunks to return.

        Returns:
            A list of ``RetrievedChunk`` ordered by descending score, with
            1-indexed ``rank``.
        """
        vector = self._embedder.embed([query])[0]
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        chunks: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    rank=rank,
                    score=float(point.score),
                    source=payload.get("source", ""),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    page_numbers=tuple(payload.get("page_numbers", ()) or ()),
                    section_path=payload.get("section_path", ""),
                    text=payload.get("text", ""),
                )
            )
        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_retriever.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/retriever.py tests/test_retriever.py
git commit -m "feat: add chunk-level Retriever and RetrievedChunk model"
```

---

### Task 3: Evaluator — gold loading + run-dict builder (TDD, pure functions)

**Files:**
- Create: `src/sportsscience_rag/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `RetrievedChunk` from Task 2.
- Produces:
  - `load_gold(path: str | Path) -> tuple[list[GoldQuery], dict[str, dict[str, int]]]` where `GoldQuery` is a frozen dataclass `(query: str, relevant: dict[str, int], notes: str)` and the second element is the ranx qrels dict `{query: {filename: grade}}`.
  - `GoldQuery` frozen dataclass: `query: str`, `relevant: dict[str, int]`, `notes: str`.
  - `build_run_dict(hits_by_query: dict[str, list[RetrievedChunk]]) -> dict[str, dict[str, float]]` — dedups chunks to **max score per bare filename** (resolves `source` → bare filename via `rsplit("/", 1)[-1]`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_evaluator.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sportsscience_rag.evaluator'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/sportsscience_rag/evaluator.py`:

```python
"""Graded IR evaluation of the retriever against a gold set (ranx-based)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
```

(Note: `field` import is unused for now but keeps room for future defaults; remove it if the linter objects — replace the import line with `from dataclasses import dataclass`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_evaluator.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/evaluator.py tests/test_evaluator.py
git commit -m "feat: add evaluator gold-loading and paper-level run-dict builder"
```

---

### Task 4: Evaluator — `evaluate_run()` and `EvalReport` (ranx integration)

**Files:**
- Modify: `src/sportsscience_rag/evaluator.py`
- Test: `tests/test_evaluator.py`

**Interfaces:**
- Consumes: `load_gold`, `build_run_dict`, `METRICS` from Task 3; `ranx.Qrels`, `ranx.Run`, `ranx.evaluate`.
- Produces:
  - `EvalReport` frozen dataclass: `metrics: dict[str, float]` (metric name -> score), `per_query: list[dict]` (each `{query, best_rank, best_relevant_score, notes}`), `n_queries: int`.
  - `evaluate_run(qrels_dict: dict, run_dict: dict, queries: list[GoldQuery]) -> EvalReport`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluator.py`:

```python
from sportsscience_rag.evaluator import EvalReport, evaluate_run


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_evaluator.py::test_evaluate_run_computes_metrics_and_per_query -v`
Expected: FAIL — `ImportError: cannot import name 'evaluate_run'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/sportsscience_rag/evaluator.py` (new imports at top, new dataclass + function at end):

```python
# add to the top-of-file imports:
from ranx import Qrels, Run, evaluate
```

```python
# add at end of file:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_evaluator.py -v 2>/dev/null`
Expected: PASS (6 tests). (Redirect stderr to hide ranx's harmless SyntaxWarnings.)

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/evaluator.py tests/test_evaluator.py
git commit -m "feat: add ranx-based evaluate_run and EvalReport"
```

---

### Task 5: Generate ~30 graded gold queries (LLM-seeded, human-verified) + rubric

**Files:**
- Create: `eval/gold.jsonl`
- Create: `eval/README.md`

**Interfaces:**
- Consumes: the live collection payloads (to draw candidate queries) and the loaded corpus filenames.
- Produces: a committed `eval/gold.jsonl` loadable by `load_gold` (Task 3), and `eval/README.md` documenting the grading rubric.

> This task is data curation, not code. It is one deliverable (the gold set) but requires a human-verification checkpoint. The executing agent MUST surface the drafted queries to the user for correction before committing — do not commit LLM-drafted grades unverified.

- [ ] **Step 1: List the 27 corpus filenames from the live collection**

Run:
```bash
source .venv/bin/activate && python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")
from qdrant_client import QdrantClient
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
coll = os.getenv("QDRANT_COLLECTION", "sport-science-documents")
srcs = set()
offset = None
while True:
    pts, offset = c.scroll(coll, limit=512, offset=offset, with_payload=["source"], with_vectors=False)
    srcs.update(p.payload["source"].rsplit("/", 1)[-1] for p in pts)
    if offset is None:
        break
for s in sorted(srcs):
    print(s)
print("TOTAL", len(srcs))
PY
```
Expected: 27 filenames + `TOTAL 27`.

- [ ] **Step 2: Draft ~40 candidate graded queries spanning the corpus**

For coverage, sample chunk text across many distinct papers (not just a few). For each candidate, draft a natural query a sports scientist would ask, and assign the primary paper grade `3` plus any secondary papers `2`/`1`. Aim for ~40 candidates covering as many of the 27 papers as possible; mostly known-item (single `3`), with ~5–8 topical (multi-paper).

Use this helper to pull representative chunk text per paper as drafting material:
```bash
source .venv/bin/activate && python - <<'PY'
import os
from dotenv import load_dotenv
load_dotenv(".env")
from qdrant_client import QdrantClient
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
coll = os.getenv("QDRANT_COLLECTION", "sport-science-documents")
seen = {}
offset = None
while True:
    pts, offset = c.scroll(coll, limit=512, offset=offset, with_payload=True, with_vectors=False)
    for p in pts:
        src = p.payload["source"].rsplit("/", 1)[-1]
        if p.payload.get("chunk_index") in (1, 2) and src not in seen:
            seen[src] = (p.payload.get("text") or "")[:400].replace("\n", " ")
    if offset is None:
        break
for src in sorted(seen):
    print(f"### {src}\n{seen[src]}\n")
PY
```

- [ ] **Step 3: HUMAN CHECKPOINT — present drafts, collect corrections**

Present the ~40 drafted `{query, relevant, notes}` records to the user. The user keeps/edits/relabels down to a curated ~30. Apply their corrections verbatim. Do not proceed until the user confirms the set.

- [ ] **Step 4: Write `eval/gold.jsonl`**

Write one JSON object per line, e.g.:
```json
{"query": "high-intensity running distance in elite soccer match-play", "relevant": {"di_salvo_2007_...pdf": 3}, "notes": "known-item primary"}
{"query": "acute chronic workload ratio and injury risk", "relevant": {"Gabbett_2016_Training-injuryparadox_Trainsmarterandharder_BJSM.pdf": 3}, "notes": "known-item"}
```
Filenames MUST be the exact bare filenames from Step 1.

- [ ] **Step 5: Write `eval/README.md` with the grading rubric**

```markdown
# Evaluation gold set

`gold.jsonl` — one JSON object per line:
`{"query": str, "relevant": {bare_filename: grade}, "notes": str}`

## Grading rubric (document-level relevance)

- **3 — primary source:** the paper is centrally about the query topic; the
  paper you would cite first.
- **2 — substantive:** the topic is a real, developed section of the paper.
- **1 — peripheral:** the topic is mentioned or touched only in passing.
- **unlabeled (0):** not relevant — omit the paper from the query's `relevant` map.

Scoring is document-level: a paper is identified by its bare filename
(one paper per `source` in the collection). Metrics (via `ranx`):
`mrr`, `recall@5`, `recall@10`, `hit_rate@1`, `ndcg@10`.
```

- [ ] **Step 6: Verify the gold set loads**

Run: `source .venv/bin/activate && python -c "from sportsscience_rag.evaluator import load_gold; q, qr = load_gold('eval/gold.jsonl'); print(len(q), 'queries')"`
Expected: prints the query count (~30).

- [ ] **Step 7: Commit**

```bash
git add eval/gold.jsonl eval/README.md
git commit -m "feat: add graded evaluation gold set and grading rubric"
```

---

### Task 6: CLI refactor to subcommands (`ingest`/`search`/`eval`)

**Files:**
- Modify: `src/sportsscience_rag/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Retriever` (Task 2); `load_gold`, `build_run_dict`, `evaluate_run` (Tasks 3-4); existing ingestion wiring.
- Produces: `build_arg_parser()` returns a parser with subcommands `ingest`, `search`, `eval`. The `ingest` subcommand carries ALL the previous flags (`--prefix`, `--collection`, `--limit`, `--quarantine-report`, `--derived-prefix`, `--dry-run`, `-v`). `search` carries positional `query` (optional), `--limit` (default 10), `--collection`, `--json`, `-v`. `eval` carries `--gold` (default `eval/gold.jsonl`), `--limit` (default 10), `--collection`, `--report` (optional JSON path), `-v`.

> Breaking change: `python -m sportsscience_rag --prefix X` becomes `python -m sportsscience_rag ingest --prefix X`. The only callers are the user and `launch.json` (updated in Task 8).

- [ ] **Step 1: Update the CLI arg-parsing tests**

Replace `tests/test_cli.py` entirely with:

```python
"""Argument-parsing tests for the subcommand CLI surface.

These tests exercise only ``build_arg_parser()``; they never call ``main()``,
which would require real S3/Qdrant/model network access.
"""

import pytest

from sportsscience_rag.cli import build_arg_parser


def test_ingest_defaults():
    args = build_arg_parser().parse_args(["ingest"])
    assert args.command == "ingest"
    assert args.prefixes is None
    assert args.dry_run is False
    assert args.limit is None
    assert args.derived_prefix == "derived/"


def test_ingest_repeatable_prefix_and_flags():
    args = build_arg_parser().parse_args(
        ["ingest", "--prefix", "a/", "--prefix", "b/", "--collection", "c",
         "--limit", "2", "--dry-run", "-v", "--quarantine-report", "q.json"]
    )
    assert args.command == "ingest"
    assert args.prefixes == ["a/", "b/"]
    assert args.collection == "c"
    assert args.limit == 2
    assert args.dry_run is True
    assert args.verbose is True
    assert args.quarantine_report == "q.json"


def test_search_one_shot_args():
    args = build_arg_parser().parse_args(["search", "training load", "--limit", "5", "--json"])
    assert args.command == "search"
    assert args.query == "training load"
    assert args.limit == 5
    assert args.json is True


def test_search_repl_mode_has_no_query():
    args = build_arg_parser().parse_args(["search"])
    assert args.command == "search"
    assert args.query is None
    assert args.limit == 10


def test_eval_defaults():
    args = build_arg_parser().parse_args(["eval"])
    assert args.command == "eval"
    assert args.gold == "eval/gold.jsonl"
    assert args.limit == 10
    assert args.report is None


def test_no_subcommand_is_error():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_cli.py -v`
Expected: FAIL — the flat parser has no `command` attribute / rejects `ingest`.

- [ ] **Step 3: Refactor `cli.py` to subcommands**

Replace `build_arg_parser()` and `main()` in `src/sportsscience_rag/cli.py`. Keep `_DryStore` and `_write_quarantine` unchanged. New top-of-file imports add:

```python
import sys
from sportsscience_rag.retriever import Retriever
from sportsscience_rag.evaluator import build_run_dict, evaluate_run, load_gold
```

New `build_arg_parser()`:

```python
def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the subcommand CLI parser (``ingest`` / ``search`` / ``eval``).

    Returns:
        An ``argparse.ArgumentParser`` with a required subcommand stored in
        ``args.command``. ``ingest`` carries the full ingestion flag set;
        ``search`` carries an optional positional ``query`` plus ``--limit``,
        ``--collection``, ``--json``; ``eval`` carries ``--gold``, ``--limit``,
        ``--collection``, ``--report``. All subcommands accept ``-v/--verbose``.
    """
    parser = argparse.ArgumentParser(
        description="SportsScience RAG: ingest, search, and evaluate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Ingest S3 PDFs into Qdrant.")
    ing.add_argument("--prefix", action="append", dest="prefixes", metavar="FOLDER",
                     help="S3 prefix to ingest (repeatable). Defaults to bucket root.")
    ing.add_argument("--collection", help="Qdrant collection (overrides .env).")
    ing.add_argument("--limit", type=int, default=None, help="Process only first N PDFs.")
    ing.add_argument("--quarantine-report", dest="quarantine_report",
                     help="Write failed/skipped list to this JSON path.")
    ing.add_argument("--derived-prefix", dest="derived_prefix", default="derived/",
                     help="S3 key prefix for persisted page renders.")
    ing.add_argument("--dry-run", action="store_true", help="List PDFs, write nothing.")
    ing.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    srch = sub.add_parser("search", help="Retrieve chunks for a query (one-shot or REPL).")
    srch.add_argument("query", nargs="?", default=None,
                      help="Query string. Omit to enter an interactive REPL.")
    srch.add_argument("--limit", type=int, default=10, help="Number of chunks to return.")
    srch.add_argument("--collection", help="Qdrant collection (overrides .env).")
    srch.add_argument("--json", action="store_true", help="Emit raw JSON results.")
    srch.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    ev = sub.add_parser("eval", help="Score the retriever against the gold set.")
    ev.add_argument("--gold", default="eval/gold.jsonl", help="Path to gold JSONL.")
    ev.add_argument("--limit", type=int, default=10, help="Retrieval depth per query.")
    ev.add_argument("--collection", help="Qdrant collection (overrides .env).")
    ev.add_argument("--report", default=None, help="Write metrics + per-query JSON here.")
    ev.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")

    return parser
```

New `main()` dispatching on `args.command`:

```python
def main(argv: list[str] | None = None) -> int:
    """Runs the CLI, dispatching to the ingest, search, or eval subcommand.

    Args:
        argv: Arguments to parse (excluding program name); ``sys.argv`` if None.

    Returns:
        Process exit code (0 on success).
    """
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    config = IngestionConfig.from_env(env_path=PROJECT_ROOT / ".env")
    collection = args.collection or config.qdrant_collection

    if args.command == "ingest":
        return _run_ingest(args, config, collection)
    if args.command == "search":
        return _run_search(args, config, collection)
    if args.command == "eval":
        return _run_eval(args, config, collection)
    return 1  # pragma: no cover - argparse requires a subcommand


def _run_ingest(args: argparse.Namespace, config: IngestionConfig, collection: str) -> int:
    """Ingestion path (formerly the whole of ``main``)."""
    prefixes = args.prefixes or [""]
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region,
    )
    source = S3Source(s3_client, config.s3_bucket)

    if args.dry_run:
        pipeline = IngestionPipeline(
            config, source, parser=None, chunker=None, embedder=None,
            render_store=None, store=_DryStore(), logger=JsonlLogger(),
            parser_version=PARSER_VERSION,
        )
        result = pipeline.run(prefixes, limit=args.limit, dry_run=True)
        logger.info("Dry run: %d PDFs found.", len(result.outcomes))
        _write_quarantine(args, result)
        return 0

    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    pipeline = IngestionPipeline(
        config=config,
        source=source,
        parser=DoclingParser(image_dpi=config.image_dpi),
        chunker=SectionChunker(config),
        embedder=TextEmbedder(),
        render_store=RenderStore(s3_client, config.s3_bucket, args.derived_prefix),
        text_store=TextStore(s3_client, config.s3_bucket, args.derived_prefix),
        store=QdrantStore(qdrant_client, collection),
        logger=JsonlLogger(),
        parser_version=PARSER_VERSION,
    )
    result = pipeline.run(prefixes, limit=args.limit)
    done = sum(1 for o in result.outcomes if o.status == "done")
    logger.info("Ingestion finished: %d done, %d quarantined, %d skipped.",
                done, len(result.quarantine),
                sum(1 for o in result.outcomes if o.status == "skipped"))
    _write_quarantine(args, result)
    return 0


def _make_retriever(config: IngestionConfig, collection: str) -> Retriever:
    """Build a Retriever wired to the live Qdrant collection and MiniLM embedder."""
    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    return Retriever(qdrant_client, TextEmbedder(), collection)


def _format_hit(hit) -> str:
    """One-line human-readable rendering of a RetrievedChunk."""
    filename = hit.source.rsplit("/", 1)[-1]
    pages = ",".join(str(p) for p in hit.page_numbers) or "-"
    snippet = " ".join(hit.text.split())[:200]
    return (f"#{hit.rank:<2} {hit.score:.4f}  {filename}  p.{pages}  "
            f"§ {hit.section_path or '-'}\n     {snippet}")


def _run_search(args: argparse.Namespace, config: IngestionConfig, collection: str) -> int:
    """Search path: one-shot when a query is given, REPL otherwise."""
    retriever = _make_retriever(config, collection)

    def _run_one(query: str) -> None:
        hits = retriever.search(query, limit=args.limit)
        if args.json:
            print(json.dumps([dataclasses.asdict(h) for h in hits], indent=2))
        elif not hits:
            print("(no results)")
        else:
            for hit in hits:
                print(_format_hit(hit))

    if args.query is not None:
        _run_one(args.query)
        return 0

    print("Interactive search. Type a query and press Enter; Ctrl-D to exit.")
    for line in sys.stdin:
        query = line.strip()
        if query:
            _run_one(query)
    return 0


def _run_eval(args: argparse.Namespace, config: IngestionConfig, collection: str) -> int:
    """Evaluation path: retrieve for each gold query, score with ranx, report."""
    queries, qrels = load_gold(args.gold)
    retriever = _make_retriever(config, collection)
    hits_by_query = {g.query: retriever.search(g.query, limit=args.limit) for g in queries}
    run_dict = build_run_dict(hits_by_query)
    report = evaluate_run(qrels, run_dict, queries)

    print(f"\nEvaluated {report.n_queries} queries (limit={args.limit}) on '{collection}':\n")
    for name, score in report.metrics.items():
        print(f"  {name:12s} {score:.4f}")
    print("\nPer-query (best relevant-paper rank; None = miss):")
    for row in report.per_query:
        rank = row["best_rank"] if row["best_rank"] is not None else "MISS"
        print(f"  [{str(rank):>4}] {row['query']}")

    if args.report:
        payload = {"collection": collection, "limit": args.limit,
                   "metrics": report.metrics, "per_query": report.per_query,
                   "n_queries": report.n_queries}
        Path(args.report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote eval report to %s", args.report)
    return 0
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_cli.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full unit suite to confirm no regressions**

Run: `source .venv/bin/activate && pytest -m "not integration" -q 2>/dev/null`
Expected: PASS (all prior tests + new retriever/evaluator/cli tests).

- [ ] **Step 6: Commit**

```bash
git add src/sportsscience_rag/cli.py tests/test_cli.py
git commit -m "feat: refactor CLI to ingest/search/eval subcommands"
```

---

### Task 7: Integration smoke test (live collection + real embedder)

**Files:**
- Create: `tests/test_eval_integration.py`

**Interfaces:**
- Consumes: `_make_retriever` path via real `TextEmbedder` + live `QdrantClient`; `build_run_dict`, `evaluate_run`, `load_gold`.
- Produces: an `@pytest.mark.integration` test asserting the end-to-end pipeline produces a sane baseline (MRR above a floor), catching gross wiring errors.

> This test requires `.env` credentials and network. It is deselected by the default `pytest -m "not integration"` run.

- [ ] **Step 1: Write the integration test**

Create `tests/test_eval_integration.py`:

```python
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
```

- [ ] **Step 2: Run the integration test**

Run: `source .venv/bin/activate && HF_HUB_OFFLINE=1 pytest tests/test_eval_integration.py -m integration -v -s 2>/dev/null`
Expected: PASS. If MRR is below the floor, that is a real signal — inspect the per-query misses via `python -m sportsscience_rag eval` before adjusting the floor. Do NOT lower the floor to force a pass without understanding why.

- [ ] **Step 3: Commit**

```bash
git add tests/test_eval_integration.py
git commit -m "test: add end-to-end retrieval+eval integration smoke test"
```

---

### Task 8: Update `launch.json` debug configs

**Files:**
- Modify: `.vscode/launch.json`

**Interfaces:**
- Consumes: the new subcommand CLI (Task 6).
- Produces: an updated ingest debug config (subcommand form) and a new search debug config.

- [ ] **Step 1: Read the current launch.json**

Run: `cat .vscode/launch.json`
Locate the ingest configs whose `args` start with `--prefix` / `--collection` (flat form).

- [ ] **Step 2: Prefix ingest configs with the `ingest` subcommand**

For each ingest-oriented config, insert `"ingest"` as the FIRST element of its `args` array. For example, a config with `"args": ["--prefix", "disalvo2007", "--collection", "debug-stepthrough"]` becomes `"args": ["ingest", "--prefix", "disalvo2007", "--collection", "debug-stepthrough"]`. Leave `"module": "sportsscience_rag"`, `envFile`, and `justMyCode` unchanged.

- [ ] **Step 3: Add a search debug config**

Add this configuration object to the `configurations` array:

```json
{
  "name": "Search: one-shot query",
  "type": "debugpy",
  "request": "launch",
  "module": "sportsscience_rag",
  "args": ["search", "training load and injury risk", "--limit", "5"],
  "console": "integratedTerminal",
  "envFile": "${workspaceFolder}/.env",
  "justMyCode": true
}
```

- [ ] **Step 4: Validate JSON**

Run: `python -c "import json,pathlib; json.loads(pathlib.Path('.vscode/launch.json').read_text()); print('valid')"`
Expected: prints `valid` (launch.json permits `//` comments in VS Code, but if the file has none this check passes; if it has comments, skip this step and eyeball the diff instead).

- [ ] **Step 5: Commit**

```bash
git add .vscode/launch.json
git commit -m "chore: update debug configs for ingest subcommand + add search"
```

---

### Task 9: Update README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing docs for `search`/`eval` and the eval harness.

- [ ] **Step 1: Update the Run section for subcommands**

In `README.md`, under `## Run`, change the invocation examples to the subcommand form and add search/eval examples:

```bash
# Ingest (formerly the bare invocation)
python -m sportsscience_rag ingest --prefix papers/ --limit 2 --collection my-test

# Search — one-shot
python -m sportsscience_rag search "high-intensity running in elite soccer" --limit 5

# Search — interactive REPL (model loaded once)
python -m sportsscience_rag search

# Evaluate retrieval against the graded gold set
python -m sportsscience_rag eval --gold eval/gold.jsonl --limit 10 --report eval/results.json
```

- [ ] **Step 2: Add a Retrieval & Evaluation section**

After the `## Run` section, add:

```markdown
## Retrieval & Evaluation

Retrieval is chunk-level: `Retriever.search(query, limit)` embeds the query with
the same FastEmbed MiniLM model used at ingestion and runs a cosine
`query_points` against the `text_embedding` named vector, returning ranked
`RetrievedChunk`s (score, source, page numbers, section path, text).

Evaluation is document-level. `eval/gold.jsonl` holds ~30 graded queries
(`{"query", "relevant": {filename: grade}, "notes"}`, grades 1-3 per the rubric
in `eval/README.md`). The `eval` subcommand retrieves each query, dedups chunks
to the best-scoring chunk per paper, and scores the run with
[`ranx`](https://github.com/AmenRa/ranx): `mrr`, `recall@5`, `recall@10`,
`hit_rate@1`, `ndcg@10` (macro-averaged), plus a per-query best-rank breakdown.
Install the eval extra with `uv pip install -e ".[eval]"`.
```

- [ ] **Step 3: Add retriever/evaluator to the Classes table**

In the `## Classes` table, add two rows:

```markdown
| `retriever.py` | `Retriever` · `RetrievedChunk` | Chunk-level semantic search: embed query → `query_points` → ranked `RetrievedChunk`s |
| `evaluator.py` | `load_gold` · `build_run_dict` · `evaluate_run` · `GoldQuery` · `EvalReport` | Graded document-level IR evaluation via ranx (paper-dedup by max chunk score) |
```

- [ ] **Step 4: Verify markdown renders (sanity read)**

Run: `sed -n '84,120p' README.md`
Expected: the updated Run + Retrieval sections appear as written.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document search/eval subcommands and evaluation harness"
```

---

## Self-Review

**Spec coverage** (against the 11 grilled decisions):
1. Document-level scoring → Task 3/4 (paper-dedup in `build_run_dict`, per-query rank). ✓
2. Paper identity = `source`, duplicate removed → done pre-plan; enforced by bare-filename dedup. ✓
3. `eval/gold.jsonl`, committed, filename-keyed → Task 5. ✓
4. ~30 curated LLM-seeded + verified queries → Task 5 (Steps 2-3 human checkpoint). ✓
5. `Retriever` (chunk-level) + evaluator dedups → Tasks 2, 3. ✓
6. Metrics MRR/Recall@5/@10/Hit@1 (+ndcg@10) macro-avg + per-query → Task 4. ✓
7. Subcommands, no numpy/pandas in our code, JSON+table → Tasks 1, 6. ✓
8. `search` one-shot/REPL, rich snippet, `--json`/`--limit` → Task 6. ✓
9. TDD evaluator math, mock-unit retriever, one integration smoke → Tasks 2-4 (TDD), 7. ✓
10. Deferred: rerankers, hybrid, generation, filtering, visual, true-MAP → not in any task (intentional). ✓
11. Graded qrels (3/2/1 rubric), `{filename: grade}` gold shape → Tasks 3, 5. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; test steps show full assertions. ✓

**Type consistency:** `RetrievedChunk` fields (`rank/score/source/chunk_index/page_numbers/section_path/text`) identical across Tasks 2, 3, 6. `build_run_dict` returns `{query: {filename: float}}` consumed identically by Task 4/6. `evaluate_run(qrels_dict, run_dict, queries)` signature identical in Tasks 4, 6, 7. `GoldQuery(query, relevant, notes)` and `load_gold -> (list[GoldQuery], qrels_dict)` consistent across Tasks 3-7. `METRICS` list consistent Task 3 → 4. ✓
