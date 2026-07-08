"""Command-line entry point for the ingestion pipeline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
from qdrant_client import QdrantClient

from sportsscience_rag.chunker import SectionChunker
from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.embedder import TextEmbedder
from sportsscience_rag.evaluator import build_run_dict, evaluate_run, load_gold
from sportsscience_rag.logging_setup import JsonlLogger
from sportsscience_rag.parser import PARSER_VERSION, DoclingParser
from sportsscience_rag.persistence import RenderStore, TextStore
from sportsscience_rag.pipeline import IngestionPipeline
from sportsscience_rag.qdrant_store import QdrantStore
from sportsscience_rag.retriever import Retriever, RetrievedChunk
from sportsscience_rag.s3_source import S3Source

if TYPE_CHECKING:
    from sportsscience_rag.pipeline import IngestionResult

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    """Runs the ingestion path (formerly the whole of ``main``).

    Args:
        args: Parsed ``ingest`` subcommand arguments.
        config: Ingestion configuration loaded from the environment.
        collection: Resolved Qdrant collection name.

    Returns:
        Process exit code (always ``0``; failures are quarantined).
    """
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
                done,
                len(result.quarantine),
                sum(1 for o in result.outcomes if o.status == "skipped"))
    _write_quarantine(args, result)
    return 0


def _make_retriever(config: IngestionConfig, collection: str) -> Retriever:
    """Builds a Retriever wired to the live Qdrant collection and MiniLM embedder.

    Args:
        config: Ingestion configuration providing Qdrant connection details.
        collection: Target Qdrant collection name.

    Returns:
        A ``Retriever`` instance ready to run queries.
    """
    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    return Retriever(qdrant_client, TextEmbedder(), collection)


def _format_hit(hit: RetrievedChunk) -> str:
    """Renders a single ``RetrievedChunk`` as a one-line human-readable string.

    Args:
        hit: The retrieved chunk to format.

    Returns:
        A two-line formatted string: a summary header plus a text snippet.
    """
    filename = hit.source.rsplit("/", 1)[-1]
    pages = ",".join(str(p) for p in hit.page_numbers) or "-"
    snippet = " ".join(hit.text.split())[:200]
    return (f"#{hit.rank:<2} {hit.score:.4f}  {filename}  p.{pages}  "
            f"§ {hit.section_path or '-'}\n     {snippet}")


def _run_search(args: argparse.Namespace, config: IngestionConfig, collection: str) -> int:
    """Runs the search path: one-shot when a query is given, REPL otherwise.

    Args:
        args: Parsed ``search`` subcommand arguments.
        config: Ingestion configuration providing Qdrant connection details.
        collection: Resolved Qdrant collection name.

    Returns:
        Process exit code (always ``0``).
    """
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
    """Runs the evaluation path: retrieve for each gold query, score with ranx, report.

    Args:
        args: Parsed ``eval`` subcommand arguments.
        config: Ingestion configuration providing Qdrant connection details.
        collection: Resolved Qdrant collection name.

    Returns:
        Process exit code (always ``0``).
    """
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


class _DryStore:
    """No-op vector store so dry-run needs no Qdrant connection.

    Satisfies the ``store`` collaborator interface expected by
    ``IngestionPipeline`` for the ``ensure_collection()`` call, which is
    skipped entirely when ``dry_run=True`` but kept here for interface
    completeness and future-proofing.
    """

    def ensure_collection(self) -> None:  # pragma: no cover - trivial
        """No-op collection setup; dry runs never touch Qdrant."""
        pass


def _write_quarantine(args: argparse.Namespace, result: "IngestionResult") -> None:
    """Writes the quarantine report to disk if requested.

    Args:
        args: Parsed CLI arguments; only ``quarantine_report`` is used.
        result: The ``IngestionResult`` returned by ``IngestionPipeline.run``,
            whose ``quarantine`` entries are serialized to JSON.
    """
    if not args.quarantine_report:
        return
    payload = [dataclasses.asdict(entry) for entry in result.quarantine]
    Path(args.quarantine_report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote quarantine report (%d entries) to %s",
                len(payload), args.quarantine_report)
