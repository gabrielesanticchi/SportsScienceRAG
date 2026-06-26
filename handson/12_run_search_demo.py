#!/usr/bin/env python3
"""Runnable CLI for the sports-science semantic search engine.

This is the preliminary, directly-testable entry point. It wires the
``sports_science_search`` package to a real Qdrant backend and exposes:

  * ingest   — extract -> chunk (3 strategies) -> vectorize -> upload all PDFs
  * search   — query all three chunking strategies and compare results
  * stats    — print collection statistics
  * analyze  — print chunking-effectiveness analysis
  * demo     — self-contained in-memory run (ingest + one search, no cloud)

By default the engine connects to Qdrant Cloud using QDRANT_URL and
QDRANT_API_KEY read from ``.env``. Pass ``--local`` to use an ephemeral
in-memory Qdrant instead. Note: in-memory data does NOT persist across
invocations, so for a local run use the ``demo`` subcommand (ingest + search
in one process).

Examples
--------
    source .venv/bin/activate

    # One-time: load every paper into Qdrant Cloud (recreate the collection)
    python examples/run_search_demo.py ingest --recreate

    # Then query it repeatedly (cloud data persists between runs)
    python examples/run_search_demo.py search "high intensity interval training"
    python examples/run_search_demo.py search "GPS tracking accuracy" --limit 5 --section Methods

    # Inspect what is stored
    python examples/run_search_demo.py stats
    python examples/run_search_demo.py analyze

    # Fully self-contained run on a local in-memory DB (no cloud needed)
    python examples/run_search_demo.py demo --query "muscle fatigue"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Make the package importable when running this file directly from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sports_science_search import COLLECTION_NAME, SemanticSearchEngine  # noqa: E402

ASSETS_DIR = PROJECT_ROOT / "assets"


def build_client(use_local: bool) -> QdrantClient:
    """Create a Qdrant client, either in-memory or against Qdrant Cloud.

    Args:
        use_local: If True, use an ephemeral in-memory instance. Otherwise
            connect to Qdrant Cloud using QDRANT_URL / QDRANT_API_KEY from .env.

    Returns:
        An initialized QdrantClient.

    Raises:
        SystemExit: If cloud credentials are requested but missing.
    """
    if use_local:
        print("[client] Using in-memory Qdrant (data is NOT persisted).")
        return QdrantClient(":memory:")

    load_dotenv(PROJECT_ROOT / ".env")
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    if not url or not api_key:
        sys.exit(
            "[client] Missing QDRANT_URL / QDRANT_API_KEY in .env. "
            "Add them or pass --local for an in-memory run."
        )
    print(f"[client] Connecting to Qdrant Cloud at {url}")
    return QdrantClient(url=url, api_key=api_key)


def build_engine(client: QdrantClient) -> SemanticSearchEngine:
    """Initialize the search engine (loads the embedding model, ~200MB once).

    Uses lenient section detection so the entire corpus ingests even when a
    paper's sections cannot be auto-detected (it falls back to a single "Body"
    section). Pass strict_sections=True for the original fail-fast behavior.
    """
    print("[engine] Loading embedding model (one-time, may take a moment)...")
    return SemanticSearchEngine(
        pdf_dir=ASSETS_DIR,
        qdrant_client=client,
        collection_name=COLLECTION_NAME,
        strict_sections=False,
    )


def _print_stats(stats: dict) -> None:
    """Pretty-print collection statistics."""
    print(f"  Total points:  {stats['total_points']}")
    print(f"  Unique papers: {stats['unique_papers']}")
    print(f"  By strategy:   {stats['chunks_by_strategy']}")
    print(f"  By section:    {stats['chunks_by_section']}")


def cmd_ingest(args: argparse.Namespace) -> None:
    """Process all PDFs and upload them to Qdrant."""
    engine = build_engine(build_client(args.local))
    print(f"[ingest] Processing PDFs from {ASSETS_DIR} ...")
    engine.process_papers(recreate_collection=args.recreate)
    print("\n[ingest] Done. Collection stats:")
    _print_stats(engine.vector_store.get_collection_stats())


def cmd_search(args: argparse.Namespace) -> None:
    """Run a query across all three chunking strategies and compare."""
    engine = build_engine(build_client(args.local))
    results = engine.search_and_compare(
        query=args.query,
        limit=args.limit,
        section_filter=args.section,
        year_filter=args.year,
        verbose=True,
    )
    if args.export:
        out = Path(args.export)
        engine.export_results(results, out)
        print(f"\n[search] Results exported to {out}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print collection statistics."""
    engine = build_engine(build_client(args.local))
    _print_stats(engine.vector_store.get_collection_stats())


def cmd_analyze(args: argparse.Namespace) -> None:
    """Print chunking-effectiveness analysis."""
    engine = build_engine(build_client(args.local))
    engine.analyze_chunking_effectiveness()


def cmd_demo(args: argparse.Namespace) -> None:
    """Self-contained run on a local in-memory DB: ingest then one search."""
    print("[demo] In-memory run: ingest + search in a single process.")
    engine = build_engine(QdrantClient(":memory:"))
    engine.process_papers(recreate_collection=True)
    engine.search_and_compare(query=args.query, limit=args.limit, verbose=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--local", action="store_true", help="Use in-memory Qdrant instead of cloud"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Process and upload all PDFs")
    p_ingest.add_argument(
        "--recreate", action="store_true", help="Recreate the collection from scratch"
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Search all strategies and compare")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument(
        "--limit", type=int, default=3, help="Results per strategy (default: 3)"
    )
    p_search.add_argument("--section", default=None, help="Filter by section (e.g. Methods)")
    p_search.add_argument("--year", type=int, default=None, help="Filter by publication year")
    p_search.add_argument("--export", default=None, help="Export results to this JSON path")
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="Print collection statistics")
    p_stats.set_defaults(func=cmd_stats)

    p_analyze = sub.add_parser("analyze", help="Print chunking-effectiveness analysis")
    p_analyze.set_defaults(func=cmd_analyze)

    p_demo = sub.add_parser("demo", help="Self-contained in-memory ingest + search")
    p_demo.add_argument(
        "--query",
        default="high intensity interval training",
        help="Query for the demo run",
    )
    p_demo.add_argument(
        "--limit", type=int, default=3, help="Results per strategy (default: 3)"
    )
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
