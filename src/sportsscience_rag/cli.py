"""Command-line entry point for the ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import boto3
from qdrant_client import QdrantClient

from sportsscience_rag.chunker import SectionChunker
from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.embedder import TextEmbedder
from sportsscience_rag.logging_setup import JsonlLogger
from sportsscience_rag.parser import PARSER_VERSION, DoclingParser
from sportsscience_rag.persistence import RenderStore
from sportsscience_rag.pipeline import IngestionPipeline
from sportsscience_rag.qdrant_store import QdrantStore
from sportsscience_rag.s3_source import S3Source

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for the ingestion CLI.

    Returns:
        An ``argparse.ArgumentParser`` configured with the ingestion CLI's
        flags: repeatable ``--prefix`` (collected into ``prefixes``),
        ``--collection``, ``--limit``, ``--quarantine-report``,
        ``--derived-prefix`` (default ``"derived/"``), ``--dry-run``, and
        ``-v``/``--verbose``. There is no ``--multimodal`` flag.
    """
    parser = argparse.ArgumentParser(
        description="Ingest S3 PDFs into Qdrant via Docling + FastEmbed.",
    )
    parser.add_argument("--prefix", action="append", dest="prefixes", metavar="FOLDER",
                        help="S3 prefix to ingest (repeatable). Defaults to bucket root.")
    parser.add_argument("--collection", help="Qdrant collection (overrides .env).")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N PDFs.")
    parser.add_argument("--quarantine-report", dest="quarantine_report",
                        help="Write failed/skipped list to this JSON path.")
    parser.add_argument("--derived-prefix", dest="derived_prefix", default="derived/",
                        help="S3 key prefix for persisted page renders.")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs, write nothing.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the ingestion CLI.

    Parses arguments, loads configuration from the environment (and a
    project-root ``.env`` file if present), and wires together the S3
    source with either a no-op dry-run store or the full Docling/FastEmbed/
    Qdrant pipeline. Optionally writes a JSON quarantine report of
    failed/skipped documents.

    Args:
        argv: Command-line arguments to parse, excluding the program name.
            If ``None``, arguments are taken from ``sys.argv``.

    Returns:
        Process exit code. Always ``0`` on a completed run (individual
        document failures are quarantined, not raised).
    """
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    config = IngestionConfig.from_env(env_path=PROJECT_ROOT / ".env")
    collection = args.collection or config.qdrant_collection
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


def _write_quarantine(args: argparse.Namespace, result) -> None:
    """Writes the quarantine report to disk if requested.

    Args:
        args: Parsed CLI arguments; only ``quarantine_report`` is used.
        result: The ``IngestionResult`` returned by ``IngestionPipeline.run``,
            whose ``quarantine`` entries are serialized to JSON.
    """
    if not args.quarantine_report:
        return
    payload = [vars(entry) for entry in result.quarantine]
    Path(args.quarantine_report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote quarantine report (%d entries) to %s",
                len(payload), args.quarantine_report)
