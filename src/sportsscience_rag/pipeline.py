"""Document-level ingestion orchestration with quarantine and resumability."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.hashing import content_hash
from sportsscience_rag.models import IngestOutcome, QuarantineEntry


@dataclass(frozen=True)
class IngestionResult:
    """Aggregate result of a full ingestion run.

    Attributes:
        outcomes: Per-document outcomes, one per PDF processed (or listed,
            in the case of a dry run), in source-listing order.
        quarantine: Entries for documents that failed and were quarantined,
            in the order the failures occurred.
    """
    outcomes: list[IngestOutcome]
    quarantine: list[QuarantineEntry]


class IngestionPipeline:
    """Runs the fetch -> hash -> skip -> parse -> render -> chunk -> embed -> upsert flow.

    Orchestrates the collaborators produced by earlier tasks (S3 source,
    parser, chunker, embedder, render store, vector store, and JSONL logger)
    into a single per-document pipeline. Each document is processed
    independently: a failure at any stage quarantines that document and
    ingestion continues with the next one, so a single corrupt or malformed
    PDF never aborts the batch. Documents whose content hash and parser/chunk
    configuration have already been ingested are skipped, making re-runs of
    the same prefixes resumable and idempotent.
    """

    def __init__(
        self,
        config: IngestionConfig,
        source,
        parser,
        chunker,
        embedder,
        render_store,
        store,
        logger,
        parser_version: str,
        text_store=None,
    ) -> None:
        """Initializes the pipeline with its collaborators.

        Args:
            config: Ingestion configuration, used for chunk-config hashing.
            source: Object exposing ``list_pdfs(prefixes, limit)`` and
                ``fetch(key)`` (e.g. ``S3Source``).
            parser: Object exposing ``parse(data, key) -> ParsedDocument``
                (e.g. ``DoclingParser``).
            chunker: Object exposing
                ``chunk(markdown, page_texts) -> list[Chunk]``
                (e.g. ``SectionChunker``).
            embedder: Object exposing ``embed(texts) -> list[list[float]]``
                (e.g. ``TextEmbedder``).
            render_store: Object exposing
                ``upload(content_hash, renders)`` (e.g. ``RenderStore``).
            text_store: Optional object exposing
                ``upload(content_hash, markdown, page_texts)`` (e.g.
                ``TextStore``) that persists the parsed markdown and per-page
                text to S3 for inspection. If ``None``, no text artifacts are
                written.
            store: Vector store exposing ``ensure_collection()``,
                ``already_ingested(content_hash, parser_version,
                chunk_config_hash) -> bool``, and ``upsert(...) -> int``
                (e.g. ``QdrantStore``).
            logger: Object exposing ``event(**fields)`` for structured JSONL
                logging (e.g. ``JsonlLogger``).
            parser_version: Version tag of the parser, recorded alongside
                each ingested document for skip-detection and provenance.
        """
        self._config = config
        self._source = source
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._render_store = render_store
        self._text_store = text_store
        self._store = store
        self._logger = logger
        self._parser_version = parser_version

    def run(
        self,
        prefixes: list[str],
        limit: int | None = None,
        dry_run: bool = False,
    ) -> IngestionResult:
        """Ingests all PDFs found under the given S3 prefixes.

        Lists matching PDFs via the configured source, then processes each
        one through the fetch -> hash -> skip-check -> parse -> render-upload
        -> chunk -> embed -> upsert flow. A failure at any stage for a given
        document results in a quarantine entry for that document only;
        processing continues with the remaining documents.

        Args:
            prefixes: S3 key prefixes to search for PDFs.
            limit: Maximum total number of PDFs to process across all
                prefixes combined. If ``None``, all matching PDFs are
                processed.
            dry_run: If ``True``, only lists matching PDFs and reports them
                as "skipped" without fetching, parsing, or writing anything
                (no render uploads, no vector upserts, no collection
                creation).

        Returns:
            An ``IngestionResult`` containing the per-document outcomes and
            any quarantine entries recorded during the run.
        """
        outcomes: list[IngestOutcome] = []
        quarantine: list[QuarantineEntry] = []
        pdfs = self._source.list_pdfs(prefixes, limit=limit)

        if not dry_run:
            self._store.ensure_collection()

        for key, source_url in pdfs:
            if dry_run:
                outcomes.append(IngestOutcome(source=source_url, content_hash="", status="skipped", n_chunks=0))
                self._logger.event(
                    source=source_url, content_hash="", stage="dry-run",
                    duration_ms=0, n_chunks=0, status="skipped", error_class=None,
                )
                continue
            outcome, entry = self._process_one(key, source_url)
            outcomes.append(outcome)
            if entry is not None:
                quarantine.append(entry)

        return IngestionResult(outcomes=outcomes, quarantine=quarantine)

    def _process_one(self, key: str, source_url: str) -> tuple[IngestOutcome, QuarantineEntry | None]:
        """Runs the full single-document ingestion flow.

        Fetches the raw PDF bytes, computes its content hash, and short
        circuits if that hash (combined with the parser version and chunk
        config hash) has already been ingested. Otherwise parses, uploads
        page renders, chunks, embeds, and upserts into the vector store.
        Any exception raised at any stage is caught here so that the
        document is quarantined with the stage at which it failed, rather
        than propagating and aborting the rest of the batch.

        Args:
            key: S3 object key of the PDF to process.
            source_url: Fully qualified ``s3://bucket/key`` URL, used as the
                stable identifier in outcomes, quarantine entries, and logs.

        Returns:
            A ``(outcome, quarantine_entry)`` tuple. ``quarantine_entry`` is
            ``None`` unless the document failed or was otherwise flagged for
            manual review (empty text layer, no chunks produced, or an
            unhandled exception).
        """
        start = time.monotonic()
        chash = ""
        stage = "fetch"
        try:
            data = self._source.fetch(key)
            chash = content_hash(data)

            stage = "skip-check"
            if self._store.already_ingested(chash, self._parser_version, self._config.chunk_config_hash):
                return self._done(source_url, chash, "skipped", 0, start, "skip-check"), None

            stage = "parse"
            parsed = self._parser.parse(data, key)
            if parsed.is_empty:
                return (
                    self._done(source_url, chash, "quarantined", 0, start, "parse"),
                    QuarantineEntry(source_url, chash, "parse", "EmptyText", "empty text layer"),
                )

            stage = "render-upload"
            self._render_store.upload(chash, parsed.renders)

            if self._text_store is not None:
                stage = "text-upload"
                self._text_store.upload(chash, parsed.markdown, parsed.page_texts)

            stage = "chunk"
            chunks = self._chunker.chunk(parsed.markdown, parsed.page_texts)
            if not chunks:
                return (
                    self._done(source_url, chash, "quarantined", 0, start, "chunk"),
                    QuarantineEntry(source_url, chash, "chunk", "NoChunks", "no chunks produced"),
                )

            stage = "embed"
            vectors = self._embedder.embed([c.text for c in chunks])

            stage = "upsert"
            n = self._store.upsert(
                chash, source_url, self._parser_version,
                self._config.chunk_config_hash, chunks, vectors,
            )
            return self._done(source_url, chash, "done", n, start, "upsert"), None
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            entry = QuarantineEntry(source_url, chash, stage, type(exc).__name__, str(exc))
            return (
                self._done(source_url, chash, "quarantined", 0, start, stage, type(exc).__name__),
                entry,
            )

    def _done(
        self,
        source: str,
        chash: str,
        status: str,
        n_chunks: int,
        start: float,
        stage: str,
        error_class: str | None = None,
    ) -> IngestOutcome:
        """Emits a JSONL log event for a document and builds its outcome.

        Args:
            source: Fully qualified source URL of the document.
            chash: Content hash of the document (empty string if fetch
                failed before hashing).
            status: Final status for this document: "done", "quarantined",
                or "skipped".
            n_chunks: Number of chunks upserted (0 unless status is "done").
            start: ``time.monotonic()`` timestamp captured at the start of
                processing, used to compute elapsed duration.
            stage: Pipeline stage this outcome corresponds to (e.g.,
                "skip-check", "parse", "chunk", "upsert", or the stage at
                which an exception was raised).
            error_class: Exception class name if this outcome resulted from
                an error, otherwise ``None``.

        Returns:
            The ``IngestOutcome`` for this document.
        """
        self._logger.event(
            source=source, content_hash=chash, stage=stage,
            duration_ms=int((time.monotonic() - start) * 1000),
            n_chunks=n_chunks, status=status, error_class=error_class,
        )
        return IngestOutcome(source=source, content_hash=chash, status=status, n_chunks=n_chunks)
