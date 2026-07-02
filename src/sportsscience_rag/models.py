"""Immutable data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image


@dataclass(frozen=True)
class RawPdf:
    """Immutable representation of a raw PDF file.

    Attributes:
        source: Source identifier for the PDF.
        key: Unique key for the PDF.
        data: Raw bytes of the PDF file.
        content_hash: Hash of the PDF content for deduplication.
    """
    source: str
    key: str
    data: bytes
    content_hash: str


@dataclass(frozen=True, eq=False)
class PageRender:
    """Rendered page image from a PDF.

    Attributes:
        page_number: Zero-indexed page number in the document.
        image: PIL Image object of the rendered page.
    """
    page_number: int
    image: Image


@dataclass(frozen=True)
class Chunk:
    """Semantic text chunk extracted from a parsed document.

    Attributes:
        index: Sequence number of the chunk within the document.
        text: Chunk content text.
        section_path: Hierarchical section path (e.g., "# Section > ## Subsection").
        page_numbers: Tuple of zero-indexed page numbers where this chunk appears.
    """
    index: int
    text: str
    section_path: str
    page_numbers: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class ParsedDocument:
    """Fully parsed and rendered PDF document.

    Attributes:
        markdown: Markdown representation of document content.
        page_count: Total number of pages in the document.
        renders: Tuple of rendered page images.
        is_empty: True if the document's text layer is empty or near-empty,
            indicating it should be quarantined for manual review.
        page_texts: Tuple of (page_no, normalized_text) pairs for best-effort
            page-to-content mapping across all pages.
    """
    markdown: str
    page_count: int
    renders: tuple[PageRender, ...]
    is_empty: bool
    page_texts: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class QuarantineEntry:
    """Record of a document quarantined during ingestion.

    Attributes:
        source: Source identifier of the quarantined document.
        content_hash: Hash of the document content.
        stage: Pipeline stage where the failure occurred (e.g., "parse", "chunk").
        error_class: Exception class name of the error.
        message: Human-readable error message.
    """
    source: str
    content_hash: str
    stage: str
    error_class: str
    message: str


@dataclass(frozen=True)
class IngestOutcome:
    """Final outcome of a single document ingestion.

    Attributes:
        source: Source identifier of the ingested document.
        content_hash: Hash of the document content.
        status: Final ingestion status, one of "done" (successfully ingested),
            "quarantined" (failed and flagged for review), or "skipped" (already
            processed or excluded).
        n_chunks: Number of semantic chunks extracted from the document.
    """
    source: str
    content_hash: str
    status: str
    n_chunks: int
