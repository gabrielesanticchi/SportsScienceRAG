"""Immutable data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image


@dataclass(frozen=True)
class RawPdf:
    source: str
    key: str
    data: bytes
    content_hash: str


@dataclass(frozen=True, eq=False)
class PageRender:
    page_number: int
    image: Image


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section_path: str
    page_numbers: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class ParsedDocument:
    markdown: str
    page_count: int
    renders: tuple[PageRender, ...]
    is_empty: bool
    page_texts: tuple[tuple[int, str], ...]  # (page_no, normalized text) for page mapping


@dataclass(frozen=True)
class QuarantineEntry:
    source: str
    content_hash: str
    stage: str
    error_class: str
    message: str


@dataclass(frozen=True)
class IngestOutcome:
    source: str
    content_hash: str
    status: str  # "done" | "quarantined" | "skipped"
    n_chunks: int
