"""Deterministic chunking utilities."""

import hashlib
import re
import uuid
from dataclasses import dataclass

from app.config import Settings, get_settings


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    page_number: int | None = None
    section: str | None = None


def deterministic_point_id(document_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point ID derived from hash(doc_id + ':' + chunk_index)."""
    payload = f"{document_id}:{chunk_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def _split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_document_text(
    text: str,
    *,
    page_count: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    settings: Settings | None = None,
) -> list[TextChunk]:
    """Chunk text while preserving page numbers when page breaks are available."""
    if "\f" in text:
        pages = text.split("\f")
    elif page_count and page_count > 1:
        flat_chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            settings=settings,
        )
        return _assign_pages_evenly(flat_chunks, page_count)
    else:
        pages = [text]

    chunks: list[TextChunk] = []
    chunk_index = 0
    for page_number, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        for chunk in chunk_text(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            settings=settings,
        ):
            chunks.append(
                TextChunk(
                    index=chunk_index,
                    text=chunk.text,
                    page_number=page_number,
                )
            )
            chunk_index += 1
    return chunks


def _assign_pages_evenly(chunks: list[TextChunk], page_count: int) -> list[TextChunk]:
    if not chunks:
        return []

    per_page = max(1, -(-len(chunks) // page_count))
    return [
        TextChunk(
            index=chunk.index,
            text=chunk.text,
            page_number=min(page_count, (index // per_page) + 1),
            section=chunk.section,
        )
        for index, chunk in enumerate(chunks)
    ]


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    settings: Settings | None = None,
) -> list[TextChunk]:
    cfg = settings or get_settings()
    size = chunk_size or cfg.chunk_size
    overlap = chunk_overlap or cfg.chunk_overlap

    words = _split_words(text)
    if not words:
        return []

    stride = max(size - overlap, 1)
    chunks: list[TextChunk] = []
    for index, start in enumerate(range(0, len(words), stride)):
        window = words[start : start + size]
        if not window:
            continue
        chunks.append(TextChunk(index=index, text=" ".join(window)))
    return chunks
