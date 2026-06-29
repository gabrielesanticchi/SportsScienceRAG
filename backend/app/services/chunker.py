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
