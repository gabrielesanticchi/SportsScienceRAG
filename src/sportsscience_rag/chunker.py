"""Heading-aware markdown chunking with a token-budget size guard."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.models import Chunk

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_NORMALIZE = re.compile(r"[^a-z0-9 ]+")
_SIGNATURE_WORDS = 6  # leading words used to locate a chunk on a page

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _default_token_length() -> Callable[[str], int]:
    """Build a token-count function from the MiniLM tokenizer (lazy import)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)

    def _length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return _length


def _normalize(text: str) -> str:
    return _NORMALIZE.sub(" ", text.lower()).strip()


class SectionChunker:
    """Splits Docling markdown into heading-scoped, token-bounded chunks."""

    def __init__(
        self,
        config: IngestionConfig,
        token_length: Callable[[str], int] | None = None,
    ) -> None:
        self._headers = list(config.headers)
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers,
            strip_headers=True,
        )
        length_function = token_length or _default_token_length()
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(
        self,
        markdown: str,
        page_texts: tuple[tuple[int, str], ...] = (),
    ) -> list[Chunk]:
        normalized_pages = [(no, _normalize(text)) for no, text in page_texts]
        chunks: list[Chunk] = []
        index = 0
        prev_values: dict[str, str] = {}
        for section in self._header_splitter.split_text(markdown):
            section_path = self._build_section_path(section.metadata, prev_values)
            prev_values = {
                name: section.metadata[name]
                for _, name in self._headers
                if name in section.metadata
            }
            for text in self._split_preserving_tables(section.page_content):
                cleaned = text.strip()
                if not cleaned:
                    continue
                chunks.append(
                    Chunk(
                        index=index,
                        text=cleaned,
                        section_path=section_path,
                        page_numbers=self._pages_for(cleaned, normalized_pages),
                    )
                )
                index += 1
        return chunks

    def _build_section_path(
        self, metadata: dict[str, str], prev_values: dict[str, str]
    ) -> str:
        """Join header values that changed since the previous section.

        Header values that repeat unchanged from the prior section (an ancestor
        heading still in scope) are dropped from the path; only the diverging
        tail is kept, e.g. a new "## Methods" under an unchanged "# Introduction"
        yields "Methods > Algorithm 3.2" rather than repeating "Introduction".
        """
        names = [name for _, name in self._headers]
        diverged = False
        parts: list[str] = []
        for name in names:
            current = metadata.get(name)
            if not diverged and current != prev_values.get(name):
                diverged = True
            if diverged and current is not None:
                parts.append(current)
        return " > ".join(parts)

    def _pages_for(
        self, chunk_text: str, normalized_pages: list[tuple[int, str]]
    ) -> tuple[int, ...]:
        if not normalized_pages:
            return ()
        signature = " ".join(_normalize(chunk_text).split()[:_SIGNATURE_WORDS])
        if not signature:
            return ()
        return tuple(no for no, page in normalized_pages if signature in page)

    def _split_preserving_tables(self, text: str) -> list[str]:
        """Split text by token budget but keep contiguous table blocks intact."""
        parts: list[str] = []
        buffer: list[str] = []
        in_table = False
        for line in text.splitlines():
            is_table = bool(_TABLE_LINE.match(line))
            if is_table and not in_table:
                parts.extend(self._flush_prose("\n".join(buffer)))
                buffer = [line]
                in_table = True
            elif is_table:
                buffer.append(line)
            elif in_table:
                parts.append("\n".join(buffer))  # emit whole table as one chunk
                buffer = [line]
                in_table = False
            else:
                buffer.append(line)
        if in_table:
            parts.append("\n".join(buffer))
        else:
            parts.extend(self._flush_prose("\n".join(buffer)))
        return parts

    def _flush_prose(self, prose: str) -> list[str]:
        if not prose.strip():
            return []
        return self._size_splitter.split_text(prose)
