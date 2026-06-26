"""Data models for the sports science search engine."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class Paper:
    """Represents an extracted academic paper with metadata."""

    filename: str
    title: str
    text: str
    sections: Dict[str, str]  # section_name → section_text
    metadata: Dict[str, Any]  # authors, year, DOI, journal, page_count


@dataclass(frozen=True)
class PaperChunk:
    """Represents a chunk of text from a paper with section metadata."""

    paper_filename: str
    paper_title: str
    section: str
    chunk_text: str
    chunk_index: int
    page_number: int
    chunk_strategy: str  # 'semantic', 'paragraph', or 'fixed'
    metadata: Dict[str, Any]  # year, authors, DOI, journal
    total_words: int = 0  # Auto-computed from chunk_text if not provided

    def __post_init__(self):
        """Compute total_words from chunk_text if not explicitly set."""
        if self.total_words == 0 and self.chunk_text:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, 'total_words', len(self.chunk_text.split()))


@dataclass(frozen=True)
class SearchResult:
    """Represents a search result with score and chunk data.

    Maintains flat structure for backwards compatibility with existing tests.
    """

    paper_filename: str
    paper_title: str
    section: str
    chunk_text: str
    score: float
    page_number: int
    metadata: Dict[str, Any]
    chunk_strategy: str
