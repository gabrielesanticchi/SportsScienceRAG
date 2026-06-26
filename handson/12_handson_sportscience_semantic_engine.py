"""Sports Science Semantic Search Engine

DEPRECATED: This file is kept for backwards compatibility only.
Import from the sports_science_search package instead:

    from sports_science_search import PDFExtractor, TextChunker, VectorStore, ...
"""

# Import everything from the new package for backwards compatibility
from sports_science_search import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    MIN_SECTIONS_FOR_AUTO_DETECTION,
    SECTION_PATTERNS,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_BUFFER_SIZE,
    PDFExtractionError,
    PDFProcessingError,
    SectionDetectionError,
    VectorUploadError,
    Paper,
    PaperChunk,
    SearchResult,
    PDFExtractor,
    TextChunker,
    VectorStore,
)

# Also import SearchEngine if needed
try:
    from sports_science_search.search_engine import SemanticSearchEngine
except ImportError:
    pass

__all__ = [
    # Constants
    "COLLECTION_NAME",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL_NAME",
    "FIXED_CHUNK_OVERLAP",
    "FIXED_CHUNK_SIZE",
    "MIN_SECTIONS_FOR_AUTO_DETECTION",
    "SECTION_PATTERNS",
    "SEMANTIC_BREAKPOINT_THRESHOLD",
    "SEMANTIC_BUFFER_SIZE",
    # Exceptions
    "PDFExtractionError",
    "PDFProcessingError",
    "SectionDetectionError",
    "VectorUploadError",
    # Models
    "Paper",
    "PaperChunk",
    "SearchResult",
    # Classes
    "PDFExtractor",
    "TextChunker",
    "VectorStore",
    "SemanticSearchEngine",
]
