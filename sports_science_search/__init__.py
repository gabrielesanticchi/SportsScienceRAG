"""Sports Science Semantic Search Engine

A semantic search engine for sports science research papers.
Extracts text from PDFs, chunks with multiple strategies, and provides semantic search via Qdrant.
"""

from sports_science_search.constants import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    MIN_SECTIONS_FOR_AUTO_DETECTION,
    SECTION_PATTERNS,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_BUFFER_SIZE,
)
from sports_science_search.exceptions import (
    PDFExtractionError,
    PDFProcessingError,
    SectionDetectionError,
    VectorUploadError,
)
from sports_science_search.models import Paper, PaperChunk, SearchResult
from sports_science_search.pdf_extractor import PDFExtractor
from sports_science_search.search_engine import SemanticSearchEngine
from sports_science_search.text_chunker import TextChunker
from sports_science_search.vector_store import VectorStore

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
    "SemanticSearchEngine",
    "TextChunker",
    "VectorStore",
]

__version__ = "0.1.0"
