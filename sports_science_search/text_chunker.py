"""Text chunking strategies for academic papers."""

import logging
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from sports_science_search.constants import (
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    SEMANTIC_BREAKPOINT_THRESHOLD,
    SEMANTIC_BUFFER_SIZE,
)
from sports_science_search.models import Paper, PaperChunk


class TextChunker:
    """Implements multiple chunking strategies for academic papers."""

    def __init__(self, embed_model: HuggingFaceEmbedding):
        """
        Initialize text chunker.

        Args:
            embed_model: Used for semantic chunking
        """
        self.embed_model = embed_model
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=SEMANTIC_BUFFER_SIZE,
            breakpoint_percentile_threshold=SEMANTIC_BREAKPOINT_THRESHOLD,
            embed_model=embed_model
        )

    def _create_chunk(
        self,
        paper: Paper,
        chunk_text: str,
        chunk_index: int,
        section: str,
        page_number: int,
        strategy: str
    ) -> PaperChunk:
        """Helper to create PaperChunk with inherited metadata."""
        return PaperChunk(
            paper_filename=paper.filename,
            paper_title=paper.title,
            section=section,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            page_number=page_number,
            total_words=len(chunk_text.split()),
            metadata=paper.metadata,
            chunk_strategy=strategy
        )

    def chunk_fixed_size(
        self,
        paper: Paper,
        chunk_size: int = FIXED_CHUNK_SIZE,
        overlap: int = FIXED_CHUNK_OVERLAP
    ) -> List[PaperChunk]:
        """
        Fixed-size chunking with word overlap.
        Baseline strategy for comparison.

        Args:
            paper: Paper object to chunk
            chunk_size: Number of words per chunk (default: 200)
            overlap: Number of overlapping words between chunks (default: 50)

        Returns:
            List of PaperChunk objects with strategy='fixed'

        Raises:
            ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size
        """
        # Input validation
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap cannot be negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        if not paper.text.strip():
            logging.warning(
                f"Empty text in paper {paper.filename}, returning empty chunks"
            )
            return []

        chunks: List[PaperChunk] = []

        # Split text into words
        words = paper.text.split()

        # Calculate estimated page for each word (rough approximation)
        page_count = paper.metadata.get("page_count", 1)
        if page_count <= 0:
            logging.warning(
                f"Invalid page_count {page_count} for {paper.filename}, defaulting to 1"
            )
            page_count = 1
        words_per_page = len(words) / page_count

        chunk_index = 0
        for start_idx in range(0, len(words), chunk_size - overlap):
            chunk_words = words[start_idx:start_idx + chunk_size]
            if not chunk_words:
                break

            chunk_text = ' '.join(chunk_words)

            # Estimate page number (1-indexed)
            estimated_page = int(start_idx / words_per_page) + 1 if words_per_page > 0 else 1

            # Determine section by matching chunk text to section texts
            # TODO(Task 8+): Improve section detection using character position mapping
            section = "Unknown"
            for section_name, section_text in paper.sections.items():
                if chunk_text[:50] in section_text or section_text[:50] in chunk_text:
                    section = section_name
                    break

            chunk = self._create_chunk(
                paper=paper,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                section=section,
                page_number=estimated_page,
                strategy="fixed"
            )
            chunks.append(chunk)
            chunk_index += 1

        return chunks

    def chunk_paragraph(self, paper: Paper) -> List[PaperChunk]:
        """
        Paragraph-based chunking on double newlines.
        Preserves natural paragraph structure.

        Args:
            paper: Paper object to chunk

        Returns:
            List of PaperChunk objects with strategy='paragraph'

        Notes:
            - Splits on double newlines (\\n\\n), falls back to single newlines
            - Filters empty paragraphs after stripping whitespace
            - Page estimation is approximate due to whitespace removal during splitting
            - Section matching uses prefix of 50 characters for consistency with chunk_fixed_size
        """
        # Guard against empty text (consistent with chunk_fixed_size pattern)
        if not paper.text.strip():
            logging.warning(
                f"Empty text in paper {paper.filename}, returning empty chunks"
            )
            return []

        chunks: List[PaperChunk] = []

        # Split on double newlines (paragraphs)
        paragraphs = paper.text.split('\n\n')

        # Fallback: if no double newlines found, split on single newlines
        if len(paragraphs) == 1:
            paragraphs = paper.text.split('\n')

        # Filter empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Calculate words per page for page estimation
        page_count = paper.metadata.get("page_count", 1)
        if page_count <= 0:
            logging.warning(
                f"Invalid page_count {page_count} for {paper.filename}, defaulting to 1"
            )
            page_count = 1
        words_per_page = len(paper.text.split()) / page_count

        chunk_index = 0
        word_position = 0

        for paragraph in paragraphs:
            # Estimate page number (1-indexed)
            # Note: This is approximate due to whitespace and newline removal during split
            estimated_page = int(word_position / words_per_page) + 1 if words_per_page > 0 else 1

            # Determine section by matching paragraph to section texts
            # Use 50-character prefix for consistency with chunk_fixed_size
            section = "Unknown"
            for section_name, section_text in paper.sections.items():
                if paragraph[:50] in section_text or section_text[:50] in paragraph:
                    section = section_name
                    break

            chunk = self._create_chunk(
                paper=paper,
                chunk_text=paragraph,
                chunk_index=chunk_index,
                section=section,
                page_number=estimated_page,
                strategy="paragraph"
            )
            chunks.append(chunk)

            chunk_index += 1
            word_position += len(paragraph.split())

        return chunks

    def chunk_semantic(self, paper: Paper) -> List[PaperChunk]:
        """
        Semantic chunking using SemanticSplitterNodeParser.
        Respects semantic boundaries, optimal for academic content.

        Args:
            paper: Paper object to chunk

        Returns:
            List of PaperChunk objects with strategy='semantic'

        Notes:
            - Uses SemanticSplitterNodeParser with buffer_size=1, threshold=95
            - Splits text into semantic nodes based on meaning boundaries
            - Page estimation is approximate based on word position
            - Section matching uses prefix of 30 characters for consistency
        """
        # Guard against empty text
        if not paper.text.strip():
            logging.warning(
                f"Empty text in paper {paper.filename}, returning empty chunks"
            )
            return []

        chunks: List[PaperChunk] = []

        # Create Document for llama-index
        document = Document(text=paper.text)

        # Split using semantic splitter
        nodes = self.semantic_splitter.get_nodes_from_documents([document])

        # Calculate words per page for page estimation
        page_count = paper.metadata.get("page_count", 1)
        if page_count <= 0:
            logging.warning(
                f"Invalid page_count {page_count} for {paper.filename}, defaulting to 1"
            )
            page_count = 1
        words_per_page = len(paper.text.split()) / page_count
        word_position = 0

        for chunk_index, node in enumerate(nodes):
            chunk_text = node.get_content()

            # Estimate page number (1-indexed)
            estimated_page = int(word_position / words_per_page) + 1 if words_per_page > 0 else 1

            # Determine section by matching chunk text to section texts
            section = "Unknown"
            for section_name, section_text in paper.sections.items():
                if chunk_text[:30] in section_text:
                    section = section_name
                    break

            chunk = self._create_chunk(
                paper=paper,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                section=section,
                page_number=estimated_page,
                strategy="semantic"
            )
            chunks.append(chunk)

            word_position += len(chunk_text.split())

        return chunks

    def chunk_all_strategies(self, paper: Paper) -> List[PaperChunk]:
        """
        Returns chunks from all three strategies.
        Combines results for parallel vectorization.

        Args:
            paper: Paper object to chunk

        Returns:
            List of PaperChunk objects combining semantic, paragraph, and fixed-size chunks
        """
        all_chunks: List[PaperChunk] = []

        # Apply all three strategies
        all_chunks.extend(self.chunk_semantic(paper))
        all_chunks.extend(self.chunk_paragraph(paper))
        all_chunks.extend(self.chunk_fixed_size(paper))

        return all_chunks
