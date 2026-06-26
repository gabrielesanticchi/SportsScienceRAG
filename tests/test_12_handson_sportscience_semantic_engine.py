"""Tests for sports science semantic search engine."""
import pytest
from pathlib import Path


def test_imports():
    """Test that all required modules can be imported."""
    import fitz  # PyMuPDF
    from sentence_transformers import SentenceTransformer
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from qdrant_client import QdrantClient, models
    from dataclasses import dataclass
    import re
    import json
    import os
    from typing import Dict, List, Optional, Any

    assert True  # If we get here, all imports work


def test_constants():
    """Test that constants are defined correctly."""
    import sys
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert len(module.SECTION_PATTERNS) == 6
    assert module.FIXED_CHUNK_SIZE == 200
    assert module.FIXED_CHUNK_OVERLAP == 50
    assert module.EMBEDDING_DIMENSION == 384


def test_error_classes():
    """Test custom error classes can be instantiated."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractionError = module.PDFExtractionError
    SectionDetectionError = module.SectionDetectionError
    VectorUploadError = module.VectorUploadError
    PDFProcessingError = module.PDFProcessingError

    # Test instantiation
    e1 = PDFExtractionError("test.pdf: cannot read")
    assert "test.pdf" in str(e1)

    e2 = SectionDetectionError("malone2017.pdf: no sections found")
    assert "malone2017.pdf" in str(e2)

    e3 = VectorUploadError("Upload failed: connection timeout")
    assert "connection timeout" in str(e3)

    e4 = PDFProcessingError("Processing failed at chunking stage")
    assert "chunking" in str(e4)


def test_paper_dataclass():
    """Test Paper dataclass is immutable and has correct fields."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Paper = module.Paper

    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text="Full text here",
        sections={"Abstract": "Abstract text", "Methods": "Methods text"},
        metadata={"authors": "Smith et al.", "year": 2020, "page_count": 10}
    )

    assert paper.filename == "test.pdf"
    assert paper.title == "Test Paper"
    assert len(paper.sections) == 2
    assert paper.metadata["year"] == 2020

    # Test immutability
    with pytest.raises(Exception):  # FrozenInstanceError
        paper.filename = "other.pdf"


def test_paper_chunk_dataclass():
    """Test PaperChunk dataclass is immutable and has correct fields."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PaperChunk = module.PaperChunk

    chunk = PaperChunk(
        paper_filename="test.pdf",
        paper_title="Test Paper",
        section="Methods",
        chunk_text="This is a chunk of text from the methods section.",
        chunk_index=0,
        page_number=3,
        metadata={"authors": "Smith et al.", "year": 2020},
        chunk_strategy="semantic"
    )

    assert chunk.paper_filename == "test.pdf"
    assert chunk.section == "Methods"
    assert chunk.chunk_index == 0
    assert chunk.chunk_strategy == "semantic"

    # Test immutability
    with pytest.raises(Exception):  # FrozenInstanceError
        chunk.chunk_text = "modified"


def test_search_result_dataclass():
    """Test SearchResult dataclass is immutable and has correct fields."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SearchResult = module.SearchResult

    result = SearchResult(
        paper_filename="test.pdf",
        paper_title="Test Paper",
        section="Results",
        chunk_text="GPS tracking showed high accuracy.",
        score=0.87,
        page_number=5,
        metadata={"authors": "Smith et al.", "year": 2020},
        chunk_strategy="paragraph"
    )

    assert result.score == 0.87
    assert result.section == "Results"
    assert result.chunk_strategy == "paragraph"

    # Test immutability
    with pytest.raises(Exception):  # FrozenInstanceError
        result.score = 0.99


def test_pdf_extractor_init():
    """Test PDFExtractor can be initialized."""
    import importlib.util
    import json

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    # Init without manual mappings
    extractor1 = PDFExtractor()
    assert extractor1.manual_mappings == {}

    # Init with non-existent manual mappings file
    extractor2 = PDFExtractor(manual_mappings_path=Path("nonexistent.json"))
    assert extractor2.manual_mappings == {}


def test_pdf_extractor_manual_mappings(tmp_path):
    """Test PDFExtractor loads manual mappings correctly."""
    import importlib.util
    import json

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    # Create test mappings file
    mappings = {
        "test.pdf": {
            "Abstract": {"start_page": 1, "end_page": 1},
            "Methods": {"start_page": 2, "end_page": 3}
        }
    }

    mappings_file = tmp_path / "mappings.json"
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f)

    # Load mappings
    extractor = PDFExtractor(manual_mappings_path=mappings_file)
    assert "test.pdf" in extractor.manual_mappings
    assert extractor.manual_mappings["test.pdf"]["Abstract"]["start_page"] == 1


def test_section_detection_success():
    """Test _detect_sections finds sections correctly."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    text = """
Abstract
This is the abstract text.

Introduction
This is the introduction.

Methods
This describes the methodology.

Results
These are the results.

Discussion
This is the discussion.

References
1. Citation one
2. Citation two
"""

    extractor = PDFExtractor()
    sections = extractor._detect_sections(text, "test.pdf")

    assert "Abstract" in sections
    assert "Methods" in sections
    assert "Results" in sections
    assert "Discussion" in sections
    assert "References" in sections
    assert "abstract text" in sections["Abstract"].lower()


def test_section_detection_failure_no_manual_mapping():
    """Test _detect_sections raises error when no sections found and no manual mapping."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor
    SectionDetectionError = module.SectionDetectionError

    text = "Some paper text with no clear sections."

    extractor = PDFExtractor()
    with pytest.raises(SectionDetectionError) as exc_info:
        extractor._detect_sections(text, "test.pdf")

    assert "test.pdf" in str(exc_info.value)
    assert "section_mappings.json" in str(exc_info.value)


def test_section_detection_with_manual_mapping(tmp_path):
    """Test _detect_sections uses manual mapping when auto-detection fails."""
    import importlib.util
    import json

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    # Text with no clear sections
    text = "Page 1 content\nPage 2 content\nPage 3 content"

    # Manual mapping
    mappings = {
        "test.pdf": {
            "Abstract": {"start_page": 1, "end_page": 1},
            "Methods": {"start_page": 2, "end_page": 3}
        }
    }

    mappings_file = tmp_path / "mappings.json"
    with open(mappings_file, 'w') as f:
        json.dump(mappings, f)

    extractor = PDFExtractor(manual_mappings_path=mappings_file)
    sections = extractor._detect_sections(text, "test.pdf")

    # Should return sections from manual mapping
    assert "Abstract" in sections
    assert "Methods" in sections


def test_metadata_extraction():
    """Test _extract_metadata extracts basic metadata."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    # Use actual PDF from assets if available, otherwise skip
    pdf_path = Path("assets/malone2017.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF not available")

    extractor = PDFExtractor()

    # Extract text first (minimal)
    import fitz
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    metadata = extractor._extract_metadata(pdf_path, text)

    assert "page_count" in metadata
    assert metadata["page_count"] > 0
    assert "year" in metadata
    assert "title" in metadata


def test_metadata_extraction_year_from_filename():
    """Test _extract_metadata can extract year from filename."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    extractor = PDFExtractor()

    # Test with a fake PDF path (year extraction from filename)
    # Method should gracefully handle non-existent PDFs
    text = "Some paper text with DOI: 10.1234/example"

    metadata = extractor._extract_metadata(Path("test2020.pdf"), text)
    # Year should be extracted from filename
    assert isinstance(metadata, dict)
    assert "page_count" in metadata
    assert metadata["year"] == 2020
    assert "title" in metadata
    assert "authors" in metadata
    assert "DOI" in metadata
    assert metadata["DOI"] == "10.1234/example"
    assert "journal" in metadata


def test_metadata_extraction_returns_correct_keys():
    """Test _extract_metadata returns all required keys."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor

    extractor = PDFExtractor()

    # Test that the method returns a dict with all required keys
    # even if the PDF doesn't exist (it should handle gracefully)
    text = "Sample text"

    metadata = extractor._extract_metadata(Path("nonexistent.pdf"), text)
    # Method should return a dict with these keys
    required_keys = {"authors", "year", "DOI", "journal", "page_count", "title"}
    assert all(key in metadata for key in required_keys)
    assert isinstance(metadata, dict)


def test_pdf_extractor_full_extraction():
    """Test PDFExtractor.extract() returns a Paper object."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor
    Paper = module.Paper

    # Use actual PDF from assets if available
    pdf_path = Path("assets/malone2017.pdf")
    if not pdf_path.exists():
        pytest.skip("Test PDF not available")

    extractor = PDFExtractor()
    paper = extractor.extract(pdf_path)

    assert isinstance(paper, Paper)
    assert paper.filename == "malone2017.pdf"
    assert len(paper.text) > 100
    assert len(paper.sections) >= 3
    assert paper.metadata["page_count"] > 0
    assert paper.title is not None


def test_pdf_extractor_nonexistent_file():
    """Test PDFExtractor.extract() raises error for nonexistent file."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    PDFExtractor = module.PDFExtractor
    PDFExtractionError = module.PDFExtractionError

    extractor = PDFExtractor()
    with pytest.raises(PDFExtractionError) as exc_info:
        extractor.extract(Path("nonexistent.pdf"))

    assert "nonexistent.pdf" in str(exc_info.value)


def test_text_chunker_init():
    """Test TextChunker can be initialized."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    assert chunker.embed_model is not None
    assert chunker.semantic_splitter is not None


def test_chunk_fixed_size():
    """Test fixed-size chunking with overlap."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    PaperChunk = module.PaperChunk
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Create test paper
    text = " ".join([f"word{i}" for i in range(500)])  # 500 words
    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 5}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_fixed_size(paper, chunk_size=200, overlap=50)

    assert len(chunks) > 0
    assert all(isinstance(chunk, PaperChunk) for chunk in chunks)
    assert all(chunk.chunk_strategy == "fixed" for chunk in chunks)
    assert all(chunk.paper_filename == "test.pdf" for chunk in chunks)

    # Check chunk sizes (approximately 200 words each, except possibly last chunk)
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.chunk_text.split())
        if i < len(chunks) - 1:
            # Non-last chunks should be close to chunk_size
            assert 150 <= word_count <= 250  # Allow some variance
        else:
            # Last chunk can be smaller (remaining words)
            assert 1 <= word_count <= 250


def test_create_chunk_helper():
    """Test _create_chunk helper method."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    PaperChunk = module.PaperChunk
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text="Full text",
        sections={"Methods": "Methods text"},
        metadata={"authors": "Smith et al.", "year": 2020, "page_count": 5}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunk = chunker._create_chunk(
        paper=paper,
        chunk_text="This is a test chunk",
        chunk_index=0,
        section="Methods",
        page_number=2,
        strategy="fixed"
    )

    assert isinstance(chunk, PaperChunk)
    assert chunk.paper_filename == "test.pdf"
    assert chunk.paper_title == "Test Paper"
    assert chunk.section == "Methods"
    assert chunk.chunk_text == "This is a test chunk"
    assert chunk.chunk_index == 0
    assert chunk.page_number == 2
    assert chunk.chunk_strategy == "fixed"
    assert chunk.metadata["year"] == 2020


def test_chunk_fixed_size_validation():
    """Test input validation in chunk_fixed_size."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text="Some text",
        sections={"Methods": "Some text"},
        metadata={"year": 2020, "page_count": 5}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    # Test negative chunk_size
    with pytest.raises(ValueError) as exc_info:
        chunker.chunk_fixed_size(paper, chunk_size=0, overlap=50)
    assert "chunk_size must be positive" in str(exc_info.value)

    # Test negative overlap
    with pytest.raises(ValueError) as exc_info:
        chunker.chunk_fixed_size(paper, chunk_size=200, overlap=-1)
    assert "overlap cannot be negative" in str(exc_info.value)

    # Test overlap >= chunk_size
    with pytest.raises(ValueError) as exc_info:
        chunker.chunk_fixed_size(paper, chunk_size=200, overlap=200)
    assert "overlap" in str(exc_info.value) and "must be less than chunk_size" in str(exc_info.value)


def test_chunk_fixed_size_empty_text():
    """Test chunk_fixed_size handles empty text gracefully."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text="",  # Empty text
        sections={},
        metadata={"year": 2020, "page_count": 5}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_fixed_size(paper, chunk_size=200, overlap=50)
    assert chunks == []


def test_chunk_paragraph():
    """Test paragraph-based chunking on double newlines."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    PaperChunk = module.PaperChunk
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Create test paper with clear paragraphs separated by double newlines
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 1}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_paragraph(paper)

    # Verify correct number of chunks
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"

    # Verify all chunks are PaperChunk instances
    assert all(isinstance(chunk, PaperChunk) for chunk in chunks)

    # Verify all chunks use paragraph strategy
    assert all(chunk.chunk_strategy == "paragraph" for chunk in chunks)

    # Verify chunk contents and order
    assert "Paragraph one" in chunks[0].chunk_text
    assert "Paragraph two" in chunks[1].chunk_text
    assert "Paragraph three" in chunks[2].chunk_text

    # Verify metadata is inherited
    assert all(chunk.paper_filename == "test.pdf" for chunk in chunks)
    assert all(chunk.paper_title == "Test Paper" for chunk in chunks)
    assert all(chunk.metadata["year"] == 2020 for chunk in chunks)


def test_chunk_paragraph_fallback_single_newline():
    """Test paragraph chunking falls back to single newlines when no double newlines."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Create test paper with single newlines only (no double newlines)
    text = "Line one\nLine two\nLine three"
    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 1}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_paragraph(paper)

    # Should split on single newlines when no double newlines found
    assert len(chunks) == 3
    assert all(chunk.chunk_strategy == "paragraph" for chunk in chunks)
    assert "Line one" in chunks[0].chunk_text
    assert "Line two" in chunks[1].chunk_text
    assert "Line three" in chunks[2].chunk_text


def test_chunk_paragraph_with_empty_paragraphs():
    """Test paragraph chunking filters out empty paragraphs."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Create test paper with multiple empty paragraphs
    text = "Paragraph one.\n\n\n\nParagraph two.\n\n   \n\nParagraph three."
    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 1}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_paragraph(paper)

    # Should only include non-empty paragraphs
    assert len(chunks) == 3
    assert all(chunk.chunk_text.strip() for chunk in chunks)


def test_chunk_paragraph_section_detection():
    """Test paragraph chunking detects sections correctly."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Create test paper with distinct sections
    methods_text = "Method paragraph one.\n\nMethod paragraph two."
    results_text = "Result paragraph one.\n\nResult paragraph two."
    text = methods_text + "\n\n" + results_text

    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": methods_text, "Results": results_text},
        metadata={"year": 2020, "page_count": 2}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_paragraph(paper)

    # All chunks should have a detected section (even if "Unknown")
    assert all(chunk.section in ["Methods", "Results", "Unknown"] for chunk in chunks)
    assert len(chunks) == 4


def test_chunk_paragraph():
    """Test paragraph-based chunking."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    PaperChunk = module.PaperChunk
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    # Create test paper with clear paragraphs
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 1}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_paragraph(paper)

    assert len(chunks) == 3
    assert all(isinstance(chunk, PaperChunk) for chunk in chunks)
    assert all(chunk.chunk_strategy == "paragraph" for chunk in chunks)
    assert all(chunk.paper_filename == "test.pdf" for chunk in chunks)
    # Verify paragraph order
    assert "Paragraph one" in chunks[0].chunk_text
    assert "Paragraph two" in chunks[1].chunk_text
    assert "Paragraph three" in chunks[2].chunk_text


def test_chunk_semantic():
    """Test semantic chunking using SemanticSplitterNodeParser."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    TextChunker = module.TextChunker
    Paper = module.Paper
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    text = "This is about GPS tracking. " * 50 + "This is about injury prevention. " * 50
    paper = Paper(
        filename="test.pdf",
        title="Test Paper",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 2}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    chunks = chunker.chunk_semantic(paper)

    assert len(chunks) > 0
    assert all(chunk.chunk_strategy == "semantic" for chunk in chunks)


def test_chunk_all_strategies():
    """Test chunk_all_strategies combines all three methods."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    TextChunker = module.TextChunker
    Paper = module.Paper

    text = " ".join([f"word{i}" for i in range(300)])
    paper = Paper(
        filename="test.pdf",
        title="Test",
        text=text,
        sections={"Methods": text},
        metadata={"year": 2020, "page_count": 2}
    )

    embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
    chunker = TextChunker(embed_model)

    all_chunks = chunker.chunk_all_strategies(paper)

    strategies = {chunk.chunk_strategy for chunk in all_chunks}
    assert "semantic" in strategies
    assert "paragraph" in strategies
    assert "fixed" in strategies


def test_vector_store_init():
    """Test VectorStore initialization."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")

    assert store.client is not None
    assert store.collection_name == "test_collection"


def test_vector_store_create_collection():
    """Test create_collection creates proper Qdrant collection."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")

    store.create_collection(dimension=384, recreate=True)

    # Verify collection exists
    assert client.collection_exists(collection_name="test_collection")

    # Verify collection has three named vectors
    collection_info = client.get_collection(collection_name="test_collection")
    assert "semantic" in collection_info.config.params.vectors
    assert "paragraph" in collection_info.config.params.vectors
    assert "fixed" in collection_info.config.params.vectors

    # Verify vector dimensions
    assert collection_info.config.params.vectors["semantic"].size == 384
    assert collection_info.config.params.vectors["paragraph"].size == 384
    assert collection_info.config.params.vectors["fixed"].size == 384


def test_vector_store_create_collection_invalid_dimension():
    """Test create_collection raises ValueError for invalid dimension."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")

    # Test negative dimension
    with pytest.raises(ValueError) as exc_info:
        store.create_collection(dimension=-1)
    assert "dimension must be positive" in str(exc_info.value)

    # Test zero dimension
    with pytest.raises(ValueError) as exc_info:
        store.create_collection(dimension=0)
    assert "dimension must be positive" in str(exc_info.value)


def test_upload_chunks_basic():
    """Test upload_chunks vectorizes and uploads chunks to Qdrant."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create test chunks with different strategies
    chunks = [
        PaperChunk(
            paper_filename="test.pdf",
            paper_title="Test Paper",
            section="Methods",
            chunk_text="This is a semantic chunk about GPS tracking.",
            chunk_index=0,
            page_number=1,
            metadata={"authors": "Smith et al.", "year": 2020, "DOI": "10.1234/test", "journal": "Nature"},
            chunk_strategy="semantic"
        ),
        PaperChunk(
            paper_filename="test.pdf",
            paper_title="Test Paper",
            section="Results",
            chunk_text="This is a paragraph chunk with results.",
            chunk_index=1,
            page_number=2,
            metadata={"authors": "Smith et al.", "year": 2020, "DOI": "10.1234/test", "journal": "Nature"},
            chunk_strategy="paragraph"
        ),
        PaperChunk(
            paper_filename="test.pdf",
            paper_title="Test Paper",
            section="Discussion",
            chunk_text="This is a fixed-size chunk from discussion.",
            chunk_index=2,
            page_number=3,
            metadata={"authors": "Smith et al.", "year": 2020, "DOI": "10.1234/test", "journal": "Nature"},
            chunk_strategy="fixed"
        ),
    ]

    # Upload chunks
    store.upload_chunks(chunks, encoder, batch_size=2)

    # Verify points were uploaded
    collection_info = client.get_collection(collection_name="test_collection")
    assert collection_info.points_count == 3

    # Verify point structure by scrolling through points
    points, _ = client.scroll(collection_name="test_collection", limit=3, with_vectors=True)

    assert len(points) == 3

    # Verify first point (semantic strategy)
    point_0 = points[0]
    assert "semantic" in point_0.vector
    assert "paragraph" not in point_0.vector
    assert "fixed" not in point_0.vector
    assert len(point_0.vector["semantic"]) == 384
    assert point_0.payload["paper_filename"] == "test.pdf"
    assert point_0.payload["section"] == "Methods"
    assert point_0.payload["chunk_strategy"] == "semantic"
    assert point_0.payload["authors"] == "Smith et al."
    assert point_0.payload["year"] == 2020
    assert point_0.payload["DOI"] == "10.1234/test"
    assert point_0.payload["journal"] == "Nature"

    # Verify second point (paragraph strategy)
    point_1 = points[1]
    assert "paragraph" in point_1.vector
    assert "semantic" not in point_1.vector
    assert "fixed" not in point_1.vector
    assert point_1.payload["chunk_strategy"] == "paragraph"

    # Verify third point (fixed strategy)
    point_2 = points[2]
    assert "fixed" in point_2.vector
    assert "semantic" not in point_2.vector
    assert "paragraph" not in point_2.vector
    assert point_2.payload["chunk_strategy"] == "fixed"


def test_upload_chunks_batch_processing():
    """Test upload_chunks processes chunks in batches."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create 250 chunks to test batch processing with batch_size=100
    chunks = [
        PaperChunk(
            paper_filename="test.pdf",
            paper_title="Test Paper",
            section="Methods",
            chunk_text=f"Chunk number {i} with some text content.",
            chunk_index=i,
            page_number=i // 10 + 1,
            metadata={"authors": "Smith et al.", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="semantic"
        )
        for i in range(250)
    ]

    # Upload with batch_size=100 (should create 3 batches: 100, 100, 50)
    store.upload_chunks(chunks, encoder, batch_size=100)

    # Verify all chunks were uploaded
    collection_info = client.get_collection(collection_name="test_collection")
    assert collection_info.points_count == 250


def test_upload_chunks_empty_list():
    """Test upload_chunks handles empty chunk list gracefully."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload empty list - should not raise error
    store.upload_chunks([], encoder, batch_size=100)

    # Verify no points were uploaded
    collection_info = client.get_collection(collection_name="test_collection")
    assert collection_info.points_count == 0


def test_upload_chunks_metadata_with_none_values():
    """Test upload_chunks handles None values in metadata."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Create chunk with None metadata values
    chunk = PaperChunk(
        paper_filename="test.pdf",
        paper_title="Test Paper",
        section="Methods",
        chunk_text="Chunk with missing metadata.",
        chunk_index=0,
        page_number=1,
        metadata={"authors": None, "year": None, "DOI": None, "journal": None},
        chunk_strategy="semantic"
    )

    # Upload - should not raise error
    store.upload_chunks([chunk], encoder)

    # Verify point was uploaded with None values
    points, _ = client.scroll(collection_name="test_collection", limit=1)
    assert len(points) == 1
    assert points[0].payload["authors"] is None
    assert points[0].payload["year"] is None
    assert points[0].payload["DOI"] is None
    assert points[0].payload["journal"] is None


# Task 13: Search Method Tests


def test_search_basic():
    """Test search returns relevant results using specified strategy."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    SearchResult = module.SearchResult
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload test chunks
    chunks = [
        PaperChunk(
            paper_filename="paper1.pdf",
            paper_title="GPS Tracking Study",
            section="Methods",
            chunk_text="GPS devices were used to track player movements at 10Hz sampling rate.",
            chunk_index=0,
            page_number=3,
            metadata={"authors": "Smith et al.", "year": 2020, "DOI": "10.1234/gps", "journal": "Nature"},
            chunk_strategy="semantic"
        ),
        PaperChunk(
            paper_filename="paper2.pdf",
            paper_title="Heart Rate Analysis",
            section="Results",
            chunk_text="Heart rate monitors showed elevated cardiovascular response during high-intensity intervals.",
            chunk_index=0,
            page_number=5,
            metadata={"authors": "Jones et al.", "year": 2019, "DOI": "10.5678/hr", "journal": "Science"},
            chunk_strategy="semantic"
        ),
    ]
    store.upload_chunks(chunks, encoder)

    # Search for GPS-related content
    results = store.search(
        query="GPS tracking accuracy",
        encoder=encoder,
        strategy="semantic",
        limit=2
    )

    # Verify results
    assert len(results) <= 2
    assert len(results) > 0
    assert isinstance(results[0], SearchResult)
    assert results[0].paper_filename in ["paper1.pdf", "paper2.pdf"]
    assert results[0].score > 0
    assert results[0].chunk_strategy == "semantic"
    assert "metadata" in results[0].__dict__


def test_search_with_section_filter():
    """Test search filters results by section."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload chunks from different sections
    chunks = [
        PaperChunk(
            paper_filename="paper1.pdf",
            paper_title="Test Paper",
            section="Methods",
            chunk_text="We used GPS tracking to monitor players.",
            chunk_index=0,
            page_number=2,
            metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="paragraph"
        ),
        PaperChunk(
            paper_filename="paper1.pdf",
            paper_title="Test Paper",
            section="Results",
            chunk_text="GPS data showed average speed of 5.2 km/h.",
            chunk_index=1,
            page_number=4,
            metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="paragraph"
        ),
    ]
    store.upload_chunks(chunks, encoder)

    # Search with section filter
    results = store.search(
        query="GPS tracking",
        encoder=encoder,
        strategy="paragraph",
        limit=5,
        section_filter="Methods"
    )

    # All results should be from Methods section
    assert len(results) > 0
    for result in results:
        assert result.section == "Methods"


def test_search_with_year_filter():
    """Test search filters results by publication year."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload chunks from different years
    chunks = [
        PaperChunk(
            paper_filename="paper2019.pdf",
            paper_title="2019 Study",
            section="Methods",
            chunk_text="Sports tracking with wearable sensors.",
            chunk_index=0,
            page_number=1,
            metadata={"authors": "Smith", "year": 2019, "DOI": None, "journal": None},
            chunk_strategy="fixed"
        ),
        PaperChunk(
            paper_filename="paper2020.pdf",
            paper_title="2020 Study",
            section="Methods",
            chunk_text="Advanced sports tracking technology.",
            chunk_index=0,
            page_number=1,
            metadata={"authors": "Jones", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="fixed"
        ),
    ]
    store.upload_chunks(chunks, encoder)

    # Search with year filter
    results = store.search(
        query="tracking",
        encoder=encoder,
        strategy="fixed",
        limit=5,
        year_filter=2020
    )

    # All results should be from 2020
    assert len(results) > 0
    for result in results:
        assert result.metadata["year"] == 2020


def test_search_with_paper_filter():
    """Test search filters results by paper filename."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload chunks from different papers
    chunks = [
        PaperChunk(
            paper_filename="smith2020.pdf",
            paper_title="Smith Study",
            section="Methods",
            chunk_text="Player performance analysis.",
            chunk_index=0,
            page_number=1,
            metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="semantic"
        ),
        PaperChunk(
            paper_filename="jones2020.pdf",
            paper_title="Jones Study",
            section="Methods",
            chunk_text="Performance tracking systems.",
            chunk_index=0,
            page_number=1,
            metadata={"authors": "Jones", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="semantic"
        ),
    ]
    store.upload_chunks(chunks, encoder)

    # Search with paper filter
    results = store.search(
        query="performance",
        encoder=encoder,
        strategy="semantic",
        limit=5,
        paper_filter="smith2020.pdf"
    )

    # All results should be from smith2020.pdf
    assert len(results) > 0
    for result in results:
        assert result.paper_filename == "smith2020.pdf"


def test_search_with_multiple_filters():
    """Test search with combined filters (section + year + paper)."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload diverse chunks
    chunks = [
        PaperChunk(
            paper_filename="smith2020.pdf",
            paper_title="Smith Study",
            section="Methods",
            chunk_text="Tracking methodology using GPS.",
            chunk_index=0,
            page_number=2,
            metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="semantic"
        ),
        PaperChunk(
            paper_filename="smith2020.pdf",
            paper_title="Smith Study",
            section="Results",
            chunk_text="GPS tracking showed high accuracy.",
            chunk_index=1,
            page_number=5,
            metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="semantic"
        ),
        PaperChunk(
            paper_filename="jones2019.pdf",
            paper_title="Jones Study",
            section="Methods",
            chunk_text="GPS-based tracking system.",
            chunk_index=0,
            page_number=3,
            metadata={"authors": "Jones", "year": 2019, "DOI": None, "journal": None},
            chunk_strategy="semantic"
        ),
    ]
    store.upload_chunks(chunks, encoder)

    # Search with multiple filters
    results = store.search(
        query="GPS",
        encoder=encoder,
        strategy="semantic",
        limit=5,
        section_filter="Methods",
        year_filter=2020,
        paper_filter="smith2020.pdf"
    )

    # Should only return the first chunk
    assert len(results) == 1
    assert results[0].section == "Methods"
    assert results[0].metadata["year"] == 2020
    assert results[0].paper_filename == "smith2020.pdf"


def test_search_no_results():
    """Test search returns empty list when no matches found."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload chunk
    chunk = PaperChunk(
        paper_filename="test.pdf",
        paper_title="Test",
        section="Methods",
        chunk_text="GPS tracking.",
        chunk_index=0,
        page_number=1,
        metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
        chunk_strategy="semantic"
    )
    store.upload_chunks([chunk], encoder)

    # Search with filter that won't match
    results = store.search(
        query="tracking",
        encoder=encoder,
        strategy="semantic",
        limit=5,
        year_filter=1999  # No chunks from 1999
    )

    assert len(results) == 0


def test_search_respects_limit():
    """Test search returns at most limit results."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload 5 chunks
    chunks = [
        PaperChunk(
            paper_filename="test.pdf",
            paper_title="Test",
            section="Methods",
            chunk_text=f"Tracking methodology chunk {i}.",
            chunk_index=i,
            page_number=i+1,
            metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
            chunk_strategy="paragraph"
        )
        for i in range(5)
    ]
    store.upload_chunks(chunks, encoder)

    # Search with limit=2
    results = store.search(
        query="tracking methodology",
        encoder=encoder,
        strategy="paragraph",
        limit=2
    )

    assert len(results) == 2


def test_search_results_have_scores():
    """Test search results contain relevance scores."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    PaperChunk = module.PaperChunk
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    store.create_collection(dimension=384)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Upload chunk
    chunk = PaperChunk(
        paper_filename="test.pdf",
        paper_title="Test",
        section="Methods",
        chunk_text="GPS tracking accuracy study.",
        chunk_index=0,
        page_number=1,
        metadata={"authors": "Smith", "year": 2020, "DOI": None, "journal": None},
        chunk_strategy="fixed"
    )
    store.upload_chunks([chunk], encoder)

    # Search
    results = store.search(
        query="GPS accuracy",
        encoder=encoder,
        strategy="fixed",
        limit=1
    )

    assert len(results) == 1
    assert isinstance(results[0].score, float)
    assert 0 <= results[0].score <= 1  # Cosine similarity range


def test_search_empty_query_raises_error():
    """Test search raises ValueError for empty query."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import pytest

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Test empty string
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        store.search(query="", encoder=encoder, strategy="semantic")

    # Test whitespace only
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        store.search(query="   ", encoder=encoder, strategy="semantic")


def test_search_invalid_strategy_raises_error():
    """Test search raises ValueError for invalid strategy."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import pytest

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Test invalid strategy
    with pytest.raises(ValueError, match="strategy must be one of"):
        store.search(query="test", encoder=encoder, strategy="invalid")


def test_search_invalid_limit_raises_error():
    """Test search raises ValueError for invalid limit."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import pytest

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Test zero limit
    with pytest.raises(ValueError, match="limit must be positive"):
        store.search(query="test", encoder=encoder, strategy="semantic", limit=0)

    # Test negative limit
    with pytest.raises(ValueError, match="limit must be positive"):
        store.search(query="test", encoder=encoder, strategy="semantic", limit=-1)

    # Test too large limit
    with pytest.raises(ValueError, match="limit must be <= 1000"):
        store.search(query="test", encoder=encoder, strategy="semantic", limit=1001)


def test_search_filter_too_long_raises_error():
    """Test search raises ValueError for excessively long filter strings."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    VectorStore = module.VectorStore
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
    import pytest

    # Setup
    client = QdrantClient(":memory:")
    store = VectorStore(client, "test_collection")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Test section_filter too long (> 200 chars)
    with pytest.raises(ValueError, match="section_filter too long"):
        store.search(
            query="test",
            encoder=encoder,
            strategy="semantic",
            section_filter="x" * 201
        )

    # Test paper_filter too long (> 500 chars)
    with pytest.raises(ValueError, match="paper_filter too long"):
        store.search(
            query="test",
            encoder=encoder,
            strategy="semantic",
            paper_filter="y" * 501
        )


# Task 14: get_collection_stats Tests


def _load_module():
    """Load the module under test and return it."""
    import importlib.util

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"
    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_collection_stats_returns_dict_with_required_keys():
    """get_collection_stats must return a dict with the four required top-level keys."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    client = MagicMock()
    # scroll returns (list_of_points, next_page_offset)
    client.scroll.return_value = ([], None)

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    assert isinstance(stats, dict), "return value must be a dict"
    for key in ("total_points", "chunks_by_strategy", "chunks_by_section", "unique_papers"):
        assert key in stats, f"missing key: {key}"


def test_get_collection_stats_empty_collection():
    """Empty collection must yield zeroed statistics."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    client = MagicMock()
    client.scroll.return_value = ([], None)

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    assert stats["total_points"] == 0
    assert stats["chunks_by_strategy"] == {}
    assert stats["chunks_by_section"] == {}
    assert stats["unique_papers"] == 0


def test_get_collection_stats_calls_scroll_with_correct_args():
    """client.scroll must be called with the collection name, limit=10000, and offset."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    client = MagicMock()
    client.scroll.return_value = ([], None)

    store = VectorStore(client, "my_collection")
    store.get_collection_stats()

    # First call should have offset=None, and since points is empty, it should exit the loop
    client.scroll.assert_called_once_with(
        collection_name="my_collection",
        limit=10000,
        offset=None
    )


def test_get_collection_stats_counts_strategies():
    """chunks_by_strategy must count each chunk_strategy value correctly."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    def _make_point(strategy: str, section: str, filename: str) -> MagicMock:
        point = MagicMock()
        point.payload = {
            "chunk_strategy": strategy,
            "section": section,
            "paper_filename": filename,
        }
        return point

    points = [
        _make_point("semantic", "Methods", "paper_a.pdf"),
        _make_point("semantic", "Results", "paper_a.pdf"),
        _make_point("paragraph", "Discussion", "paper_b.pdf"),
        _make_point("fixed", "Abstract", "paper_a.pdf"),
        _make_point("fixed", "Methods", "paper_c.pdf"),
    ]

    client = MagicMock()
    client.scroll.return_value = (points, None)

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    assert stats["chunks_by_strategy"]["semantic"] == 2
    assert stats["chunks_by_strategy"]["paragraph"] == 1
    assert stats["chunks_by_strategy"]["fixed"] == 2


def test_get_collection_stats_counts_sections():
    """chunks_by_section must aggregate point counts per section name."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    def _make_point(strategy: str, section: str, filename: str) -> MagicMock:
        point = MagicMock()
        point.payload = {
            "chunk_strategy": strategy,
            "section": section,
            "paper_filename": filename,
        }
        return point

    points = [
        _make_point("semantic", "Methods", "paper_a.pdf"),
        _make_point("paragraph", "Methods", "paper_b.pdf"),
        _make_point("fixed", "Results", "paper_a.pdf"),
        _make_point("semantic", "Discussion", "paper_c.pdf"),
    ]

    client = MagicMock()
    client.scroll.return_value = (points, None)

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    assert stats["chunks_by_section"]["Methods"] == 2
    assert stats["chunks_by_section"]["Results"] == 1
    assert stats["chunks_by_section"]["Discussion"] == 1


def test_get_collection_stats_unique_papers():
    """unique_papers must count distinct paper_filename values."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    def _make_point(strategy: str, section: str, filename: str) -> MagicMock:
        point = MagicMock()
        point.payload = {
            "chunk_strategy": strategy,
            "section": section,
            "paper_filename": filename,
        }
        return point

    # Three points but only two distinct filenames
    points = [
        _make_point("semantic", "Methods", "paper_a.pdf"),
        _make_point("paragraph", "Results", "paper_a.pdf"),
        _make_point("fixed", "Discussion", "paper_b.pdf"),
    ]

    client = MagicMock()
    client.scroll.return_value = (points, None)

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    assert stats["unique_papers"] == 2


def test_get_collection_stats_total_points_matches_scroll_length():
    """total_points must equal the number of points returned by scroll."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    def _make_point(strategy: str, section: str, filename: str) -> MagicMock:
        point = MagicMock()
        point.payload = {
            "chunk_strategy": strategy,
            "section": section,
            "paper_filename": filename,
        }
        return point

    points = [_make_point("semantic", "Abstract", f"p{i}.pdf") for i in range(42)]

    client = MagicMock()
    client.scroll.return_value = (points, None)

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    assert stats["total_points"] == 42


def test_get_collection_stats_missing_payload_fields_handled_gracefully():
    """Points missing payload fields must not cause KeyError; use .get() defaults."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    # Point with entirely empty payload
    point_empty = MagicMock()
    point_empty.payload = {}

    # Point with only chunk_strategy set
    point_partial = MagicMock()
    point_partial.payload = {"chunk_strategy": "semantic"}

    client = MagicMock()
    client.scroll.return_value = ([point_empty, point_partial], None)

    store = VectorStore(client, "sports_papers")
    # Should not raise
    stats = store.get_collection_stats()

    assert stats["total_points"] == 2
    # None values from missing fields must not appear as real keys causing errors
    assert isinstance(stats["chunks_by_strategy"], dict)
    assert isinstance(stats["chunks_by_section"], dict)


def test_get_collection_stats_pagination():
    """get_collection_stats must paginate through scroll results with offset."""
    from unittest.mock import MagicMock

    module = _load_module()
    VectorStore = module.VectorStore

    def _make_point(strategy: str, section: str, filename: str) -> MagicMock:
        point = MagicMock()
        point.payload = {
            "chunk_strategy": strategy,
            "section": section,
            "paper_filename": filename,
        }
        return point

    # Simulate pagination: first call returns 2 points with offset=1,
    # second call (with offset=1) returns 1 point with offset=None (end)
    first_batch = [
        _make_point("semantic", "Methods", "paper_a.pdf"),
        _make_point("paragraph", "Results", "paper_b.pdf"),
    ]

    second_batch = [
        _make_point("fixed", "Discussion", "paper_c.pdf"),
    ]

    client = MagicMock()
    client.scroll.side_effect = [
        (first_batch, 1),  # First call: points, next_offset
        (second_batch, None),  # Second call: points, no more offset (end)
    ]

    store = VectorStore(client, "sports_papers")
    stats = store.get_collection_stats()

    # Verify both batches were processed
    assert stats["total_points"] == 3
    assert stats["chunks_by_strategy"]["semantic"] == 1
    assert stats["chunks_by_strategy"]["paragraph"] == 1
    assert stats["chunks_by_strategy"]["fixed"] == 1
    assert stats["unique_papers"] == 3


def test_get_collection_stats_error_handling():
    """get_collection_stats must raise VectorUploadError on client failures."""
    from unittest.mock import MagicMock
    import pytest

    module = _load_module()
    VectorStore = module.VectorStore
    VectorUploadError = module.VectorUploadError

    client = MagicMock()
    client.scroll.side_effect = RuntimeError("Connection timeout")

    store = VectorStore(client, "sports_papers")

    with pytest.raises(VectorUploadError, match="Failed to get collection stats"):
        store.get_collection_stats()


# Task 15: SemanticSearchEngine Tests


def test_semantic_search_engine_init():
    """Test SemanticSearchEngine initialization."""
    import importlib.util
    from qdrant_client import QdrantClient

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SemanticSearchEngine = module.SemanticSearchEngine

    # Create temporary PDF directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_dir = Path(tmp_dir)
        client = QdrantClient(":memory:")

        # Initialize engine
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        # Verify all components initialized
        assert engine.pdf_dir == pdf_dir
        assert engine.extractor is not None
        assert engine.embed_model is not None
        assert engine.chunker is not None
        assert engine.vector_store is not None
        assert engine.encoder is not None


def test_semantic_search_engine_init_with_manual_mappings():
    """Test SemanticSearchEngine initialization with manual mappings."""
    import importlib.util
    from qdrant_client import QdrantClient
    import json

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SemanticSearchEngine = module.SemanticSearchEngine

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_dir = Path(tmp_dir)

        # Create manual mappings file
        mappings = {"test.pdf": {"Abstract": {"start_page": 1, "end_page": 1}}}
        mappings_path = Path(tmp_dir) / "mappings.json"
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f)

        client = QdrantClient(":memory:")

        # Initialize with mappings
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection",
            manual_mappings_path=mappings_path
        )

        # Verify extractor has manual mappings
        assert engine.extractor.manual_mappings == mappings


def test_semantic_search_engine_process_papers_no_pdfs():
    """Test process_papers raises error when no PDFs found."""
    import importlib.util
    from qdrant_client import QdrantClient

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SemanticSearchEngine = module.SemanticSearchEngine
    PDFProcessingError = module.PDFProcessingError

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_dir = Path(tmp_dir)
        client = QdrantClient(":memory:")

        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        # Should raise error when no PDFs
        with pytest.raises(PDFProcessingError, match="No PDF files found"):
            engine.process_papers()


def test_semantic_search_engine_process_papers_creates_collection(tmp_path):
    """Test process_papers creates Qdrant collection."""
    import importlib.util
    from qdrant_client import QdrantClient
    import fitz  # PyMuPDF

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SemanticSearchEngine = module.SemanticSearchEngine

    # Create a minimal test PDF with proper sections
    pdf_dir = Path(tmp_path) / "test_pdfs"
    pdf_dir.mkdir()

    # Create a simple test PDF with text content
    doc = fitz.open()
    page = doc.new_page()

    # Add text with clear sections
    text = """
Abstract
This is the abstract text.

Introduction
This is the introduction.

Methods
This describes the methodology.

Results
These are the results.

Discussion
This is the discussion.

References
1. Citation one
"""
    page.insert_text((50, 50), text)
    pdf_path = pdf_dir / "test_paper.pdf"
    doc.save(pdf_path)
    doc.close()

    client = QdrantClient(":memory:")
    engine = SemanticSearchEngine(
        pdf_dir=pdf_dir,
        qdrant_client=client,
        collection_name="test_collection"
    )

    # Process papers
    engine.process_papers()

    # Verify collection was created
    assert client.collection_exists(collection_name="test_collection")


def test_semantic_search_engine_process_papers_recreate_collection(tmp_path):
    """Test process_papers recreates collection when recreate=True."""
    import importlib.util
    from qdrant_client import QdrantClient
    import fitz

    project_root = Path(__file__).parent.parent
    module_path = project_root / "12_handson_sportscience_semantic_engine.py"

    spec = importlib.util.spec_from_file_location(
        "handson_sportscience_semantic_engine",
        module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    SemanticSearchEngine = module.SemanticSearchEngine

    # Create a minimal test PDF
    pdf_dir = Path(tmp_path) / "test_pdfs"
    pdf_dir.mkdir()

    doc = fitz.open()
    page = doc.new_page()
    text = """
Abstract
This is the abstract.

Introduction
Introduction text.

Methods
Methods text.

Results
Results text.
"""
    page.insert_text((50, 50), text)
    pdf_path = pdf_dir / "test_paper.pdf"
    doc.save(pdf_path)
    doc.close()

    client = QdrantClient(":memory:")
    engine = SemanticSearchEngine(
        pdf_dir=pdf_dir,
        qdrant_client=client,
        collection_name="test_collection"
    )

    # Process once
    engine.process_papers(recreate_collection=False)
    first_count = client.get_collection("test_collection").points_count

    # Process again with recreate=True
    engine.process_papers(recreate_collection=True)
    second_count = client.get_collection("test_collection").points_count

    # Count should be the same (collection was recreated)
    assert second_count == first_count
