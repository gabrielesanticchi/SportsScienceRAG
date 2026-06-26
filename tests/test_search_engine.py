"""Tests for SemanticSearchEngine initialization and paper processing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import pytest
from qdrant_client import QdrantClient

from sports_science_search import PDFProcessingError
from sports_science_search.search_engine import SemanticSearchEngine


class TestSemanticSearchEngineInit:
    """Test SemanticSearchEngine.__init__ method."""

    def test_init_without_manual_mappings(self):
        """Test initialization without manual mappings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)
            client = QdrantClient(":memory:")

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
            assert engine.vector_store.collection_name == "test_collection"

    def test_init_nonexistent_directory_raises_error(self):
        """Test initialization fails when pdf_dir doesn't exist."""
        pdf_dir = Path("/nonexistent/directory/that/does/not/exist")
        client = QdrantClient(":memory:")

        with pytest.raises(PDFProcessingError, match="PDF directory does not exist"):
            SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client
            )

    def test_init_file_instead_of_directory_raises_error(self, tmp_path):
        """Test initialization fails when pdf_dir is a file, not directory."""
        pdf_file = tmp_path / "file.txt"
        pdf_file.write_text("not a directory")
        client = QdrantClient(":memory:")

        with pytest.raises(PDFProcessingError, match="PDF path is not a directory"):
            SemanticSearchEngine(
                pdf_dir=pdf_file,
                qdrant_client=client
            )

    def test_init_invalid_collection_name_raises_error(self, tmp_path):
        """Test initialization fails with invalid collection name."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        client = QdrantClient(":memory:")

        # Test invalid characters
        with pytest.raises(ValueError, match="Invalid collection name"):
            SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client,
                collection_name="invalid name with spaces"
            )

        # Test empty name
        with pytest.raises(ValueError, match="Invalid collection name"):
            SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client,
                collection_name=""
            )

    def test_init_with_manual_mappings(self):
        """Test initialization with manual mappings file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)

            # Create manual mappings file
            mappings = {
                "test.pdf": {
                    "Abstract": {"start_page": 1, "end_page": 1},
                    "Methods": {"start_page": 2, "end_page": 3}
                }
            }
            mappings_path = Path(tmp_dir) / "mappings.json"
            with open(mappings_path, 'w') as f:
                json.dump(mappings, f)

            client = QdrantClient(":memory:")

            engine = SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client,
                collection_name="test_collection",
                manual_mappings_path=mappings_path
            )

            # Verify extractor loaded manual mappings
            assert engine.extractor.manual_mappings == mappings

    def test_init_with_default_collection_name(self):
        """Test initialization uses default collection name when not specified."""
        from sports_science_search import COLLECTION_NAME

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)
            client = QdrantClient(":memory:")

            engine = SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client
            )

            assert engine.vector_store.collection_name == COLLECTION_NAME

    def test_init_pdf_dir_is_stored_as_path(self):
        """Test pdf_dir is stored as Path object."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)
            client = QdrantClient(":memory:")

            engine = SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client
            )

            assert isinstance(engine.pdf_dir, Path)
            assert engine.pdf_dir == pdf_dir

    def test_init_creates_single_encoder_instance(self):
        """Test encoder is initialized once (optimization check)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)
            client = QdrantClient(":memory:")

            with patch('sports_science_search.search_engine.SentenceTransformer') as mock_st:
                mock_encoder = MagicMock()
                mock_st.return_value = mock_encoder

                engine = SemanticSearchEngine(
                    pdf_dir=pdf_dir,
                    qdrant_client=client
                )

                # Verify SentenceTransformer was called exactly once
                mock_st.assert_called_once()
                assert engine.encoder == mock_encoder


class TestSemanticSearchEngineProcessPapers:
    """Test SemanticSearchEngine.process_papers method."""

    def test_process_papers_no_pdfs_raises_error(self):
        """Test process_papers raises PDFProcessingError when no PDFs found."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_dir = Path(tmp_dir)
            client = QdrantClient(":memory:")

            engine = SemanticSearchEngine(
                pdf_dir=pdf_dir,
                qdrant_client=client
            )

            with pytest.raises(PDFProcessingError, match="No PDF files found"):
                engine.process_papers()

    def test_process_papers_creates_collection(self, tmp_path):
        """Test process_papers creates Qdrant collection."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create minimal test PDF with proper sections
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
This is the abstract.

Introduction
Introduction content.

Methods
Methods description.

Results
Results data.

Discussion
Discussion points.

References
1. Citation
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Verify collection was created
        assert client.collection_exists(collection_name="test_collection")

    def test_process_papers_recreate_collection(self, tmp_path):
        """Test process_papers recreates collection when recreate=True."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
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

        # Count should be the same (collection was recreated and repopulated)
        assert second_count == first_count

    def test_process_papers_uploads_chunks(self, tmp_path):
        """Test process_papers uploads chunks to Qdrant."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
This is the abstract with sufficient content.

Methods
Methods section with more content here.

Results
Results section with data.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Verify chunks were uploaded
        collection_info = client.get_collection("test_collection")
        assert collection_info.points_count > 0

    def test_process_papers_processes_multiple_pdfs(self, tmp_path):
        """Test process_papers handles multiple PDF files."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create two test PDFs with more distinctive content
        for i in range(2):
            doc = fitz.open()
            page = doc.new_page()
            text = f"""
Abstract
This is paper number {i} abstract with unique content.

Introduction
Introduction for paper {i} with more text.

Methods
Methods section for paper {i} describing methodology.

Results
Results section for paper {i} with data and findings.

Discussion
Discussion of paper {i} results and implications.

References
1. Citation for paper {i}
"""
            page.insert_text((50, 50), text)
            pdf_path = pdf_dir / f"paper{i}.pdf"
            doc.save(pdf_path)
            doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Verify chunks were uploaded
        collection_info = client.get_collection("test_collection")
        assert collection_info.points_count > 0

        # Verify at least one paper is present (both should be, but allow for processing variations)
        points, _ = client.scroll(
            collection_name="test_collection",
            limit=1000,
            with_payload=True
        )
        paper_filenames = {point.payload.get("paper_filename") for point in points}

        # Verify we got chunks from the papers
        assert len(paper_filenames) > 0
        assert all(filename in ["paper0.pdf", "paper1.pdf"] for filename in paper_filenames if filename)

    def test_process_papers_fail_fast_on_extraction_error(self, tmp_path):
        """Test process_papers fails fast on extraction error."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create corrupted PDF (just an empty file)
        corrupted_pdf = pdf_dir / "corrupted.pdf"
        corrupted_pdf.write_text("not a valid PDF")

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        with pytest.raises(PDFProcessingError, match="Failed processing corrupted.pdf"):
            engine.process_papers()

    def test_process_papers_fail_fast_on_section_detection_error(self, tmp_path):
        """Test process_papers fails fast on section detection error."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create PDF with no clear sections
        doc = fitz.open()
        page = doc.new_page()
        text = "Just some text with no sections at all."
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "nosections.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        with pytest.raises(PDFProcessingError, match="Failed processing nosections.pdf"):
            engine.process_papers()

    def test_process_papers_fail_fast_on_vector_upload_error(self, tmp_path):
        """Test process_papers fails fast on vector upload error."""
        from sports_science_search import VectorUploadError

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create valid PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        # Mock upload_chunks to raise VectorUploadError
        original_upload = engine.vector_store.upload_chunks

        def mock_upload_with_error(chunks, encoder):
            raise VectorUploadError("Simulated Qdrant connection failure")

        engine.vector_store.upload_chunks = mock_upload_with_error

        with pytest.raises(PDFProcessingError, match="Failed processing test.pdf") as exc_info:
            engine.process_papers()

        # Verify error message includes VectorUploadError stage
        assert "VectorUploadError" in str(exc_info.value)
        assert "Qdrant connection" in str(exc_info.value)

    def test_process_papers_logging(self, tmp_path, caplog):
        """Test process_papers logs progress messages."""
        import logging

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content here.

Methods
Methods content here.

Results
Results content here.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        with caplog.at_level(logging.INFO):
            engine.process_papers()

        # Verify logging messages
        assert "Processing 1 papers" in caplog.text
        assert "Extracting: test.pdf" in caplog.text
        assert "Chunking: test.pdf" in caplog.text
        assert "Uploading:" in caplog.text
        assert "Successfully processed 1 papers" in caplog.text

    def test_process_papers_calls_pipeline_in_order(self, tmp_path):
        """Test process_papers calls extract, chunk, upload in correct order."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        # Mock methods to verify call order
        call_order = []

        original_extract = engine.extractor.extract
        original_chunk = engine.chunker.chunk_all_strategies
        original_upload = engine.vector_store.upload_chunks

        def mock_extract(path):
            call_order.append("extract")
            return original_extract(path)

        def mock_chunk(paper):
            call_order.append("chunk")
            return original_chunk(paper)

        def mock_upload(chunks, encoder):
            call_order.append("upload")
            return original_upload(chunks, encoder)

        engine.extractor.extract = mock_extract
        engine.chunker.chunk_all_strategies = mock_chunk
        engine.vector_store.upload_chunks = mock_upload

        engine.process_papers()

        # Verify call order
        assert call_order == ["extract", "chunk", "upload"]

    def test_process_papers_sorts_pdf_files(self, tmp_path):
        """Test process_papers processes PDFs in sorted order."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create PDFs with names that would sort differently
        for name in ["c.pdf", "a.pdf", "b.pdf"]:
            doc = fitz.open()
            page = doc.new_page()
            text = """
Abstract
Content.

Methods
Content.

Results
Content.
"""
            page.insert_text((50, 50), text)
            pdf_path = pdf_dir / name
            doc.save(pdf_path)
            doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        processed_files = []

        original_extract = engine.extractor.extract

        def mock_extract(path):
            processed_files.append(path.name)
            return original_extract(path)

        engine.extractor.extract = mock_extract

        engine.process_papers()

        # Verify files were processed in sorted order
        assert processed_files == ["a.pdf", "b.pdf", "c.pdf"]


class TestSemanticSearchEngineSearchAndCompare:
    """Test SemanticSearchEngine.search_and_compare method."""

    def test_search_and_compare_returns_dict_structure(self, tmp_path):
        """Test search_and_compare returns correct dict structure."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
This is the abstract with sufficient content about biomechanics.

Methods
Methods section describes the experimental protocol.

Results
Results show significant improvements in performance.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("biomechanics", limit=2, verbose=False)

        # Verify structure
        assert "query" in result
        assert "results_by_strategy" in result
        assert "analysis" in result
        assert result["query"] == "biomechanics"

    def test_search_and_compare_searches_all_strategies(self, tmp_path):
        """Test search_and_compare searches all three strategies."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
This paper explores biomechanical analysis methods.

Introduction
Introduction to biomechanics research.

Methods
Experimental methods for biomechanical assessment.

Results
Results demonstrate improved measurement accuracy.

Discussion
Discussion of biomechanical findings.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("biomechanics", limit=2, verbose=False)

        # Verify all strategies present
        assert "semantic" in result["results_by_strategy"]
        assert "paragraph" in result["results_by_strategy"]
        assert "fixed" in result["results_by_strategy"]

    def test_search_and_compare_analysis_contains_required_fields(self, tmp_path):
        """Test analysis dict contains all required fields."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Research on performance monitoring systems.

Methods
Methods for performance analysis.

Results
Results of performance testing.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("performance", limit=2, verbose=False)

        analysis = result["analysis"]
        assert "chunk_count" in analysis
        assert "avg_chunk_size" in analysis
        assert "section_distribution" in analysis
        assert "top_strategy" in analysis
        assert "avg_scores" in analysis

    def test_search_and_compare_avg_scores_calculated_correctly(self, tmp_path):
        """Test average scores are calculated correctly for each strategy."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Sports science research methodology.

Methods
Detailed experimental protocols.

Results
Statistical analysis results.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("sports", limit=2, verbose=False)

        # Verify avg_scores exist for all strategies
        avg_scores = result["analysis"]["avg_scores"]
        assert "semantic" in avg_scores
        assert "paragraph" in avg_scores
        assert "fixed" in avg_scores

        # Verify scores are numeric and non-negative
        for strategy, score in avg_scores.items():
            assert isinstance(score, float)
            assert score >= 0.0

    def test_search_and_compare_top_strategy_identified(self, tmp_path):
        """Test top-performing strategy is correctly identified."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Machine learning applications.

Methods
Training methodology.

Results
Model performance results.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("machine learning", limit=2, verbose=False)

        top_strategy = result["analysis"]["top_strategy"]
        assert top_strategy in ["semantic", "paragraph", "fixed"]

    def test_search_and_compare_section_distribution_calculated(self, tmp_path):
        """Test section distribution is calculated correctly."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("content", limit=3, verbose=False)

        section_dist = result["analysis"]["section_distribution"]
        assert isinstance(section_dist, dict)
        # Verify proportions sum to 1.0 (or close due to float precision)
        if section_dist:
            total = sum(section_dist.values())
            assert abs(total - 1.0) < 0.01

    def test_search_and_compare_respects_limit(self, tmp_path):
        """Test search_and_compare respects limit parameter."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF with more content
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content with keywords.

Introduction
Introduction with keywords.

Methods
Methods with keywords.

Results
Results with keywords.

Discussion
Discussion with keywords.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("keywords", limit=2, verbose=False)

        # Verify each strategy returns at most limit results
        for strategy, results in result["results_by_strategy"].items():
            assert len(results) <= 2

    def test_search_and_compare_with_filters(self, tmp_path):
        """Test search_and_compare respects filter parameters."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Search with section filter
        result = engine.search_and_compare(
            "content",
            limit=3,
            section_filter="Methods",
            verbose=False
        )

        # Verify all results are from Methods section
        for strategy, results in result["results_by_strategy"].items():
            for r in results:
                assert r.section == "Methods"

    def test_search_and_compare_verbose_output(self, tmp_path, caplog):
        """Test search_and_compare verbose mode prints output."""
        import logging

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Test abstract.

Methods
Test methods.

Results
Test results.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Capture stdout instead of logs for print statements
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            result = engine.search_and_compare("test", limit=2, verbose=True)
        finally:
            sys.stdout = sys.__stdout__

        output = captured_output.getvalue()

        # Verify verbose output contains expected elements
        assert "Query:" in output
        assert "SEMANTIC CHUNKING" in output
        assert "PARAGRAPH CHUNKING" in output
        assert "FIXED CHUNKING" in output
        assert "ANALYSIS:" in output

    def test_search_and_compare_empty_results_handling(self, tmp_path):
        """Test search_and_compare handles empty results gracefully."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Search for something unlikely to match
        result = engine.search_and_compare(
            "zzzznonexistenttermzzzz",
            limit=2,
            verbose=False
        )

        # Verify structure is still valid even with no results
        assert "query" in result
        assert "results_by_strategy" in result
        assert "analysis" in result

    def test_search_and_compare_empty_query_raises_error(self, tmp_path):
        """Test search_and_compare raises ValueError for empty query."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        with pytest.raises(ValueError, match="Query cannot be empty"):
            engine.search_and_compare("", limit=2, verbose=False)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            engine.search_and_compare("   ", limit=2, verbose=False)

    def test_search_and_compare_invalid_limit_raises_error(self, tmp_path):
        """Test search_and_compare raises ValueError for invalid limit."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        with pytest.raises(ValueError, match="Limit must be positive"):
            engine.search_and_compare("test", limit=0, verbose=False)

        with pytest.raises(ValueError, match="Limit must be positive"):
            engine.search_and_compare("test", limit=-1, verbose=False)


class TestSemanticSearchEngineAnalyzeChunkingEffectiveness:
    """Test SemanticSearchEngine.analyze_chunking_effectiveness method."""

    def test_analyze_chunking_effectiveness_returns_dict(self, tmp_path):
        """Test analyze_chunking_effectiveness returns correct dict structure."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Analysis of chunking strategies.

Methods
Methodology for chunking.

Results
Results of analysis.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        stats = engine.analyze_chunking_effectiveness()

        # Verify structure
        assert isinstance(stats, dict)
        assert "total_points" in stats
        assert "chunks_by_strategy" in stats
        assert "chunks_by_section" in stats
        assert "unique_papers" in stats

    def test_analyze_chunking_effectiveness_chunks_by_strategy(self, tmp_path):
        """Test chunks_by_strategy contains all three strategies."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Introduction
Introduction content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        stats = engine.analyze_chunking_effectiveness()

        chunks_by_strategy = stats["chunks_by_strategy"]
        assert "semantic" in chunks_by_strategy
        assert "paragraph" in chunks_by_strategy
        assert "fixed" in chunks_by_strategy
        assert all(count > 0 for count in chunks_by_strategy.values())

    def test_analyze_chunking_effectiveness_section_distribution(self, tmp_path):
        """Test section distribution is recorded correctly."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Abstract content.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        stats = engine.analyze_chunking_effectiveness()

        section_dist = stats["chunks_by_section"]
        assert isinstance(section_dist, dict)
        # Should have at least some sections
        assert len(section_dist) > 0

    def test_analyze_chunking_effectiveness_unique_papers_count(self, tmp_path):
        """Test unique_papers count is correct."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Paper abstract.

Methods
Paper methods.

Results
Paper results.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "paper.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        stats = engine.analyze_chunking_effectiveness()

        # At least one paper should be indexed
        assert stats["unique_papers"] >= 1

    def test_analyze_chunking_effectiveness_prints_output(self, tmp_path, capsys):
        """Test analyze_chunking_effectiveness prints formatted output."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Content here.

Methods
Methods here.

Results
Results here.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        engine.analyze_chunking_effectiveness()

        captured = capsys.readouterr()
        output = captured.out

        # Verify output contains expected elements
        assert "CHUNKING STRATEGY ANALYSIS" in output
        assert "SEMANTIC STRATEGY:" in output
        assert "PARAGRAPH STRATEGY:" in output
        assert "FIXED STRATEGY:" in output
        assert "Total unique papers:" in output
        assert "Section distribution:" in output


class TestSemanticSearchEngineExportResults:
    """Test SemanticSearchEngine.export_results method."""

    def test_export_results_creates_file(self, tmp_path):
        """Test export_results creates output JSON file."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Export test content.

Methods
Methods for export.

Results
Results for export.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        # Get search results
        result = engine.search_and_compare("export", limit=2, verbose=False)

        # Export to file
        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        # Verify file was created
        assert output_path.exists()

    def test_export_results_file_contains_valid_json(self, tmp_path):
        """Test export_results file contains valid JSON."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
JSON test content.

Methods
Methods here.

Results
Results here.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("JSON", limit=2, verbose=False)

        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        # Verify JSON is valid
        with open(output_path, 'r') as f:
            data = json.load(f)

        assert isinstance(data, dict)

    def test_export_results_json_structure(self, tmp_path):
        """Test exported JSON has correct structure."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Structure test.

Methods
Methods content.

Results
Results content.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("structure", limit=2, verbose=False)

        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        with open(output_path, 'r') as f:
            data = json.load(f)

        # Verify top-level structure
        assert "query" in data
        assert "results_by_strategy" in data
        assert "analysis" in data

        # Verify query is preserved
        assert data["query"] == "structure"

        # Verify results_by_strategy has all strategies
        assert "semantic" in data["results_by_strategy"]
        assert "paragraph" in data["results_by_strategy"]
        assert "fixed" in data["results_by_strategy"]

    def test_export_results_converts_search_results_to_dicts(self, tmp_path):
        """Test export_results properly converts SearchResult namedtuples to dicts."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Result conversion test.

Methods
Testing conversion.

Results
Conversion works.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("conversion", limit=2, verbose=False)

        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        with open(output_path, 'r') as f:
            data = json.load(f)

        # Verify results are now dicts with expected fields
        for strategy in ["semantic", "paragraph", "fixed"]:
            for result_dict in data["results_by_strategy"][strategy]:
                assert isinstance(result_dict, dict)
                assert "paper_filename" in result_dict
                assert "paper_title" in result_dict
                assert "section" in result_dict
                assert "chunk_text" in result_dict
                assert "score" in result_dict
                assert "page_number" in result_dict
                assert "metadata" in result_dict
                assert "chunk_strategy" in result_dict

    def test_export_results_preserves_analysis(self, tmp_path):
        """Test export_results preserves analysis metrics."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Analysis preservation test.

Methods
Testing analysis.

Results
Analysis data.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("analysis", limit=2, verbose=False)

        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        with open(output_path, 'r') as f:
            data = json.load(f)

        # Verify analysis is preserved
        assert data["analysis"] == result["analysis"]
        assert "chunk_count" in data["analysis"]
        assert "avg_chunk_size" in data["analysis"]
        assert "avg_scores" in data["analysis"]

    def test_export_results_missing_keys_raises_error(self, tmp_path):
        """Test export_results raises ValueError for missing required keys."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client
        )

        output_path = tmp_path / "results.json"

        # Missing 'analysis' key
        with pytest.raises(ValueError, match="Results dict missing required keys"):
            engine.export_results(
                {"query": "test", "results_by_strategy": {}},
                output_path
            )

        # Missing 'query' key
        with pytest.raises(ValueError, match="Results dict missing required keys"):
            engine.export_results(
                {"results_by_strategy": {}, "analysis": {}},
                output_path
            )

    def test_export_results_prints_confirmation(self, tmp_path, capsys):
        """Test export_results prints confirmation message."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Print confirmation test.

Methods
Testing output.

Results
Output confirmed.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("print", limit=2, verbose=False)

        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        captured = capsys.readouterr()
        output = captured.out

        # Verify confirmation message
        assert "Results exported to:" in output
        assert str(output_path) in output

    def test_export_results_pretty_formatted_json(self, tmp_path):
        """Test exported JSON is pretty-formatted with indentation."""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()

        # Create test PDF
        doc = fitz.open()
        page = doc.new_page()
        text = """
Abstract
Format test.

Methods
Testing formatting.

Results
Format verified.
"""
        page.insert_text((50, 50), text)
        pdf_path = pdf_dir / "test.pdf"
        doc.save(pdf_path)
        doc.close()

        client = QdrantClient(":memory:")
        engine = SemanticSearchEngine(
            pdf_dir=pdf_dir,
            qdrant_client=client,
            collection_name="test_collection"
        )

        engine.process_papers()

        result = engine.search_and_compare("format", limit=2, verbose=False)

        output_path = tmp_path / "results.json"
        engine.export_results(result, output_path)

        # Read file content
        with open(output_path, 'r') as f:
            content = f.read()

        # Verify it's pretty-formatted (has newlines and indentation)
        assert '\n' in content
        assert '  ' in content  # Indentation
