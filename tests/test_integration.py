"""End-to-end integration tests for the sports science semantic search engine.

These tests exercise the complete pipeline:
    init → process/upload → search_and_compare → export_results

A tiny in-memory synthetic corpus is used so the suite stays fast and
deterministic without requiring a live Qdrant instance or network access.
Heavy end-to-end tests that touch the real embedding model are marked with
``@pytest.mark.integration`` so they can be skipped in CI if needed.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import fitz
import pytest
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sports_science_search import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    Paper,
    PaperChunk,
    SearchResult,
    VectorStore,
)
from sports_science_search.search_engine import SemanticSearchEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_pdf(directory: Path, filename: str = "test.pdf") -> Path:
    """Create a minimal valid PDF with clearly labelled sections in *directory*.

    The PDF contains the six canonical section headings so the auto-detector
    finds enough sections to proceed without a manual mapping.
    """
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "Abstract\n"
        "This study investigates GPS tracking accuracy in soccer players "
        "during high-intensity training sessions.\n\n"
        "Introduction\n"
        "Wearable technology enables continuous load monitoring in team sports. "
        "GPS and IMU sensors provide complementary data streams.\n\n"
        "Methods\n"
        "Twenty soccer players performed standardised drills. Data were "
        "collected at 10 Hz via GPS and 100 Hz via accelerometer.\n\n"
        "Results\n"
        "Mean positional error was 2.5 m. Acceleration detection sensitivity "
        "reached 92 % compared to the optical reference system.\n\n"
        "Discussion\n"
        "GPS-IMU fusion improved accuracy. Load metrics correlated strongly "
        "with subjective fatigue scores.\n\n"
        "References\n"
        "1. Aughey RJ (2011) Applications of GPS technologies.\n"
    )
    page.insert_text((50, 50), text)
    pdf_path = directory / filename
    doc.save(pdf_path)
    doc.close()
    return pdf_path


def _make_engine(
    pdf_dir: Path,
    collection_name: str = "test_integration",
) -> SemanticSearchEngine:
    """Return an initialised SemanticSearchEngine backed by an in-memory client."""
    client = QdrantClient(":memory:")
    return SemanticSearchEngine(
        pdf_dir=pdf_dir,
        qdrant_client=client,
        collection_name=collection_name,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_engine(tmp_path_factory):
    """Module-scoped fixture: one engine shared across tests in this module.

    The collection is populated once, making the suite significantly faster
    since the SentenceTransformer model is loaded only once per module.
    """
    pdf_dir = tmp_path_factory.mktemp("pdfs")
    _make_minimal_pdf(pdf_dir, "paper_a.pdf")
    _make_minimal_pdf(pdf_dir, "paper_b.pdf")

    engine = _make_engine(pdf_dir, collection_name="test_integration_shared")
    engine.process_papers(recreate_collection=True)
    return engine


# ---------------------------------------------------------------------------
# Integration: init → process → search → export
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEndToEndPipeline:
    """Full pipeline integration tests using a tiny synthetic corpus."""

    def test_engine_initialises_without_error(self, tmp_path: Path) -> None:
        """SemanticSearchEngine initialises when pdf_dir exists and is a dir."""
        engine = _make_engine(tmp_path)
        assert engine.pdf_dir == tmp_path
        assert engine.vector_store is not None
        assert engine.encoder is not None

    def test_process_papers_populates_collection(self, tmp_path: Path) -> None:
        """process_papers uploads chunks so the collection has points."""
        _make_minimal_pdf(tmp_path)
        engine = _make_engine(tmp_path)
        engine.process_papers(recreate_collection=True)

        info = engine.vector_store.client.get_collection(
            collection_name=engine.vector_store.collection_name
        )
        assert info.points_count > 0

    def test_search_and_compare_returns_correct_shape(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """search_and_compare returns a dict with the three required top-level keys."""
        result = shared_engine.search_and_compare(
            "GPS tracking accuracy", limit=2, verbose=False
        )
        assert set(result.keys()) == {"query", "results_by_strategy", "analysis"}

    def test_search_and_compare_preserves_query(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """The returned dict echoes back the exact query string."""
        query = "load monitoring injury prevention"
        result = shared_engine.search_and_compare(query, limit=2, verbose=False)
        assert result["query"] == query

    def test_search_and_compare_searches_all_three_strategies(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """results_by_strategy contains entries for semantic, paragraph, and fixed."""
        result = shared_engine.search_and_compare(
            "wearable sensor IMU", limit=2, verbose=False
        )
        strategies = set(result["results_by_strategy"].keys())
        assert strategies == {"semantic", "paragraph", "fixed"}

    def test_search_and_compare_results_are_search_result_instances(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """Each item in results_by_strategy lists is a SearchResult dataclass."""
        result = shared_engine.search_and_compare(
            "acceleration detection", limit=2, verbose=False
        )
        for strategy, results in result["results_by_strategy"].items():
            for item in results:
                assert isinstance(item, SearchResult), (
                    f"Expected SearchResult for strategy '{strategy}', got {type(item)}"
                )

    def test_search_and_compare_analysis_has_required_keys(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """The analysis sub-dict contains all documented keys."""
        result = shared_engine.search_and_compare(
            "GPS accuracy", limit=2, verbose=False
        )
        analysis = result["analysis"]
        required = {
            "chunk_count",
            "avg_chunk_size",
            "section_distribution",
            "top_strategy",
            "avg_scores",
        }
        assert required.issubset(set(analysis.keys())), (
            f"Missing keys: {required - set(analysis.keys())}"
        )

    def test_search_and_compare_top_strategy_is_valid(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """top_strategy must be one of the three known strategy names."""
        result = shared_engine.search_and_compare(
            "training intensity", limit=2, verbose=False
        )
        assert result["analysis"]["top_strategy"] in {
            "semantic", "paragraph", "fixed"
        }

    def test_search_and_compare_avg_scores_non_negative(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """Average similarity scores must be non-negative floats."""
        result = shared_engine.search_and_compare(
            "fatigue load", limit=2, verbose=False
        )
        for strategy, score in result["analysis"]["avg_scores"].items():
            assert isinstance(score, float), f"Score for '{strategy}' is not float"
            assert score >= 0.0, f"Negative score for strategy '{strategy}': {score}"

    def test_export_results_writes_valid_json_file(
        self, shared_engine: SemanticSearchEngine, tmp_path: Path
    ) -> None:
        """export_results writes a JSON file that can be parsed back."""
        result = shared_engine.search_and_compare(
            "GPS tracking", limit=2, verbose=False
        )
        output_path = tmp_path / "results.json"
        shared_engine.export_results(result, output_path)

        assert output_path.exists(), "Output file was not created"
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_export_results_json_preserves_query(
        self, shared_engine: SemanticSearchEngine, tmp_path: Path
    ) -> None:
        """The exported JSON preserves the original query string."""
        query = "IMU sensor soccer"
        result = shared_engine.search_and_compare(query, limit=2, verbose=False)
        output_path = tmp_path / "results_query.json"
        shared_engine.export_results(result, output_path)

        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["query"] == query

    def test_export_results_json_has_all_strategies(
        self, shared_engine: SemanticSearchEngine, tmp_path: Path
    ) -> None:
        """The exported JSON contains results for all three chunking strategies."""
        result = shared_engine.search_and_compare(
            "acceleration", limit=2, verbose=False
        )
        output_path = tmp_path / "results_strategies.json"
        shared_engine.export_results(result, output_path)

        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert "semantic" in data["results_by_strategy"]
        assert "paragraph" in data["results_by_strategy"]
        assert "fixed" in data["results_by_strategy"]

    def test_export_results_json_results_are_dicts_with_required_fields(
        self, shared_engine: SemanticSearchEngine, tmp_path: Path
    ) -> None:
        """Each result entry in the exported JSON is a dict with all SearchResult fields."""
        result = shared_engine.search_and_compare(
            "optical tracking", limit=3, verbose=False
        )
        output_path = tmp_path / "results_fields.json"
        shared_engine.export_results(result, output_path)

        required_fields = {
            "paper_filename",
            "paper_title",
            "section",
            "chunk_text",
            "score",
            "page_number",
            "metadata",
            "chunk_strategy",
        }
        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)

        for strategy, results_list in data["results_by_strategy"].items():
            for entry in results_list:
                assert isinstance(entry, dict), (
                    f"Entry for strategy '{strategy}' is not a dict"
                )
                missing = required_fields - set(entry.keys())
                assert not missing, (
                    f"Fields missing from '{strategy}' result: {missing}"
                )

    def test_export_results_json_preserves_analysis(
        self, shared_engine: SemanticSearchEngine, tmp_path: Path
    ) -> None:
        """The analysis dict round-trips correctly through JSON export."""
        result = shared_engine.search_and_compare(
            "training load", limit=2, verbose=False
        )
        output_path = tmp_path / "results_analysis.json"
        shared_engine.export_results(result, output_path)

        with open(output_path, encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["analysis"] == result["analysis"]

    def test_export_results_is_pretty_formatted(
        self, shared_engine: SemanticSearchEngine, tmp_path: Path
    ) -> None:
        """The exported JSON file uses indentation (pretty-print)."""
        result = shared_engine.search_and_compare(
            "positional error", limit=2, verbose=False
        )
        output_path = tmp_path / "results_pretty.json"
        shared_engine.export_results(result, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "\n" in content
        assert "  " in content  # Two-space indentation

    def test_full_pipeline_recreate_collection(self, tmp_path: Path) -> None:
        """Calling process_papers twice with recreate_collection=True resets data."""
        _make_minimal_pdf(tmp_path)
        engine = _make_engine(tmp_path)

        engine.process_papers(recreate_collection=False)
        first_count = engine.vector_store.client.get_collection(
            collection_name=engine.vector_store.collection_name
        ).points_count

        engine.process_papers(recreate_collection=True)
        second_count = engine.vector_store.client.get_collection(
            collection_name=engine.vector_store.collection_name
        ).points_count

        # After recreation the count should equal the first run
        # (same corpus, same number of chunks)
        assert second_count == first_count

    def test_analyze_chunking_effectiveness_returns_required_keys(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """analyze_chunking_effectiveness returns a dict with documented keys."""
        stats = shared_engine.analyze_chunking_effectiveness()
        required = {
            "total_points",
            "chunks_by_strategy",
            "chunks_by_section",
            "unique_papers",
        }
        assert required.issubset(set(stats.keys()))

    def test_analyze_chunking_effectiveness_counts_all_strategies(
        self, shared_engine: SemanticSearchEngine
    ) -> None:
        """chunks_by_strategy includes counts for all three strategies."""
        stats = shared_engine.analyze_chunking_effectiveness()
        cbs = stats["chunks_by_strategy"]
        assert "semantic" in cbs
        assert "paragraph" in cbs
        assert "fixed" in cbs
        assert all(count > 0 for count in cbs.values())


# ---------------------------------------------------------------------------
# VectorStore-level integration (lighter — no full engine init overhead)
# ---------------------------------------------------------------------------

class TestVectorStoreIntegration:
    """Integration tests at the VectorStore level using the real SentenceTransformer.

    These tests bypass SemanticSearchEngine init overhead and test the upload →
    search round-trip directly on a tiny synthetic corpus.
    """

    @pytest.fixture(scope="class")
    def store_and_encoder(self, tmp_path_factory):
        """Shared VectorStore + SentenceTransformer for the class."""
        client = QdrantClient(":memory:")
        store = VectorStore(client, collection_name="vs_integration_test")
        store.create_collection(dimension=EMBEDDING_DIMENSION, recreate=True)
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        return store, encoder

    @pytest.fixture(scope="class")
    def populated_store(self, store_and_encoder):
        """Upload a small set of synthetic chunks once per class."""
        store, encoder = store_and_encoder
        chunks = [
            PaperChunk(
                paper_filename="paper_x.pdf",
                paper_title="GPS Accuracy in Soccer",
                section="Methods",
                chunk_text=(
                    "GPS receivers operating at 10 Hz were attached to player "
                    "vests. Positional data were filtered with a Butterworth filter."
                ),
                chunk_index=0,
                page_number=2,
                chunk_strategy=strategy,
                metadata={"year": 2021, "authors": "Smith et al.", "DOI": None, "journal": "IJSPP"},
            )
            for strategy in ("semantic", "paragraph", "fixed")
        ] + [
            PaperChunk(
                paper_filename="paper_y.pdf",
                paper_title="Load Monitoring and Injury Prevention",
                section="Results",
                chunk_text=(
                    "Accelerometer-derived Player Load correlated significantly "
                    "with injury incidence rate (r=0.72, p<0.001)."
                ),
                chunk_index=1,
                page_number=4,
                chunk_strategy=strategy,
                metadata={"year": 2022, "authors": "Jones et al.", "DOI": None, "journal": "BJSM"},
            )
            for strategy in ("semantic", "paragraph", "fixed")
        ]
        store.upload_chunks(chunks, encoder)
        return store, encoder

    def test_upload_creates_points(self, populated_store) -> None:
        store, _ = populated_store
        info = store.client.get_collection(store.collection_name)
        assert info.points_count > 0

    def test_search_semantic_returns_results(self, populated_store) -> None:
        store, encoder = populated_store
        results = store.search(
            query="GPS positional accuracy",
            encoder=encoder,
            strategy="semantic",
            limit=3,
        )
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_paragraph_returns_results(self, populated_store) -> None:
        store, encoder = populated_store
        results = store.search(
            query="injury prevention",
            encoder=encoder,
            strategy="paragraph",
            limit=3,
        )
        assert len(results) > 0

    def test_search_fixed_returns_results(self, populated_store) -> None:
        store, encoder = populated_store
        results = store.search(
            query="player load monitoring",
            encoder=encoder,
            strategy="fixed",
            limit=3,
        )
        assert len(results) > 0

    def test_search_respects_limit(self, populated_store) -> None:
        store, encoder = populated_store
        for limit in (1, 2):
            results = store.search(
                query="GPS",
                encoder=encoder,
                strategy="semantic",
                limit=limit,
            )
            assert len(results) <= limit

    def test_search_result_fields_are_populated(self, populated_store) -> None:
        store, encoder = populated_store
        results = store.search(
            query="Butterworth filter",
            encoder=encoder,
            strategy="semantic",
            limit=1,
        )
        assert results
        r = results[0]
        assert r.paper_filename
        assert r.paper_title
        assert r.section
        assert r.chunk_text
        assert 0.0 <= r.score <= 1.0
        assert isinstance(r.page_number, int)
        assert isinstance(r.metadata, dict)
        assert r.chunk_strategy == "semantic"

    def test_search_section_filter(self, populated_store) -> None:
        store, encoder = populated_store
        results = store.search(
            query="GPS",
            encoder=encoder,
            strategy="semantic",
            limit=5,
            section_filter="Methods",
        )
        assert all(r.section == "Methods" for r in results)

    def test_get_collection_stats_returns_all_keys(self, populated_store) -> None:
        store, _ = populated_store
        stats = store.get_collection_stats()
        required = {"total_points", "chunks_by_strategy", "chunks_by_section", "unique_papers"}
        assert required.issubset(set(stats.keys()))

    def test_get_collection_stats_counts_strategies(self, populated_store) -> None:
        store, _ = populated_store
        stats = store.get_collection_stats()
        cbs = stats["chunks_by_strategy"]
        assert "semantic" in cbs
        assert "paragraph" in cbs
        assert "fixed" in cbs

    def test_get_collection_stats_unique_papers(self, populated_store) -> None:
        store, _ = populated_store
        stats = store.get_collection_stats()
        # Two distinct paper filenames were uploaded
        assert stats["unique_papers"] == 2
