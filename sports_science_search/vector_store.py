"""Qdrant vector database operations."""

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from sports_science_search.constants import EMBEDDING_DIMENSION
from sports_science_search.exceptions import PDFProcessingError, VectorUploadError
from sports_science_search.models import PaperChunk, SearchResult


class VectorStore:
    """
    Manages Qdrant vector database operations.

    Supports collection creation, chunk uploading, semantic search across
    multiple chunking strategies, and collection statistics.
    """

    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Initialize vector store.

        Args:
            client: Initialized QdrantClient (stateful, maintains connection)
            collection_name: Name of collection to create/use

        Note:
            VectorStore maintains a reference to the client and performs
            stateful operations on the Qdrant database.
        """
        self.client = client
        self.collection_name = collection_name

    def create_collection(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        recreate: bool = False
    ) -> None:
        """
        Create Qdrant collection with three named vectors and payload indexes.

        Args:
            dimension: Vector dimension (384 for all-MiniLM-L6-v2)
            recreate: If True, delete existing collection first

        Raises:
            ValueError: If dimension is not positive
            VectorUploadError: If collection creation or indexing fails
        """
        # Input validation
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        try:
            if recreate and self.client.collection_exists(
                collection_name=self.collection_name
            ):
                logging.info(f"Deleting existing collection '{self.collection_name}'")
                self.client.delete_collection(collection_name=self.collection_name)

            if not self.client.collection_exists(collection_name=self.collection_name):
                logging.info(
                    f"Creating collection '{self.collection_name}' with dimension {dimension}"
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "semantic": models.VectorParams(
                            size=dimension, distance=models.Distance.COSINE
                        ),
                        "paragraph": models.VectorParams(
                            size=dimension, distance=models.Distance.COSINE
                        ),
                        "fixed": models.VectorParams(
                            size=dimension, distance=models.Distance.COSINE
                        ),
                    },
                )

                logging.info(f"Creating payload indexes for '{self.collection_name}'")
                # Create payload indexes for filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="section",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="paper_filename",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="year",
                    field_schema=models.PayloadSchemaType.INTEGER,
                )

                logging.info(
                    f"Collection '{self.collection_name}' created successfully"
                )
            else:
                logging.info(
                    f"Collection '{self.collection_name}' already exists, skipping creation"
                )
        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logging.error(f"Failed to create collection '{self.collection_name}': {e}")
            raise VectorUploadError(
                f"Failed to create collection '{self.collection_name}': {str(e)}"
            ) from e

    def upload_chunks(
        self,
        chunks: List[PaperChunk],
        encoder: SentenceTransformer,
        batch_size: int = 100
    ) -> None:
        """
        Vectorize and upload chunks to Qdrant.

        Args:
            chunks: List of PaperChunk objects to vectorize and upload
            encoder: SentenceTransformer model for encoding chunks
            batch_size: Number of chunks to upload per batch (default: 100)

        Raises:
            VectorUploadError: If upload fails
        """
        if not chunks:
            return

        def _flush(batch: List[models.PointStruct]) -> None:
            """Upload one batch of points, wrapping failures."""
            try:
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=batch,
                )
            except Exception as e:
                raise VectorUploadError(f"Failed to upload batch: {str(e)}") from e

        points: List[models.PointStruct] = []

        # Process in batches: encode each batch in one vectorized call (much
        # faster than per-chunk encoding) and assign a unique UUID to every
        # point so repeated upload_chunks calls never overwrite each other.
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start:start + batch_size]

            # Batch-encode all texts in this slice at once.
            texts = [chunk.chunk_text for chunk in batch_chunks]
            vectors = encoder.encode(texts)

            for chunk, vector in zip(batch_chunks, vectors):
                payload = {
                    "paper_filename": chunk.paper_filename,
                    "paper_title": chunk.paper_title,
                    "section": chunk.section,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "chunk_strategy": chunk.chunk_strategy,
                    "authors": chunk.metadata.get("authors"),
                    "year": chunk.metadata.get("year"),
                    "DOI": chunk.metadata.get("DOI"),
                    "journal": chunk.metadata.get("journal"),
                }
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={chunk.chunk_strategy: vector.tolist()},
                        payload=payload,
                    )
                )

            _flush(points)
            points = []

    def search(
        self,
        query: str,
        encoder: SentenceTransformer,
        strategy: str,
        limit: int = 3,
        section_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        paper_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search using specific chunking strategy with optional filters.

        Args:
            query: Search query string
            encoder: SentenceTransformer model for encoding query
            strategy: Chunking strategy to search ("semantic", "paragraph", "fixed")
            limit: Maximum number of results to return (default: 3)
            section_filter: Filter by section name (e.g., "Methods")
            year_filter: Filter by publication year
            paper_filter: Filter by paper filename

        Returns:
            List of SearchResult objects ordered by relevance.
            Returns empty list if no results match the query and filters.

        Raises:
            ValueError: If query is empty, strategy is invalid, or limit is invalid
            PDFProcessingError: If query encoding fails
            VectorUploadError: If Qdrant search operation fails
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        VALID_STRATEGIES = {"semantic", "paragraph", "fixed"}
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {VALID_STRATEGIES}, got '{strategy}'"
            )

        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        if limit > 1000:
            raise ValueError(f"limit must be <= 1000, got {limit}")

        # Validate filter string lengths to prevent abuse
        if section_filter and len(section_filter) > 200:
            raise ValueError("section_filter too long (max 200 chars)")

        if paper_filter and len(paper_filter) > 500:
            raise ValueError("paper_filter too long (max 500 chars)")

        logging.info(
            f"Searching collection '{self.collection_name}' with strategy='{strategy}', "
            f"limit={limit}, filters=(section={section_filter}, year={year_filter}, "
            f"paper={paper_filter})"
        )

        # Encode query
        try:
            query_vector: List[float] = encoder.encode(query).tolist()
        except Exception as e:
            raise PDFProcessingError(f"Failed to encode query: {str(e)}") from e

        # Build filter conditions
        filter_conditions = []

        if section_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="section",
                    match=models.MatchValue(value=section_filter)
                )
            )

        if year_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="year",
                    match=models.MatchValue(value=year_filter)
                )
            )

        if paper_filter:
            filter_conditions.append(
                models.FieldCondition(
                    key="paper_filename",
                    match=models.MatchValue(value=paper_filter)
                )
            )

        # Create filter if conditions exist
        search_filter = None
        if filter_conditions:
            search_filter = models.Filter(must=filter_conditions)

        # Search Qdrant
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=strategy,  # Named vector to query (semantic/paragraph/fixed)
                query_filter=search_filter,
                limit=limit
            )
        except Exception as e:
            raise VectorUploadError(
                f"Search failed for strategy '{strategy}': {str(e)}"
            ) from e

        # Convert to SearchResult objects
        search_results = []
        for point in results.points:
            # Validate required fields exist
            required_fields = ["paper_filename", "paper_title", "section", "chunk_text", "page_number"]
            missing = [f for f in required_fields if f not in point.payload]
            if missing:
                logging.warning(f"Skipping point {point.id}: missing fields {missing}")
                continue

            # Create SearchResult with flat structure
            result = SearchResult(
                paper_filename=point.payload["paper_filename"],
                paper_title=point.payload["paper_title"],
                section=point.payload["section"],
                chunk_text=point.payload["chunk_text"],
                score=point.score,
                page_number=point.payload["page_number"],
                metadata={
                    "authors": point.payload.get("authors"),
                    "year": point.payload.get("year"),
                    "DOI": point.payload.get("DOI"),
                    "journal": point.payload.get("journal"),
                },
                chunk_strategy=strategy
            )
            search_results.append(result)

        logging.info(f"Search returned {len(search_results)} results")
        return search_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return collection statistics.

        Returns:
            Dict with total_points, chunks_by_strategy, chunks_by_section, unique_papers

        Raises:
            VectorUploadError: If Qdrant scroll operation fails
        """
        # Calculate stats
        chunks_by_strategy = {}
        chunks_by_section = {}
        unique_papers = set()
        total_points = 0
        offset = None

        try:
            # Scroll through all points (may require pagination)
            while True:
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                    offset=offset
                )

                points = scroll_result[0]
                offset = scroll_result[1]

                if not points:
                    break

                total_points += len(points)

                for point in points:
                    strategy = point.payload.get("chunk_strategy")
                    if strategy:
                        chunks_by_strategy[strategy] = chunks_by_strategy.get(strategy, 0) + 1

                    section = point.payload.get("section")
                    if section:
                        chunks_by_section[section] = chunks_by_section.get(section, 0) + 1

                    paper = point.payload.get("paper_filename")
                    if paper:
                        unique_papers.add(paper)

                # If no next offset, we've reached the end
                if offset is None:
                    break

        except Exception as e:
            raise VectorUploadError(
                f"Failed to get collection stats for '{self.collection_name}': {str(e)}"
            ) from e

        return {
            "total_points": total_points,
            "chunks_by_strategy": chunks_by_strategy,
            "chunks_by_section": chunks_by_section,
            "unique_papers": len(unique_papers)
        }
