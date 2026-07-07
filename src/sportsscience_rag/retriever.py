"""Chunk-level semantic retrieval over the Qdrant collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sportsscience_rag.qdrant_store import VECTOR_NAME


@dataclass(frozen=True)
class RetrievedChunk:
    """A single ranked chunk returned by a retrieval query.

    Attributes:
        rank: 1-indexed position in the returned ranking.
        score: Similarity score from Qdrant (cosine).
        source: Source identifier of the document (e.g., S3 URI).
        chunk_index: Sequence number of the chunk within its document.
        page_numbers: 1-indexed page numbers the chunk spans; empty if unknown.
        section_path: Hierarchical heading path of the chunk.
        text: Chunk content text.
    """

    rank: int
    score: float
    source: str
    chunk_index: int
    page_numbers: tuple[int, ...]
    section_path: str
    text: str


class Retriever:
    """Embeds a query and returns chunk-level ranked results from Qdrant.

    Attributes:
        _client: Qdrant client used to run the vector query.
        _embedder: Embedder exposing ``embed(list[str]) -> list[list[float]]``.
        _collection: Name of the collection to query.
    """

    def __init__(self, client: Any, embedder: Any, collection: str) -> None:
        """Initialize the retriever.

        Args:
            client: Qdrant client instance.
            embedder: Embedder with an ``embed`` method matching ``TextEmbedder``.
            collection: Target Qdrant collection name.
        """
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def search(self, query: str, limit: int = 10) -> list[RetrievedChunk]:
        """Embed ``query`` and return the top ``limit`` chunks, ranked.

        Args:
            query: Natural-language query string.
            limit: Maximum number of chunks to return.

        Returns:
            A list of ``RetrievedChunk`` ordered by descending score, with
            1-indexed ``rank``.
        """
        vector = self._embedder.embed([query])[0]
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=VECTOR_NAME,
            limit=limit,
            with_payload=True,
        )
        chunks: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    rank=rank,
                    score=float(point.score),
                    source=payload.get("source", ""),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    page_numbers=tuple(payload.get("page_numbers", ()) or ()),
                    section_path=payload.get("section_path", ""),
                    text=payload.get("text", ""),
                )
            )
        return chunks
