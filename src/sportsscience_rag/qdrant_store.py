"""Qdrant collection management and idempotent chunk upsert."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from sportsscience_rag.models import Chunk

NAMESPACE = uuid.UUID("6f9b1d2a-3c4e-5a6b-8c9d-0e1f2a3b4c5d")
VECTOR_NAME = "text_embedding"
_INDEXED_FIELDS = ("content_hash", "parser_version", "chunk_config_hash")


def point_id(content_hash: str, chunk_index: int) -> str:
    """Derive a deterministic Qdrant point ID from a content hash and chunk index.

    Using ``uuid5`` with a fixed namespace ensures that re-ingesting the same
    document chunk always produces the same point ID, making upserts
    idempotent.

    Args:
        content_hash: Hash of the source document's content.
        chunk_index: Sequence number of the chunk within the document.

    Returns:
        A deterministic UUID string suitable for use as a Qdrant point ID.
    """
    return str(uuid.uuid5(NAMESPACE, f"{content_hash}:{chunk_index}"))


class QdrantStore:
    """Manages a Qdrant collection and performs idempotent chunk upserts.

    Attributes:
        _client: Injected Qdrant client instance.
        _collection: Name of the target Qdrant collection.
        _dimension: Dimensionality of the stored embedding vectors.
    """

    def __init__(self, client: Any, collection: str, dimension: int = 384) -> None:
        """Initialize the store with a Qdrant client and target collection.

        Args:
            client: Qdrant client used to perform collection and point operations.
            collection: Name of the collection to manage.
            dimension: Dimensionality of the embedding vectors. Defaults to 384.
        """
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist.

        The collection is configured with a single named vector
        (``text_embedding``) using cosine distance. If the collection already
        exists, collection creation is a no-op.

        Regardless of whether the collection was just created or already
        existed, this also ensures a KEYWORD payload index exists on each
        field used by ``already_ingested``'s filter (``content_hash``,
        ``parser_version``, ``chunk_config_hash``). Qdrant Cloud rejects
        filtered ``count``/``search`` calls with HTTP 400 when the filtered
        fields lack a payload index, so the index must be created here.

        Returns:
            None.
        """
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    VECTOR_NAME: models.VectorParams(
                        size=self._dimension,
                        distance=models.Distance.COSINE,
                    )
                },
            )
        for field in _INDEXED_FIELDS:
            self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _match_filter(
        self, content_hash: str, parser_version: str, chunk_config_hash: str
    ) -> models.Filter:
        """Build a Qdrant payload filter matching an ingestion fingerprint.

        Args:
            content_hash: Hash of the source document's content.
            parser_version: Version identifier of the parser used to produce chunks.
            chunk_config_hash: Hash of the chunking configuration used.

        Returns:
            A ``models.Filter`` requiring all three payload fields to match.
        """
        return models.Filter(
            must=[
                models.FieldCondition(key="content_hash", match=models.MatchValue(value=content_hash)),
                models.FieldCondition(key="parser_version", match=models.MatchValue(value=parser_version)),
                models.FieldCondition(key="chunk_config_hash", match=models.MatchValue(value=chunk_config_hash)),
            ]
        )

    def already_ingested(
        self, content_hash: str, parser_version: str, chunk_config_hash: str
    ) -> bool:
        """Check whether points matching this ingestion fingerprint already exist.

        Args:
            content_hash: Hash of the source document's content.
            parser_version: Version identifier of the parser used to produce chunks.
            chunk_config_hash: Hash of the chunking configuration used.

        Returns:
            True if at least one matching point exists in the collection, False otherwise.
        """
        result = self._client.count(
            collection_name=self._collection,
            count_filter=self._match_filter(content_hash, parser_version, chunk_config_hash),
            exact=True,
        )
        return result.count > 0

    def upsert(
        self,
        content_hash: str,
        source: str,
        parser_version: str,
        chunk_config_hash: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """Upsert one Qdrant point per chunk, pairing each chunk with its vector.

        Args:
            content_hash: Hash of the source document's content.
            source: Source identifier of the document (e.g., S3 URI).
            parser_version: Version identifier of the parser used to produce chunks.
            chunk_config_hash: Hash of the chunking configuration used.
            chunks: Chunks extracted from the document.
            vectors: Embedding vectors, one per chunk, aligned by position.

        Returns:
            The number of points upserted.
        """
        points = [
            models.PointStruct(
                id=point_id(content_hash, chunk.index),
                vector={VECTOR_NAME: vector},
                payload={
                    "source": source,
                    "content_hash": content_hash,
                    "page_numbers": list(chunk.page_numbers),
                    "section_path": chunk.section_path,
                    "chunk_index": chunk.index,
                    "parser_version": parser_version,
                    "chunk_config_hash": chunk_config_hash,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
        return len(points)
