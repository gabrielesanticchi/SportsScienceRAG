"""Initialize the shared multi-tenant hybrid Qdrant collection."""

import logging

from qdrant_client import models

from app.config import get_settings
from app.qdrant.client import get_qdrant_client

logger = logging.getLogger(__name__)


def ensure_hybrid_collection(recreate: bool = False) -> None:
    settings = get_settings()
    client = get_qdrant_client(settings.qdrant_url)
    collection = settings.qdrant_collection

    if recreate and client.collection_exists(collection):
        client.delete_collection(collection)

    if client.collection_exists(collection):
        logger.info("Collection %s already exists", collection)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": models.VectorParams(
                size=settings.dense_dimension,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            "multi": models.VectorParams(
                size=settings.colbert_dimension,
                distance=models.Distance.COSINE,
                on_disk=True,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=True),
            ),
        },
        hnsw_config=models.HnswConfigDiff(
            m=0,
            payload_m=16,
            ef_construct=100,
        ),
        on_disk_payload=True,
    )

    client.create_payload_index(
        collection_name=collection,
        field_name="tenant_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )
    client.create_payload_index(
        collection_name=collection,
        field_name="document_id",
        field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD),
    )

    logger.info("Created hybrid collection %s with tenant-scoped HNSW", collection)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_hybrid_collection(recreate=False)


if __name__ == "__main__":
    main()
