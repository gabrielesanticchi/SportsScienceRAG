"""Two-stage hybrid retrieval service."""

from dataclasses import dataclass
from uuid import UUID

from qdrant_client import models
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import Document, User
from app.qdrant.client import get_qdrant_client
from app.services.embedders import get_hybrid_embedder


@dataclass(frozen=True)
class SearchHit:
    point_id: str
    score: float
    text: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None
    section: str | None


class HybridSearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.qdrant = get_qdrant_client(self.settings.qdrant_url)
        self.embedder = get_hybrid_embedder()

    async def validate_tenant_user(self, session: AsyncSession, user_id: UUID, tenant_id: UUID) -> User:
        result = await session.execute(
            select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        )
        user = result.scalar_one_or_none()
        if user is None:
            raise PermissionError("User is not authorized for this tenant")
        return user

    def _tenant_filter(self, tenant_id: UUID) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=str(tenant_id)),
                )
            ]
        )

    def search(
        self,
        query: str,
        tenant_id: UUID,
        *,
        document_ids: list[str] | None = None,
        limit: int = 10,
    ) -> list[SearchHit]:
        query_vectors = self.embedder.embed_query(query)
        tenant_filter = self._tenant_filter(tenant_id)

        if document_ids:
            tenant_filter.must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=document_ids),
                )
            )

        response = self.qdrant.query_points(
            collection_name=self.settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=query_vectors.sparse,
                    using="sparse",
                    filter=tenant_filter,
                    limit=20,
                ),
                models.Prefetch(
                    query=query_vectors.dense,
                    using="dense",
                    filter=tenant_filter,
                    limit=20,
                ),
            ],
            query=query_vectors.multi,
            using="multi",
            query_filter=tenant_filter,
            limit=limit,
            with_payload=True,
        )

        hits: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            hits.append(
                SearchHit(
                    point_id=str(point.id),
                    score=float(point.score or 0.0),
                    text=str(payload.get("text", "")),
                    document_id=str(payload.get("document_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    page_number=payload.get("page_number"),
                    section=payload.get("section"),
                )
            )
        return hits
