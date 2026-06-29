"""Hybrid search REST endpoints."""

import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import TenantContext, get_current_tenant
from app.db.session import tenant_session
from app.services.search import HybridSearchService, SearchHit

router = APIRouter(prefix="/v1/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4096)
    document_ids: list[UUID] | None = None
    limit: int = Field(default=10, ge=1, le=50)


class SearchResultItem(BaseModel):
    point_id: str
    score: float
    text: str
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None = None
    section: str | None = None
    citation: str


class SearchResponse(BaseModel):
    query: str
    tenant_id: UUID
    results: list[SearchResultItem]


def _format_citation(hit: SearchHit) -> str:
    location = f"p.{hit.page_number}" if hit.page_number else f"chunk {hit.chunk_index}"
    return f"{hit.filename} ({location})"


@router.post("", response_model=SearchResponse)
async def hybrid_search(
    body: SearchRequest,
    tenant: TenantContext = Depends(get_current_tenant),
) -> SearchResponse:
    service = HybridSearchService()

    async with tenant_session(tenant.tenant_id) as session:
        try:
            await service.validate_tenant_user(session, tenant.user_id, tenant.tenant_id)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    document_filter = [str(doc_id) for doc_id in body.document_ids] if body.document_ids else None
    hits = await asyncio.to_thread(
        service.search,
        body.query,
        tenant.tenant_id,
        document_ids=document_filter,
        limit=body.limit,
    )

    return SearchResponse(
        query=body.query,
        tenant_id=tenant.tenant_id,
        results=[
            SearchResultItem(
                point_id=hit.point_id,
                score=hit.score,
                text=hit.text,
                document_id=hit.document_id,
                filename=hit.filename,
                chunk_index=hit.chunk_index,
                page_number=hit.page_number,
                section=hit.section,
                citation=_format_citation(hit),
            )
            for hit in hits
        ],
    )
