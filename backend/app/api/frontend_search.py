"""Frontend search query contract."""

import asyncio
import uuid
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import TenantContext, get_current_tenant
from app.db.session import tenant_session
from app.services.search import HybridSearchService

router = APIRouter(prefix="/api/v1/search", tags=["frontend-search"])


class SearchQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    workspace_id: str
    document_ids: list[UUID] = Field(min_length=1)


@router.post("/query")
async def search_query(
    body: SearchQueryRequest,
    tenant: TenantContext = Depends(get_current_tenant),
) -> dict:
    service = HybridSearchService()

    async with tenant_session(tenant.tenant_id) as session:
        try:
            await service.validate_tenant_user(session, tenant.user_id, tenant.tenant_id)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    hits = await asyncio.to_thread(
        service.search,
        body.query,
        tenant.tenant_id,
        document_ids=[str(doc_id) for doc_id in body.document_ids],
        limit=10,
    )

    if not hits:
        return {
            "answer": "No relevant passages were found in the selected documents.",
            "citations": [],
        }

    citations = []
    for index, hit in enumerate(hits[:5], start=1):
        citations.append(
            {
                "id": str(uuid.uuid4()),
                "documentId": hit.document_id,
                "boundingBoxId": hit.point_id,
                "text": hit.text[:500],
                "page": hit.page_number or 1,
                "label": str(index),
            }
        )

    top = hits[0]
    answer = (
        f"Based on {len(body.document_ids)} selected source(s), here is a synthesized answer "
        f'to: "{body.query}". The most relevant passage: {top.text[:400]}'
    )

    return {"answer": answer, "citations": citations}
