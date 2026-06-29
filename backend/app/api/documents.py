"""Serve stored PDF bytes for the citation viewer."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import select

from app.auth import TenantContext, get_current_tenant
from app.db.session import Document, tenant_session
from app.storage.s3 import get_storage

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("/{document_id}/pdf")
async def get_document_pdf(
    document_id: UUID,
    tenant: TenantContext = Depends(get_current_tenant),
) -> Response:
    async with tenant_session(tenant.tenant_id) as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Not found")

    storage = get_storage()
    pdf_bytes = storage.get_bytes(document.s3_key)
    return Response(content=pdf_bytes, media_type="application/pdf")
