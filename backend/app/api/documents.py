"""Serve stored PDF bytes for the citation viewer."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response

from app.auth import TenantContext, decode_token, get_current_tenant
from app.db.session import Document, tenant_session
from app.storage.s3 import get_storage

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def _resolve_tenant(request: Request, token: str | None) -> TenantContext:
    if token:
        return decode_token(token)

    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return decode_token(header.removeprefix("Bearer "))

    raise HTTPException(status_code=401, detail="Missing auth token")


@router.get("/{document_id}/pdf")
async def get_document_pdf(
    document_id: UUID,
    request: Request,
    token: str | None = Query(default=None),
) -> Response:
    tenant = _resolve_tenant(request, token)

    async with tenant_session(tenant.tenant_id) as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Not found")

    storage = get_storage()
    pdf_bytes = storage.get_bytes(document.s3_key)
    return Response(content=pdf_bytes, media_type="application/pdf")
