"""Frontend workspace API contract."""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.auth import TenantContext, decode_token, get_current_tenant
from app.celery_app.tasks import process_document
from app.config import Settings, get_settings
from app.db.session import Document, DocumentChunk, DocumentStatus, tenant_session
from app.services.bounding_boxes import chunks_to_bounding_boxes
from app.services.event_bus import subscribe_parsing_events
from app.storage.s3 import ObjectStorage, build_s3_key, get_storage

router = APIRouter(prefix="/api/v1/workspaces", tags=["workspace"])


class UploadInitRequest(BaseModel):
    filename: str
    size_bytes: int = Field(ge=1)


class UploadInitResponse(BaseModel):
    document_id: UUID
    upload_url: str


def _document_to_workspace_doc(
    document: Document,
    chunks: list[DocumentChunk],
    *,
    api_base: str,
) -> dict:
    status = document.status.value
    if status == "PENDING":
        status = "UPLOADING"
    return {
        "id": str(document.id),
        "name": document.filename,
        "status": status,
        "pageCount": document.page_count,
        "boundingBoxes": chunks_to_bounding_boxes(chunks),
        "pdfUrl": f"{api_base}/api/v1/documents/{document.id}/pdf",
        "selected": True,
        "uploadedAt": document.created_at.isoformat() if document.created_at else datetime.now(UTC).isoformat(),
    }


@router.post("/{workspace_id}/documents/upload", status_code=status.HTTP_201_CREATED)
async def init_document_upload(
    workspace_id: str,
    body: UploadInitRequest,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
    settings: Settings = Depends(get_settings),
) -> UploadInitResponse:
    if body.size_bytes > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds the 50MB size limit")

    document_id = uuid.uuid4()
    upload_token = uuid.uuid4().hex
    origin = str(request.base_url).rstrip("/")

    async with tenant_session(tenant.tenant_id) as session:
        document = Document(
            id=document_id,
            tenant_id=tenant.tenant_id,
            workspace_id=workspace_id,
            filename=body.filename,
            content_type="application/pdf",
            sha256=f"pending:{upload_token}",
            s3_key=f"pending/{upload_token}",
            status=DocumentStatus.PENDING,
            upload_token=upload_token,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(document)
        await session.commit()

    return UploadInitResponse(
        document_id=document_id,
        upload_url=f"{origin}/api/v1/upload/{upload_token}",
    )


@router.get("/{workspace_id}/documents")
async def list_workspace_documents(
    workspace_id: str,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
) -> dict:
    api_base = str(request.base_url).rstrip("/")
    async with tenant_session(tenant.tenant_id) as session:
        result = await session.execute(
            select(Document).where(
                Document.tenant_id == tenant.tenant_id,
                Document.workspace_id == workspace_id,
                Document.status != DocumentStatus.PENDING,
            )
        )
        documents = result.scalars().all()
        payload = []
        for document in documents:
            chunks_result = await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document.id)
            )
            chunks = list(chunks_result.scalars().all())
            payload.append(_document_to_workspace_doc(document, chunks, api_base=api_base))
    return {"documents": payload}


@router.get("/{workspace_id}/documents/{document_id}")
async def get_workspace_document(
    workspace_id: str,
    document_id: UUID,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
) -> dict:
    api_base = str(request.base_url).rstrip("/")
    async with tenant_session(tenant.tenant_id) as session:
        document = await session.get(Document, document_id)
        if document is None or document.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Not found")
        chunks_result = await session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
        chunks = list(chunks_result.scalars().all())
        return _document_to_workspace_doc(document, chunks, api_base=api_base)


@router.get("/{workspace_id}/documents/events")
async def parsing_events(
    workspace_id: str,
    request: Request,
    token: str | None = Query(default=None),
) -> StreamingResponse:
    auth_token = token
    if not auth_token:
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            auth_token = header.removeprefix("Bearer ")

    if not auth_token:
        raise HTTPException(status_code=401, detail="Missing auth token")

    tenant = decode_token(auth_token)

    async def event_stream():
        pubsub = subscribe_parsing_events(str(tenant.tenant_id), workspace_id)
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                message = pubsub.get_message(timeout=1.0)
                if message and message.get("type") == "message":
                    data = message["data"]
                    yield f"event: parsing_update\ndata: {data}\n\n"
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)
        finally:
            pubsub.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
