"""Presigned-style raw upload endpoint for the frontend."""

import hashlib
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from sqlalchemy import select

from app.celery_app.tasks import process_document
from app.db.session import Document, DocumentStatus, SyncSession, set_tenant_context_sync
from app.services.event_bus import publish_parsing_event
from app.storage.s3 import ObjectStorage, build_s3_key

router = APIRouter(prefix="/api/v1/upload", tags=["upload"])


@router.put("/{upload_token}")
async def complete_upload(upload_token: str, request: Request) -> Response:
    raw_bytes = await request.body()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    with SyncSession() as session:
        result = session.execute(
            select(Document).where(Document.upload_token == upload_token)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise HTTPException(status_code=404, detail="Invalid upload token")

        set_tenant_context_sync(session, str(document.tenant_id))

        duplicate = session.execute(
            select(Document).where(
                Document.tenant_id == document.tenant_id,
                Document.sha256 == sha256,
                Document.id != document.id,
            )
        ).scalar_one_or_none()
        if duplicate is not None:
            session.delete(document)
            session.commit()
            raise HTTPException(status_code=409, detail="Duplicate document")

        storage = ObjectStorage()
        s3_key = build_s3_key(str(document.tenant_id), str(document.id), document.filename)
        storage.put_bytes(s3_key, raw_bytes, content_type="application/pdf")

        document.sha256 = sha256
        document.s3_key = s3_key
        document.status = DocumentStatus.PROCESSING
        document.upload_token = None
        document.updated_at = datetime.now(UTC)
        session.commit()

        tenant_id = str(document.tenant_id)
        workspace_id = document.workspace_id
        document_id = str(document.id)

    publish_parsing_event(
        tenant_id,
        workspace_id,
        {
            "documentId": document_id,
            "status": "PROCESSING",
            "progress": 10,
            "subState": "Extracting text",
        },
    )

    process_document.apply_async(args=[document_id, tenant_id], countdown=1)

    return Response(status_code=status.HTTP_200_OK)
