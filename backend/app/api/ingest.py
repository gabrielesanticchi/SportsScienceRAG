"""Ingestion REST endpoints."""

import hashlib
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.auth import TenantContext, get_current_tenant
from app.celery_app.tasks import process_document
from app.config import Settings, get_settings
from app.db.session import Document, DocumentStatus, tenant_session
from app.storage.s3 import ObjectStorage, build_s3_key, get_storage

router = APIRouter(prefix="/v1/ingest", tags=["ingest"])


class UploadAcceptedResponse(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    sha256: str
    message: str = Field(default="Document accepted for asynchronous processing")


class DuplicateResponse(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    sha256: str
    duplicate: bool = True
    message: str = Field(default="Duplicate document detected; ingestion skipped")


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadAcceptedResponse | DuplicateResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(get_current_tenant),
    settings: Settings = Depends(get_settings),
    storage: ObjectStorage = Depends(get_storage),
) -> UploadAcceptedResponse | DuplicateResponse:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF uploads are supported",
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload")

    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    filename = file.filename or "upload.pdf"

    async with tenant_session(tenant.tenant_id) as session:
        existing = await session.execute(
            select(Document).where(
                Document.tenant_id == tenant.tenant_id,
                Document.sha256 == sha256,
            )
        )
        duplicate = existing.scalar_one_or_none()
        if duplicate is not None:
            return DuplicateResponse(
                document_id=duplicate.id,
                status=duplicate.status,
                sha256=sha256,
            )

        document_id = uuid.uuid4()
        s3_key = build_s3_key(str(tenant.tenant_id), str(document_id), filename)
        storage.put_bytes(s3_key, raw_bytes, content_type="application/pdf")

        document = Document(
            id=document_id,
            tenant_id=tenant.tenant_id,
            workspace_id="default",
            filename=filename,
            content_type="application/pdf",
            sha256=sha256,
            s3_key=s3_key,
            status=DocumentStatus.PENDING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        session.add(document)
        await session.commit()

    process_document.apply_async(
        args=[str(document_id), str(tenant.tenant_id)],
        countdown=1,
    )

    return UploadAcceptedResponse(
        document_id=document_id,
        status=DocumentStatus.PENDING,
        sha256=sha256,
    )
