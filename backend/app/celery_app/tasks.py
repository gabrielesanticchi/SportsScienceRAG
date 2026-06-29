"""Asynchronous ingestion pipeline tasks."""

import logging
import random
import uuid
from datetime import UTC, datetime

from celery.exceptions import MaxRetriesExceededError
from qdrant_client import models
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.celery_app.celery import celery
from app.config import get_settings
from app.db.session import Document, DocumentChunk, DocumentStatus
from app.qdrant.client import get_qdrant_client
from app.services.chunker import chunk_text, deterministic_point_id
from app.services.embedders import check_embedder_rate_limit, get_hybrid_embedder
from app.services.event_bus import publish_parsing_event
from app.services.parsers import parse_pdf
from app.storage.s3 import ObjectStorage

logger = logging.getLogger(__name__)
settings = get_settings()

sync_engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
SyncSession = sessionmaker(bind=sync_engine, expire_on_commit=False)


def _compute_retry_delay(attempt: int, jitter_bound: float) -> float:
    return (2**attempt) + random.uniform(0, jitter_bound)


def _emit(
    document: Document,
    *,
    progress: int,
    sub_state: str,
    status: str = "PROCESSING",
    error: str | None = None,
) -> None:
    publish_parsing_event(
        str(document.tenant_id),
        document.workspace_id,
        {
            "documentId": str(document.id),
            "status": status,
            "progress": progress,
            "subState": sub_state,
            **({"error": error} if error else {}),
        },
    )


def _set_document_status(
    session: Session,
    document_id: uuid.UUID,
    status: DocumentStatus,
    *,
    error_message: str | None = None,
    page_count: int | None = None,
) -> None:
    document = session.get(Document, document_id)
    if document is None:
        return
    document.status = status
    document.updated_at = datetime.now(UTC)
    if error_message is not None:
        document.error_message = error_message
    if page_count is not None:
        document.page_count = page_count
    session.commit()


@celery.task(
    bind=True,
    name="ingestion.process_document",
    max_retries=settings.celery_task_max_retries,
)
def process_document(self, document_id: str, tenant_id: str) -> dict:
    doc_uuid = uuid.UUID(document_id)
    tenant_uuid = uuid.UUID(tenant_id)
    storage = ObjectStorage()
    embedder = get_hybrid_embedder()
    qdrant = get_qdrant_client()

    with SyncSession() as session:
        session.execute(
            text("SELECT set_config('app.current_tenant', :tenant_id, true)"),
            {"tenant_id": tenant_id},
        )
        _set_document_status(session, doc_uuid, DocumentStatus.PROCESSING)
        document = session.get(Document, doc_uuid)
        if document is None:
            raise ValueError(f"Document {document_id} not found")

        try:
            _emit(document, progress=20, sub_state="Detecting sections")
            pdf_bytes = storage.get_bytes(document.s3_key)
            _emit(document, progress=35, sub_state="Extracting tables")
            parsed = parse_pdf(pdf_bytes)
            chunks = chunk_text(parsed.markdown)
            if not chunks:
                raise ValueError("No chunks produced from parsed document")

            points: list[models.PointStruct] = []
            for chunk in chunks:
                point_id = deterministic_point_id(document_id, chunk.index)
                _emit(document, progress=55, sub_state="Generating embeddings")
                try:
                    vectors = embedder.embed_text(chunk.text)
                except Exception as exc:
                    if check_embedder_rate_limit(exc) and self.request.retries < self.max_retries:
                        delay = _compute_retry_delay(
                            self.request.retries,
                            settings.embedder_rate_limit_jitter,
                        )
                        raise self.retry(exc=exc, countdown=delay) from exc
                    raise

                chunk_row = session.execute(
                    select(DocumentChunk).where(DocumentChunk.point_id == point_id)
                ).scalar_one_or_none()
                if chunk_row is None:
                    chunk_row = DocumentChunk(
                        id=uuid.uuid4(),
                        document_id=doc_uuid,
                        tenant_id=tenant_uuid,
                        chunk_index=chunk.index,
                        point_id=point_id,
                        text=chunk.text,
                        page_number=chunk.page_number,
                        section=chunk.section,
                    )
                    session.add(chunk_row)
                else:
                    chunk_row.text = chunk.text

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector={
                            "dense": vectors.dense,
                            "sparse": vectors.sparse,
                            "multi": vectors.multi,
                        },
                        payload={
                            "tenant_id": tenant_id,
                            "document_id": document_id,
                            "chunk_index": chunk.index,
                            "text": chunk.text,
                            "filename": document.filename,
                            "page_number": chunk.page_number,
                            "section": chunk.section,
                            "parser": parsed.parser_used,
                        },
                    )
                )

            _emit(document, progress=85, sub_state="Indexing vectors")
            qdrant.upsert(
                collection_name=settings.qdrant_collection,
                points=points,
                wait=True,
            )
            session.commit()
            _set_document_status(
                session,
                doc_uuid,
                DocumentStatus.COMPLETED,
                page_count=parsed.page_count,
            )
            _emit(document, progress=100, sub_state="Ready", status="COMPLETED")
            return {
                "document_id": document_id,
                "chunks": len(points),
                "parser": parsed.parser_used,
            }
        except MaxRetriesExceededError:
            _set_document_status(
                session,
                doc_uuid,
                DocumentStatus.FAILED,
                error_message="Max embedder retries exceeded",
            )
            raise
        except Exception as exc:
            logger.exception("Ingestion failed for %s", document_id)
            _set_document_status(
                session,
                doc_uuid,
                DocumentStatus.FAILED,
                error_message=str(exc),
            )
            if document is not None:
                _emit(document, progress=100, sub_state="Failed", status="FAILED", error=str(exc))
            raise
