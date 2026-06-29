"""SQLAlchemy database layer with tenant-scoped RLS sessions."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
import enum

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    role: Mapped[str] = mapped_column(Text, default="member")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tenants.id"))
    workspace_id: Mapped[str] = mapped_column(Text, default="default", nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(Text, default="application/pdf")
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    s3_key: Mapped[str] = mapped_column(Text, nullable=False)
    upload_token: Mapped[str | None] = mapped_column(Text, nullable=True, unique=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status", create_type=False),
        default=DocumentStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[object] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship(back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True)
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("documents.id"))
    tenant_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("tenants.id"))
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    point_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


settings = get_settings()
engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def set_tenant_context(session: AsyncSession, tenant_id: UUID) -> None:
    await session.execute(
        text("SELECT set_config('app.current_tenant', :tenant_id, true)"),
        {"tenant_id": str(tenant_id)},
    )


def set_tenant_context_sync(session, tenant_id: str) -> None:
    session.execute(
        text("SELECT set_config('app.current_tenant', :tenant_id, true)"),
        {"tenant_id": tenant_id},
    )


@asynccontextmanager
async def tenant_session(tenant_id: UUID) -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        await set_tenant_context(session, tenant_id)
        yield session


async def get_db_session(
    tenant_id: UUID,
) -> AsyncGenerator[AsyncSession, None]:
    async with tenant_session(tenant_id) as session:
        yield session
