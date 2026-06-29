"""Qdrant client helpers."""

from functools import lru_cache

from qdrant_client import QdrantClient

from app.config import Settings, get_settings


@lru_cache
def get_qdrant_client(url: str | None = None) -> QdrantClient:
    cfg = get_settings()
    return QdrantClient(url=url or cfg.qdrant_url, prefer_grpc=False)
