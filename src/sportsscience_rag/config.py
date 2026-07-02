"""Environment-driven, validated ingestion configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from sportsscience_rag.hashing import chunk_config_hash

DEFAULT_HEADERS: tuple[tuple[str, str], ...] = (
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
)
_REQUIRED = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "S3_BUCKET",
    "QDRANT_URL",
    "QDRANT_API_KEY",
)


@dataclass(frozen=True)
class IngestionConfig:
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    s3_bucket: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "sport-science-documents"
    chunk_size: int = 256   # MiniLM truncates at 256 tokens
    chunk_overlap: int = 32
    headers: tuple[tuple[str, str], ...] = field(default=DEFAULT_HEADERS)
    image_dpi: int = 150
    derived_prefix: str = "derived/"

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "IngestionConfig":
        if env_path is not None:
            load_dotenv(env_path)
        values = {key: os.getenv(key, "") for key in _REQUIRED}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError("Missing required environment variables: " + ", ".join(missing))
        return cls(
            aws_access_key_id=values["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=values["AWS_SECRET_ACCESS_KEY"],
            aws_region=values["AWS_REGION"],
            s3_bucket=values["S3_BUCKET"],
            qdrant_url=values["QDRANT_URL"],
            qdrant_api_key=values["QDRANT_API_KEY"],
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "sport-science-documents"),
        )

    @property
    def chunk_config_hash(self) -> str:
        return chunk_config_hash(self.chunk_size, self.chunk_overlap, self.headers)
