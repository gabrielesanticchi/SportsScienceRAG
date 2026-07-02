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
    """Environment-driven configuration for document ingestion pipeline.

    Immutable configuration container for AWS S3, Qdrant vector DB, and chunking
    parameters. Load from environment variables via from_env() classmethod.

    Attributes:
        aws_access_key_id: AWS access key for S3 authentication.
        aws_secret_access_key: AWS secret access key for S3 authentication.
        aws_region: AWS region where S3 bucket and services are located.
        s3_bucket: S3 bucket name for storing traces and processed documents.
        qdrant_url: URL endpoint of the Qdrant vector database.
        qdrant_api_key: API key for Qdrant authentication.
        qdrant_collection: Qdrant collection name (default: "sport-science-documents").
        chunk_size: Maximum tokens per chunk (default: 256, aligned with MiniLM).
        chunk_overlap: Token overlap between consecutive chunks (default: 32).
        headers: Tuple of markdown header (text, tag) pairs for hierarchical parsing.
        image_dpi: DPI setting for PDF image rendering (default: 150).
        derived_prefix: S3 prefix for derived/processed artifacts (default: "derived/").
    """
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

    def __post_init__(self) -> None:
        """Validates chunking bounds.

        Raises:
            ValueError: If ``chunk_size`` is not positive, if
                ``chunk_overlap`` is negative, or if ``chunk_overlap`` is
                greater than or equal to ``chunk_size`` (which would prevent
                the splitter from making forward progress).
        """
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "IngestionConfig":
        """Load configuration from environment variables.

        Args:
            env_path: Optional path to a .env file to load before reading variables.
                If provided, variables are loaded via dotenv before lookup.

        Returns:
            IngestionConfig instance populated from environment.

        Raises:
            ValueError: If any required environment variable is missing or empty.
                Required variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                AWS_REGION, S3_BUCKET, QDRANT_URL, QDRANT_API_KEY.
        """
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
        """Deterministic hash of chunking configuration.

        Combines chunk_size, chunk_overlap, and headers tuple into a single hash
        for deduplication and caching of chunked documents with identical settings.

        Returns:
            Hexadecimal hash string of the chunking configuration.
        """
        return chunk_config_hash(self.chunk_size, self.chunk_overlap, self.headers)
