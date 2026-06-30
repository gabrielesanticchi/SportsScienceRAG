"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+asyncpg://ingest:ingest_secret@localhost:5432/ingestion"
    database_url_sync: str = "postgresql+psycopg2://ingest:ingest_secret@localhost:5432/ingestion"

    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "redis://localhost:6379/0"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hybrid_ingestion"

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "ingestion-raw"
    s3_region: str = "us-east-1"

    jwt_secret: str = "dev-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"

    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_dimension: int = 384
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_dimension: int = 128

    chunk_size: int = 512
    chunk_overlap: int = 64

    llamaparse_api_key: str = ""
    reducto_api_key: str = ""

    pdf_parser: str = "unstructured"
    unstructured_strategy: str = "fast"
    unstructured_infer_tables: bool = False

    embedder_rate_limit_jitter: float = 1.0
    celery_task_max_retries: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
