"""S3-compatible object storage (MinIO / LocalStack)."""

from io import BytesIO

import boto3
from botocore.client import Config

from app.config import Settings, get_settings


class ObjectStorage:
    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.bucket = cfg.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=cfg.s3_endpoint_url,
            aws_access_key_id=cfg.s3_access_key,
            aws_secret_access_key=cfg.s3_secret_key,
            region_name=cfg.s3_region,
            config=Config(signature_version="s3v4"),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/pdf") -> str:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        return key

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def get_stream(self, key: str) -> BytesIO:
        return BytesIO(self.get_bytes(key))


def build_s3_key(tenant_id: str, document_id: str, filename: str) -> str:
    safe_name = filename.replace("/", "_")
    return f"tenants/{tenant_id}/documents/{document_id}/{safe_name}"


def get_storage(settings: Settings | None = None) -> ObjectStorage:
    return ObjectStorage(settings)
