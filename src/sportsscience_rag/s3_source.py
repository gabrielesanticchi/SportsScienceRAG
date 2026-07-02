"""List and fetch PDF objects from S3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def parse_s3_url(url: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    parts = url.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


class S3Source:
    """Enumerates and downloads ``.pdf`` objects under given prefixes."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def list_pdfs(
        self,
        prefixes: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """Return ``(key, source_url)`` for each PDF, capped at ``limit``."""
        found: list[tuple[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/") or not key.lower().endswith(".pdf"):
                        continue
                    found.append((key, f"s3://{self._bucket}/{key}"))
                    if limit is not None and len(found) >= limit:
                        return found
        return found

    def fetch(self, key: str) -> bytes:
        """Download and return the raw bytes of an object."""
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()
