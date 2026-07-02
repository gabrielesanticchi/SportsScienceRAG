"""List and fetch PDF objects from S3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def parse_s3_url(url: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URL into its bucket and key components.

    Args:
        url: A fully qualified S3 URL, e.g. ``"s3://bucket/path/to/object.pdf"``.
            The ``"s3://"`` scheme prefix is optional; if the URL contains no
            ``/`` after the bucket name, the key defaults to an empty string.

    Returns:
        A ``(bucket, key)`` tuple, where ``bucket`` is the S3 bucket name and
        ``key`` is the object key (empty string if the URL has no key).
    """
    parts = url.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


class S3Source:
    """Enumerates and downloads ``.pdf`` objects under given prefixes.

    Wraps a boto3-compatible S3 client to provide PDF-focused listing and
    fetching operations scoped to a single bucket.

    Attributes:
        _client: A boto3-compatible S3 client used to list and fetch objects.
        _bucket: The name of the S3 bucket this source operates against.
    """

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def list_pdfs(
        self,
        prefixes: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """Return ``(key, source_url)`` for each PDF, capped at ``limit``.

        Iterates over each prefix in order, paginating through
        ``list_objects_v2`` results and collecting keys that end in
        ``.pdf`` (case-insensitive), skipping "directory" keys that end
        with ``/``. The cap on ``limit`` applies to the running total
        across all prefixes combined, not per-prefix: once the total
        number of matches reaches ``limit``, iteration stops immediately
        even if later prefixes were not yet scanned.

        Args:
            prefixes: S3 key prefixes to search, scanned in order.
            limit: Maximum total number of PDFs to return across all
                prefixes combined. If ``None``, all matching PDFs under
                every prefix are returned.

        Returns:
            A list of ``(key, source_url)`` tuples, where ``key`` is the
            S3 object key and ``source_url`` is the corresponding
            ``s3://bucket/key`` URL. The list is capped at ``limit`` items
            (or unbounded if ``limit`` is ``None``).
        """
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
        """Download and return the raw bytes of an object.

        Args:
            key: The S3 object key to fetch, relative to the configured
                bucket.

        Returns:
            The raw bytes of the object's body.
        """
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()
