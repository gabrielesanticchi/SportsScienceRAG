"""Idempotent persistence of per-page renders to S3."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

from botocore.exceptions import ClientError

from sportsscience_rag.models import PageRender


class RenderStore:
    """Writes page PNGs to a derived-artifact S3 prefix, skipping duplicates.

    Uploads are idempotent: before writing a page render, the store checks
    whether the destination object already exists (via ``head_object``) and
    skips the upload if so. This makes re-ingestion of a document cheap and
    safe to retry.
    """

    def __init__(self, client: Any, bucket: str, derived_prefix: str = "derived/") -> None:
        """Initializes the store with an S3 client and target location.

        Args:
            client: A boto3 S3 client (or compatible object) implementing
                ``head_object`` and ``put_object``. Injected for testability.
            bucket: Name of the S3 bucket to write renders to.
            derived_prefix: Key prefix under which derived artifacts
                (screenshots) are stored. Defaults to ``"derived/"``.
        """
        self._client = client
        self._bucket = bucket
        self._prefix = derived_prefix

    def render_key(self, content_hash: str, page_number: int) -> str:
        """Builds the S3 object key for a given document page render.

        Args:
            content_hash: Content hash identifying the source document.
            page_number: Page number of the render within the document.

        Returns:
            The S3 key in the form
            ``"<derived_prefix><content_hash>/screenshots/page-<page_number>.png"``.
        """
        return f"{self._prefix}{content_hash}/screenshots/page-{page_number}.png"

    def _exists(self, key: str) -> bool:
        """Checks whether an object already exists at the given S3 key.

        Args:
            key: S3 object key to check.

        Returns:
            True if the object exists, False otherwise (including on any
            ``ClientError`` such as a 404 Not Found response).
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def upload(self, content_hash: str, renders: Sequence[PageRender]) -> list[str]:
        """Uploads page renders to S3, skipping objects that already exist.

        Each render is serialized to PNG in memory and uploaded via
        ``put_object`` unless an object already exists at its computed key,
        in which case the upload is skipped. All computed keys are returned
        regardless of whether they were newly uploaded or already present.

        Args:
            content_hash: Content hash identifying the source document.
            renders: Sequence of ``PageRender`` objects to persist.

        Returns:
            The list of S3 keys corresponding to every render, in order.
        """
        keys: list[str] = []
        for render in renders:
            key = self.render_key(content_hash, render.page_number)
            keys.append(key)
            if self._exists(key):
                continue
            buffer = io.BytesIO()
            render.image.save(buffer, format="PNG")
            buffer.seek(0)
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=buffer.getvalue(),
                ContentType="image/png",
            )
        return keys
