"""Idempotent persistence of derived artifacts (page renders, parsed text) to S3."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

from botocore.exceptions import ClientError

from sportsscience_rag.models import PageRender


class _S3ArtifactStore:
    """Base class for idempotent writers of derived artifacts to an S3 prefix.

    Encapsulates the S3 client, target bucket, and derived-artifact key prefix
    shared by every artifact store, together with the existence check used to
    make uploads idempotent. Subclasses build their own keys and decide what to
    serialize, but reuse :meth:`_exists` so re-ingesting a document never
    re-writes an artifact that is already present.
    """

    def __init__(self, client: Any, bucket: str, derived_prefix: str = "derived/") -> None:
        """Initializes the store with an S3 client and target location.

        Args:
            client: A boto3 S3 client (or compatible object) implementing
                ``head_object`` and ``put_object``. Injected for testability.
            bucket: Name of the S3 bucket to write artifacts to.
            derived_prefix: Key prefix under which derived artifacts are stored.
                Defaults to ``"derived/"``.
        """
        self._client = client
        self._bucket = bucket
        self._prefix = derived_prefix

    def _exists(self, key: str) -> bool:
        """Checks whether an object already exists at the given S3 key.

        Args:
            key: S3 object key to check.

        Returns:
            True if the object exists, False if the head_object call
            returns a genuine not-found error (``404``, ``NotFound``, or
            ``NoSuchKey``).

        Raises:
            ClientError: If ``head_object`` fails with any error other than
                a not-found response (e.g. ``403`` AccessDenied, throttling,
                or a 5xx server error). These are re-raised rather than
                silently treated as "object absent" to avoid masking
                failures during batch ingestion.
        """
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in {"404", "NotFound", "NoSuchKey"}:
                return False
            raise


class RenderStore(_S3ArtifactStore):
    """Writes page PNGs to a derived-artifact S3 prefix, skipping duplicates.

    Uploads are idempotent: before writing a page render, the store checks
    whether the destination object already exists (via ``head_object``) and
    skips the upload if so. This makes re-ingestion of a document cheap and
    safe to retry.
    """

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


class TextStore(_S3ArtifactStore):
    """Writes parsed markdown and per-page text to S3 for human inspection.

    Persists the full Docling-exported markdown alongside the per-page text
    used for page attribution, under the same ``derived/<content_hash>/`` tree
    as the page renders but in a sibling ``text/`` folder. This lets a reviewer
    open the parsed output directly from S3 and compare each ``page-<N>.txt``
    against the corresponding ``screenshots/page-<N>.png``. Uploads are
    idempotent via the inherited existence check.
    """

    def markdown_key(self, content_hash: str) -> str:
        """Builds the S3 object key for a document's full parsed markdown.

        Args:
            content_hash: Content hash identifying the source document.

        Returns:
            The S3 key in the form
            ``"<derived_prefix><content_hash>/text/document.md"``.
        """
        return f"{self._prefix}{content_hash}/text/document.md"

    def page_text_key(self, content_hash: str, page_number: int) -> str:
        """Builds the S3 object key for a single page's extracted text.

        Args:
            content_hash: Content hash identifying the source document.
            page_number: 1-indexed page number (Docling page numbering).

        Returns:
            The S3 key in the form
            ``"<derived_prefix><content_hash>/text/page-<page_number>.txt"``.
        """
        return f"{self._prefix}{content_hash}/text/page-{page_number}.txt"

    def upload(
        self,
        content_hash: str,
        markdown: str,
        page_texts: Sequence[tuple[int, str]],
    ) -> list[str]:
        """Uploads parsed markdown and per-page text, skipping existing objects.

        The full markdown is written to ``text/document.md`` and each
        ``(page_no, text)`` pair to ``text/page-<page_no>.txt``. Each object is
        skipped if it already exists at its computed key. All computed keys are
        returned regardless of whether they were newly uploaded or already
        present, with the markdown key first.

        Args:
            content_hash: Content hash identifying the source document.
            markdown: The full Docling-exported markdown for the document.
            page_texts: Sequence of ``(page_no, text)`` pairs, one per page
                that carried non-empty text.

        Returns:
            The list of S3 keys written or found, markdown key first followed
            by one key per page in order.
        """
        keys: list[str] = []
        md_key = self.markdown_key(content_hash)
        keys.append(md_key)
        if not self._exists(md_key):
            self._client.put_object(
                Bucket=self._bucket,
                Key=md_key,
                Body=markdown.encode("utf-8"),
                ContentType="text/markdown; charset=utf-8",
            )
        for page_no, text in page_texts:
            key = self.page_text_key(content_hash, page_no)
            keys.append(key)
            if self._exists(key):
                continue
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=text.encode("utf-8"),
                ContentType="text/plain; charset=utf-8",
            )
        return keys
