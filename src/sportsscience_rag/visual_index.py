"""Dormant extension point for a future visual (ColPali/Qwen-VL) collection.

Nothing here is wired into the ingestion pipeline. Page renders are already
persisted to S3 at ``<derived_prefix><content_hash>/screenshots/page-<N>.png``
by ``persistence.RenderStore``. A future job would:

  1. List render keys for each ``content_hash`` under the derived prefix.
  2. Embed each page image with a multi-vector visual model (e.g. ColPali).
  3. Create a SEPARATE Qdrant collection with a multi-vector config and upsert
     one point per page, payload-linked back to ``content_hash``/``page``.

Kept deliberately unimplemented — this is the seed, not the build.
"""

from __future__ import annotations

from typing import Any


class VisualIndexPlaceholder:
    """Marks where visual indexing would live. Intentionally inert."""

    def index_from_renders(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Visual (ColPali/Qwen-VL) indexing is a future job; renders are "
            "already persisted to S3 as the seed."
        )
