"""Map stored chunks to frontend citation anchors (no fake PDF coordinates)."""

from app.db.session import DocumentChunk


def chunks_to_bounding_boxes(chunks: list[DocumentChunk]) -> list[dict]:
    """Return citation anchors keyed by chunk id; spatial overlays are not estimated."""
    return [
        {
            "id": chunk.point_id,
            "page": chunk.page_number or 1,
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "type": "paragraph",
            "text": chunk.text[:500],
            "estimated": True,
        }
        for chunk in chunks
    ]
