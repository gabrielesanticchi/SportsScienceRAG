"""Map stored chunks to frontend bounding-box overlays."""

from app.db.session import DocumentChunk


def chunks_to_bounding_boxes(chunks: list[DocumentChunk]) -> list[dict]:
    boxes: list[dict] = []
    for index, chunk in enumerate(chunks):
        page = chunk.page_number or 1
        y_offset = 100 + (index % 4) * 90
        boxes.append(
            {
                "id": chunk.point_id,
                "page": page,
                "x": 48,
                "y": y_offset,
                "width": 500,
                "height": 72,
                "type": "paragraph",
                "text": chunk.text[:500],
            }
        )
    return boxes
