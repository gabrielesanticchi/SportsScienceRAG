"""PDF parsing via Unstructured."""

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.services.unstructured_parser import parse_with_unstructured


@dataclass(frozen=True)
class ParsedDocument:
    markdown: str
    page_count: int
    parser_used: str


def parse_pdf(pdf_bytes: bytes, settings: Settings | None = None) -> ParsedDocument:
    cfg = settings or get_settings()
    markdown, page_count, parser_used, _, _ = parse_with_unstructured(
        pdf_bytes,
        strategy=cfg.unstructured_strategy,
        infer_table_structure=cfg.unstructured_infer_tables,
    )
    if not markdown.strip():
        raise ValueError("Unstructured produced no text from PDF")

    return ParsedDocument(
        markdown=markdown,
        page_count=page_count,
        parser_used=parser_used,
    )
