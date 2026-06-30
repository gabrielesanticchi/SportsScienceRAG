"""Layout-aware PDF parsing with Docling and optional fallbacks."""

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fitz
import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedDocument:
    markdown: str
    page_count: int
    parser_used: str
    has_complex_tables: bool = False


def _detect_complex_tables(text: str) -> bool:
    pipe_rows = len(re.findall(r"\|.*\|.*\|", text))
    return pipe_rows >= 6


def _parse_with_docling(pdf_bytes: bytes) -> ParsedDocument:
    from docling.document_converter import DocumentConverter

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        converter = DocumentConverter()
        result = converter.convert(str(tmp_path))
        markdown = result.document.export_to_markdown()
        page_count = len(result.document.pages) if result.document.pages else 1
        return ParsedDocument(
            markdown=markdown,
            page_count=page_count,
            parser_used="docling",
            has_complex_tables=_detect_complex_tables(markdown),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_with_pymupdf(pdf_bytes: bytes) -> ParsedDocument:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    markdown = "\f".join(pages)
    return ParsedDocument(
        markdown=markdown,
        page_count=len(doc),
        parser_used="pymupdf",
        has_complex_tables=_detect_complex_tables(markdown),
    )


def _parse_with_llamaparse(pdf_bytes: bytes, api_key: str) -> ParsedDocument:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        with httpx.Client(timeout=120.0) as client:
            with tmp_path.open("rb") as handle:
                response = client.post(
                    "https://api.cloud.llamaindex.ai/api/parsing/upload",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (tmp_path.name, handle, "application/pdf")},
                )
            response.raise_for_status()
            job_id = response.json()["id"]

            status_response = client.get(
                f"https://api.cloud.llamaindex.ai/api/parsing/job/{job_id}",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            status_response.raise_for_status()
            markdown = status_response.json().get("markdown", "")
            return ParsedDocument(
                markdown=markdown,
                page_count=max(markdown.count("\f"), 1),
                parser_used="llamaparse",
            )
    finally:
        tmp_path.unlink(missing_ok=True)


def _parse_with_reducto(pdf_bytes: bytes, api_key: str) -> ParsedDocument:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://platform.reducto.ai/parse",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
        )
        response.raise_for_status()
        payload = response.json()
        markdown = payload.get("result", {}).get("markdown", "")
        return ParsedDocument(
            markdown=markdown,
            page_count=payload.get("result", {}).get("num_pages", 1),
            parser_used="reducto",
        )


def parse_pdf(pdf_bytes: bytes, settings: Settings | None = None) -> ParsedDocument:
    """Parse PDF using Docling, with LlamaParse/Reducto fallback for complex tables."""
    cfg = settings or get_settings()

    try:
        parsed = _parse_with_docling(pdf_bytes)
    except Exception as exc:
        logger.warning("Docling parse failed, falling back to PyMuPDF: %s", exc)
        parsed = _parse_with_pymupdf(pdf_bytes)

    if parsed.has_complex_tables:
        if cfg.llamaparse_api_key:
            try:
                return _parse_with_llamaparse(pdf_bytes, cfg.llamaparse_api_key)
            except Exception as exc:
                logger.warning("LlamaParse fallback failed: %s", exc)
        if cfg.reducto_api_key:
            try:
                return _parse_with_reducto(pdf_bytes, cfg.reducto_api_key)
            except Exception as exc:
                logger.warning("Reducto fallback failed: %s", exc)

    return parsed
