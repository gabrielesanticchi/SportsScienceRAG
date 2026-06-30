"""PDF parsing with Unstructured (primary) and Docling fallbacks."""

import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal

import fitz
import httpx

from app.config import Settings, get_settings
from app.services.unstructured_parser import parse_with_unstructured

logger = logging.getLogger(__name__)

PdfParserMode = Literal["unstructured", "docling", "auto", "compare"]


@dataclass(frozen=True)
class ParsedDocument:
    markdown: str
    page_count: int
    parser_used: str
    has_complex_tables: bool = False
    comparison: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParserRunMetrics:
    parser: str
    duration_ms: float
    char_count: int
    page_count: int
    error: str | None = None


def _detect_complex_tables(text: str) -> bool:
    pipe_rows = len(re.findall(r"\|.*\|.*\|", text))
    html_tables = len(re.findall(r"<table[\s>]", text, flags=re.IGNORECASE))
    return pipe_rows >= 6 or html_tables >= 1


def _metrics_from_parsed(parsed: ParsedDocument, duration_ms: float) -> ParserRunMetrics:
    return ParserRunMetrics(
        parser=parsed.parser_used,
        duration_ms=round(duration_ms, 2),
        char_count=len(parsed.markdown),
        page_count=parsed.page_count,
    )


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


def _run_docling(pdf_bytes: bytes) -> tuple[ParsedDocument | None, ParserRunMetrics | None]:
    started = time.perf_counter()
    try:
        parsed = _parse_with_docling(pdf_bytes)
        return parsed, _metrics_from_parsed(parsed, (time.perf_counter() - started) * 1000)
    except Exception as exc:
        logger.warning("Docling parse failed: %s", exc)
        return None, ParserRunMetrics(
            parser="docling",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            char_count=0,
            page_count=0,
            error=str(exc),
        )


def _run_unstructured(
    pdf_bytes: bytes,
    settings: Settings,
) -> tuple[ParsedDocument | None, ParserRunMetrics | None]:
    try:
        markdown, page_count, parser_used, has_complex_tables, metrics = parse_with_unstructured(
            pdf_bytes,
            strategy=settings.unstructured_strategy,
            infer_table_structure=settings.unstructured_infer_tables,
        )
        parsed = ParsedDocument(
            markdown=markdown,
            page_count=page_count,
            parser_used=parser_used,
            has_complex_tables=has_complex_tables,
        )
        return parsed, ParserRunMetrics(
            parser=parsed.parser_used,
            duration_ms=metrics.duration_ms,
            char_count=metrics.char_count,
            page_count=metrics.page_count,
        )
    except Exception as exc:
        logger.warning("Unstructured parse failed: %s", exc)
        return None, ParserRunMetrics(
            parser=f"unstructured:{settings.unstructured_strategy}",
            duration_ms=0,
            char_count=0,
            page_count=0,
            error=str(exc),
        )


def _apply_table_fallbacks(pdf_bytes: bytes, parsed: ParsedDocument, cfg: Settings) -> ParsedDocument:
    if not parsed.has_complex_tables:
        return parsed

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


def _comparison_payload(
    *,
    selected: str,
    unstructured: ParserRunMetrics | None,
    docling: ParserRunMetrics | None,
) -> dict[str, object]:
    payload: dict[str, object] = {"selected": selected}
    if unstructured is not None:
        payload["unstructured"] = unstructured.__dict__
    if docling is not None:
        payload["docling"] = docling.__dict__
    return payload


def parse_pdf(pdf_bytes: bytes, settings: Settings | None = None) -> ParsedDocument:
    """Parse PDF using Unstructured by default, with optional Docling comparison."""
    cfg = settings or get_settings()
    mode = cfg.pdf_parser

    if mode == "compare":
        unstructured_doc, unstructured_metrics = _run_unstructured(pdf_bytes, cfg)
        docling_doc, docling_metrics = _run_docling(pdf_bytes)

        selected_doc = unstructured_doc or docling_doc
        if selected_doc is None:
            logger.warning("Both parsers failed, falling back to PyMuPDF")
            selected_doc = _parse_with_pymupdf(pdf_bytes)
            selected_name = selected_doc.parser_used
        else:
            selected_name = selected_doc.parser_used

        comparison = _comparison_payload(
            selected=selected_name,
            unstructured=unstructured_metrics,
            docling=docling_metrics,
        )
        logger.info("Parser comparison: %s", comparison)

        result = _apply_table_fallbacks(pdf_bytes, selected_doc, cfg)
        return ParsedDocument(
            markdown=result.markdown,
            page_count=result.page_count,
            parser_used=result.parser_used,
            has_complex_tables=result.has_complex_tables,
            comparison=comparison,
        )

    if mode == "unstructured":
        unstructured_doc, _ = _run_unstructured(pdf_bytes, cfg)
        if unstructured_doc is None:
            logger.warning("Unstructured failed, falling back to Docling/PyMuPDF chain")
            return parse_pdf_with_docling_chain(pdf_bytes, cfg)
        result = _apply_table_fallbacks(pdf_bytes, unstructured_doc, cfg)
        return result

    if mode == "docling":
        return parse_pdf_with_docling_chain(pdf_bytes, cfg)

    # auto: unstructured first, then docling chain
    unstructured_doc, _ = _run_unstructured(pdf_bytes, cfg)
    if unstructured_doc is not None:
        return _apply_table_fallbacks(pdf_bytes, unstructured_doc, cfg)
    return parse_pdf_with_docling_chain(pdf_bytes, cfg)


def parse_pdf_with_docling_chain(pdf_bytes: bytes, settings: Settings | None = None) -> ParsedDocument:
    """Legacy Docling-first pipeline."""
    cfg = settings or get_settings()

    docling_doc, _ = _run_docling(pdf_bytes)
    if docling_doc is None:
        parsed = _parse_with_pymupdf(pdf_bytes)
    else:
        parsed = docling_doc

    return _apply_table_fallbacks(pdf_bytes, parsed, cfg)
