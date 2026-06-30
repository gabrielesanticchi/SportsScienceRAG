"""PDF parsing via Unstructured (https://github.com/Unstructured-IO/unstructured)."""

import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnstructuredParseMetrics:
    duration_ms: float
    element_count: int
    char_count: int
    page_count: int


def _detect_complex_tables(text: str) -> bool:
    pipe_rows = len(re.findall(r"\|.*\|.*\|", text))
    html_tables = len(re.findall(r"<table[\s>]", text, flags=re.IGNORECASE))
    return pipe_rows >= 6 or html_tables >= 1


def _format_element_text(element) -> str:
    category = getattr(element, "category", type(element).__name__)
    text = str(element).strip()
    if category == "Table":
        html = getattr(element.metadata, "text_as_html", None)
        if html:
            return html
    if category in {"Title", "Header"} and text:
        return f"## {text}"
    return text


def _elements_to_markdown(elements) -> tuple[str, int, bool]:
    page_blocks: list[list[str]] = [[]]
    max_page = 1

    for element in elements:
        category = getattr(element, "category", type(element).__name__)
        if category == "PageBreak":
            page_blocks.append([])
            continue

        text = _format_element_text(element)
        if not text:
            continue

        page_number = 1
        metadata = getattr(element, "metadata", None)
        if metadata is not None and metadata.page_number:
            page_number = metadata.page_number
            max_page = max(max_page, page_number)

        while len(page_blocks) < page_number:
            page_blocks.append([])

        page_blocks[page_number - 1].append(text)

    rendered_pages = ["\n\n".join(block) for block in page_blocks if block]
    markdown = "\f".join(rendered_pages)
    page_count = max(max_page, len(rendered_pages), 1)
    return markdown, page_count, _detect_complex_tables(markdown)


def parse_with_unstructured(
    pdf_bytes: bytes,
    *,
    strategy: str = "fast",
    infer_table_structure: bool = False,
    languages: list[str] | None = None,
) -> tuple[str, int, str, bool, UnstructuredParseMetrics]:
    from unstructured.partition.pdf import partition_pdf

    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    kwargs: dict = {
        "filename": str(tmp_path),
        "strategy": strategy,
        "include_page_breaks": True,
    }
    if infer_table_structure:
        kwargs["infer_table_structure"] = True
        if strategy == "fast":
            kwargs["strategy"] = "hi_res"
    if languages:
        kwargs["languages"] = languages

    try:
        elements = partition_pdf(**kwargs)
    finally:
        tmp_path.unlink(missing_ok=True)

    markdown, page_count, has_complex_tables = _elements_to_markdown(elements)
    duration_ms = (time.perf_counter() - started) * 1000
    parser_used = f"unstructured:{kwargs['strategy']}"

    metrics = UnstructuredParseMetrics(
        duration_ms=round(duration_ms, 2),
        element_count=len(elements),
        char_count=len(markdown),
        page_count=page_count,
    )

    logger.info(
        "Unstructured parsed PDF strategy=%s pages=%s elements=%s chars=%s duration_ms=%s",
        kwargs["strategy"],
        page_count,
        metrics.element_count,
        metrics.char_count,
        metrics.duration_ms,
    )

    return markdown, page_count, parser_used, has_complex_tables, metrics
