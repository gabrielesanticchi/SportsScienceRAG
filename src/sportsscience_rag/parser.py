"""Docling-based PDF parsing with page rendering and empty-text triage."""

from __future__ import annotations

import io
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from sportsscience_rag.models import PageRender, ParsedDocument

logger = logging.getLogger(__name__)


def _docling_version() -> str:
    """Resolve the installed Docling distribution version.

    Docling 2.x does not expose a module-level ``__version__`` attribute, so
    the version is read from the installed package metadata instead.

    Returns:
        The installed Docling version string, or ``"unknown"`` when the
        distribution metadata cannot be located.
    """
    try:
        return version("docling")
    except PackageNotFoundError:
        return "unknown"


PARSER_VERSION = f"docling-{_docling_version()}"


class DoclingParser:
    """Converts PDF bytes to markdown, page renders, and per-page text.

    The parser runs Docling with OCR disabled and per-page image generation
    enabled. It exports the structured document to markdown, rasterizes each
    page at the requested DPI, and builds a best-effort page-to-text mapping
    from the document's text items for downstream page attribution. Documents
    whose stripped markdown is shorter than ``min_chars`` are flagged as empty
    so the ingestion pipeline can quarantine them for manual review.

    Attributes:
        _min_chars: Minimum stripped-markdown length for a document to be
            considered non-empty.
        _converter: Configured Docling ``DocumentConverter`` instance.
    """

    def __init__(self, image_dpi: int = 150, min_chars: int = 20) -> None:
        """Initialize the parser and its Docling converter.

        Args:
            image_dpi: Target resolution for per-page renders. Docling scales
                images relative to 72 DPI, so ``images_scale`` is set to
                ``image_dpi / 72``.
            min_chars: Minimum stripped-markdown length below which a parsed
                document is flagged as empty.
        """
        self._min_chars = min_chars
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.generate_page_images = True
        options.images_scale = image_dpi / 72.0  # Docling scale is relative to 72 DPI
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse(self, data: bytes, name: str) -> ParsedDocument:
        """Parse PDF bytes into a fully rendered :class:`ParsedDocument`.

        Args:
            data: Raw bytes of the PDF file.
            name: File name used to identify the in-memory document stream.

        Returns:
            A :class:`ParsedDocument` containing the exported markdown, page
            count, per-page renders, an emptiness flag, and per-page text.
        """
        stream = DocumentStream(name=name, stream=io.BytesIO(data))
        result = self._converter.convert(stream)
        document = result.document
        markdown = document.export_to_markdown()
        pages = sorted(document.pages.values(), key=lambda p: p.page_no)
        renders = tuple(
            PageRender(page_number=page.page_no, image=page.image.pil_image)
            for page in pages
            if page.image is not None and page.image.pil_image is not None
        )
        if len(renders) < len(pages):
            logger.warning(
                "Only %d of %d pages produced a render for %s",
                len(renders), len(pages), name,
            )
        is_empty = len(markdown.strip()) < self._min_chars
        return ParsedDocument(
            markdown=markdown,
            page_count=len(pages),
            renders=renders,
            is_empty=is_empty,
            page_texts=self._page_texts(document),
        )

    @staticmethod
    def _page_texts(document: Any) -> tuple[tuple[int, str], ...]:
        """Concatenate each text item's text under the page it originated from.

        Iterates the structured document's text items and groups their content
        by the page number recorded in each item's provenance entries. The
        result supports best-effort mapping of chunks back to source pages.

        Args:
            document: Docling structured document exposing ``texts``, where each
                item has a ``text`` string and a ``prov`` list carrying a
                ``page_no`` per provenance entry.

        Returns:
            A tuple of ``(page_no, concatenated_text)`` pairs sorted by ascending
            page number, omitting pages that carry no non-empty text.
        """
        by_page: dict[int, list[str]] = {}
        for item in getattr(document, "texts", []):
            text = getattr(item, "text", "") or ""
            if not text.strip():
                continue
            for prov in getattr(item, "prov", []) or []:
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    by_page.setdefault(page_no, []).append(text)
        return tuple((no, " ".join(by_page[no])) for no in sorted(by_page))
