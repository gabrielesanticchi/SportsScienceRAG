from pathlib import Path

import pytest

from sportsscience_rag.parser import PARSER_VERSION, DoclingParser

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def test_parser_version_is_string():
    assert isinstance(PARSER_VERSION, str)
    assert PARSER_VERSION.startswith("docling-")


@pytest.mark.integration
def test_parse_real_pdf_produces_markdown_and_renders():
    pdf = next(ASSETS.glob("*.pdf"))
    result = DoclingParser().parse(pdf.read_bytes(), pdf.name)
    assert not result.is_empty
    assert len(result.markdown.strip()) > 100
    assert result.page_count >= 1
    assert len(result.renders) == result.page_count
    assert result.renders[0].image.width > 0
    assert len(result.page_texts) >= 1
    assert result.page_texts[0][0] >= 1  # a page number
    assert isinstance(result.page_texts[0][1], str)
