from sportsscience_rag.chunker import SectionChunker
from sportsscience_rag.config import IngestionConfig

CFG = IngestionConfig(
    aws_access_key_id="x", aws_secret_access_key="x", aws_region="x",
    s3_bucket="x", qdrant_url="x", qdrant_api_key="x",
    chunk_size=40, chunk_overlap=5,
)

# Fast, deterministic fake tokenizer (word count) so tests need no model.
WORD_LEN = lambda t: len(t.split())


def _chunker():
    return SectionChunker(CFG, token_length=WORD_LEN)


MD = """# Introduction

Soccer performance monitoring uses wearable sensors extensively today.

## Methods

### Algorithm 3.2

We compute metabolic power from GPS and IMU fusion signals here.
"""


def test_sections_carry_absolute_hierarchical_path():
    # section_path is the FULL active-header chain (absolute provenance),
    # not a diff against the previous section.
    chunks = _chunker().chunk(MD)
    paths = {c.section_path for c in chunks}
    assert "Introduction" in paths
    assert "Introduction > Methods > Algorithm 3.2" in paths


def test_distinct_sections_do_not_collide():
    # Two different top-level sections each containing a "## Setup" must yield
    # distinct absolute paths (no ancestor dropping / collision).
    md = (
        "# StudyA\n\n## Setup\n\nalpha setup details here\n\n"
        "# StudyB\n\n## Setup\n\nbeta setup details here\n"
    )
    paths = [c.section_path for c in _chunker().chunk(md)]
    assert "StudyA > Setup" in paths
    assert "StudyB > Setup" in paths


def test_empty_markdown_returns_no_chunks():
    assert _chunker().chunk("") == []


def test_indices_are_sequential_from_zero():
    chunks = _chunker().chunk(MD)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_table_block_not_split():
    md = "# T\n\n| a | b |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |\n"
    chunks = _chunker().chunk(md)
    table_chunks = [c for c in chunks if "| a | b |" in c.text]
    assert len(table_chunks) == 1
    assert "| 3 | 4 |" in table_chunks[0].text


def test_large_section_is_split_by_size_guard():
    big = "# Big\n\n" + ("word " * 300)
    chunks = _chunker().chunk(big)
    assert len(chunks) > 1


def test_page_numbers_best_effort_from_page_texts():
    md = "# Intro\n\nsoccer performance monitoring uses wearable sensors\n"
    page_texts = (
        (1, "soccer performance monitoring uses wearable sensors extensively"),
        (2, "unrelated content about nutrition and recovery protocols"),
    )
    chunks = _chunker().chunk(md, page_texts)
    assert chunks[0].page_numbers == (1,)


def test_prose_without_headers_still_chunks():
    chunks = _chunker().chunk("just some prose text with no markdown headers at all")
    assert len(chunks) >= 1
    assert chunks[0].section_path == ""


def test_page_numbers_empty_when_no_match():
    md = "# Intro\n\ncontent that appears on no page at all here\n"
    page_texts = ((1, "totally different words about hydration"),)
    chunks = _chunker().chunk(md, page_texts)
    assert chunks[0].page_numbers == ()
