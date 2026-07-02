# S3 → Docling → Qdrant Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, class-oriented, document-level RAG ingestion pipeline that loads PDFs from S3, parses them with Docling, chunks along heading structure, embeds chunks locally with FastEmbed, upserts one Qdrant point per chunk, and writes per-page renders back to S3 — replacing all prior ingestion attempts.

**Architecture:** A `src/sportsscience_rag/` package with one responsibility per module (config, models, s3_source, parser, chunker, embedder, persistence, qdrant_store, hashing, logging_setup, pipeline, visual_index, cli). Documents are iterated one at a time; each flows fetch → hash → skip-if-present → parse → render-upload → chunk → embed → upsert, wrapped in per-document error handling that quarantines failures and continues. Idempotency comes from content-hash-derived `uuid5` point IDs.

**Tech Stack:** Python 3.14, boto3, Docling, langchain-text-splitters, fastembed (all-MiniLM-L6-v2, 384-d), qdrant-client, Pillow, python-dotenv. Tooling: uv, pytest.

## Global Constraints

- **PDFs only.** No image-object input code paths anywhere.
- **Parser:** Docling, local, `do_ocr=False`. Export markdown + structured content. Preserve headings/tables/equations. No cloud parser.
- **Embedding:** local FastEmbed `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, client-side vectors. No Qdrant server-side `models.Document` inference.
- **Point ID:** `uuid5(NAMESPACE, f"{content_hash}:{chunk_index}")`; `content_hash = sha256(raw_pdf_bytes)` hex.
- **Chunking:** `MarkdownHeaderTextSplitter` on `#`/`##`/`###` → `RecursiveCharacterTextSplitter` size guard. Default child chunk ≤ **256 tokens, 32-token overlap**, measured by a **real tokenizer length function** built from the MiniLM tokenizer (`sentence-transformers/all-MiniLM-L6-v2`, which truncates at 256 tokens) — injectable so unit tests stay fast. Never split mid-table.
- **Page provenance:** best-effort. The parser emits per-page text; the chunker attributes `page_numbers` to each chunk by matching its normalized leading text against per-page text. Unmatched → empty tuple.
- **Qdrant collection:** single named vector `text_embedding`, size 384, cosine. Create only if absent.
- **Payload per chunk:** `source`, `content_hash`, `page_numbers`, `section_path`, `chunk_index`, `parser_version`, `chunk_config_hash`, `text`.
- **Page images:** 150 DPI renders → `s3://<bucket>/<derived_prefix><content_hash>/screenshots/page-<N>.png`; idempotent upload (skip if exists); never embedded.
- **Idempotency:** skip a document already fully ingested for current `(content_hash, parser_version, chunk_config_hash)`. Re-run = no-op, no duplicates.
- **Robustness:** per-document try/except → structured error + quarantine list; one bad file must not abort the batch.
- **Logging:** structured JSONL per document: `source`, `content_hash`, `stage`, `duration_ms`, `n_chunks`, `status` (`done`/`quarantined`/`skipped`), `error_class`.
- **CLI:** `--prefix` (repeatable), `--collection`, `--dry-run`, `-v/--verbose`, `--limit N`, `--quarantine-report PATH`, `--derived-prefix` (default `derived/`). No `--multimodal`.
- **Style:** PEP8, type hints, Google docstrings, frozen dataclasses, files < 800 lines.
- **Install:** `uv pip install` only (never plain pip). Activate project root `.venv` first.

---

### Task 1: Repo reshape + package scaffold + dependency install

**Files:**
- Delete: `backend/`, `frontend/`, `handson/`
- Move: `scripts/bulk_download_papers.py` → `tools/bulk_download_papers.py`
- Remove: `scripts/s3_qdrant_ingestion.py`, `scripts/requirements-s3-ingestion.txt`, `scripts/__pycache__/`, root `__pycache__/`
- Create: `src/sportsscience_rag/__init__.py`, `pyproject.toml`, `.env.example`, `tests/__init__.py`, `tests/conftest.py`

**Interfaces:**
- Produces: importable package `sportsscience_rag` (editable install); `PACKAGE_VERSION` in `__init__.py`.

- [ ] **Step 1: Confirm the correct venv, then delete/move dirs**

```bash
cd /Users/soccerment/Desktop/Projects/SportsScienceRAG
source .venv/bin/activate
python --version   # expect Python 3.14.x

# git-tracked deletions (recoverable from history)
git rm -r backend frontend handson

# preserve acquisition helper, drop reimplemented ingestion script
mkdir -p tools
git mv scripts/bulk_download_papers.py tools/bulk_download_papers.py 2>/dev/null || mv scripts/bulk_download_papers.py tools/bulk_download_papers.py
rm -rf scripts __pycache__ assets/__pycache__
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "sportsscience-rag"
version = "0.1.0"
description = "S3 -> Docling -> Qdrant ingestion pipeline for sports-science papers"
requires-python = ">=3.12"
dependencies = [
    "boto3>=1.35",
    "python-dotenv>=1.0",
    "docling>=2.0",
    "langchain-text-splitters>=0.3",
    "fastembed>=0.4",
    "qdrant-client>=1.12",
    "Pillow>=10.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = [
    "integration: touches models/network/heavy deps (deselect with -m 'not integration')",
]
```

- [ ] **Step 3: Create `src/sportsscience_rag/__init__.py`**

```python
"""S3 -> Docling -> Qdrant ingestion pipeline for sports-science papers."""

PACKAGE_VERSION = "0.1.0"
```

- [ ] **Step 4: Create `.env.example`**

```bash
# AWS S3 — source PDFs and derived page renders
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=eu-west-1
S3_BUCKET=your-bucket

# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key
QDRANT_COLLECTION=sport-science-documents
```

- [ ] **Step 5: Create `tests/__init__.py` (empty) and `tests/conftest.py`**

```python
"""Shared test fixtures."""

import io

import pytest
from PIL import Image


@pytest.fixture
def one_px_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 6: Install dependencies (Docling is the risk on py3.14)**

Run:
```bash
uv pip install docling
uv pip install -e ".[dev]"
python -c "import docling, langchain_text_splitters, fastembed, qdrant_client, boto3, PIL; print('imports OK')"
python -c "from docling.document_converter import DocumentConverter; print('docling OK')"
```
Expected: `imports OK` then `docling OK`. If Docling fails to build on 3.14, STOP and report — do not proceed; the parser task depends on it.

- [ ] **Step 7: Verify package import**

Run: `python -c "import sportsscience_rag as p; print(p.PACKAGE_VERSION)"`
Expected: `0.1.0`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: remove backend/frontend/handson, scaffold sportsscience_rag package"
```

---

### Task 2: Foundation — `hashing.py`, `models.py`, `config.py`

**Files:**
- Create: `src/sportsscience_rag/hashing.py`, `src/sportsscience_rag/models.py`, `src/sportsscience_rag/config.py`
- Test: `tests/test_hashing.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `hashing.content_hash(data: bytes) -> str` (sha256 hex)
  - `hashing.chunk_config_hash(chunk_size: int, chunk_overlap: int, headers: tuple[tuple[str, str], ...]) -> str`
  - `models.RawPdf(source: str, key: str, data: bytes, content_hash: str)` (frozen)
  - `models.PageRender(page_number: int, image: PIL.Image.Image)` (frozen, eq=False)
  - `models.Chunk(index: int, text: str, section_path: str, page_numbers: tuple[int, ...])` (frozen)
  - `models.ParsedDocument(markdown: str, page_count: int, renders: tuple[PageRender, ...], is_empty: bool, page_texts: tuple[tuple[int, str], ...])` (frozen, eq=False) — `page_texts` is `(page_no, normalized_page_text)` for best-effort page mapping
  - `models.QuarantineEntry(source: str, content_hash: str, stage: str, error_class: str, message: str)` (frozen)
  - `models.IngestOutcome(source: str, content_hash: str, status: str, n_chunks: int)` (frozen)
  - `config.IngestionConfig` (frozen) with `from_env(env_path: Path | None = None) -> IngestionConfig` and property `chunk_config_hash: str`

- [ ] **Step 1: Write failing tests for hashing**

`tests/test_hashing.py`:
```python
from sportsscience_rag.hashing import content_hash, chunk_config_hash


def test_content_hash_is_deterministic_and_hex():
    h1 = content_hash(b"hello world")
    h2 = content_hash(b"hello world")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_content_hash_differs_on_different_bytes():
    assert content_hash(b"a") != content_hash(b"b")


def test_chunk_config_hash_stable_and_sensitive():
    headers = (("#", "h1"), ("##", "h2"))
    base = chunk_config_hash(512, 64, headers)
    assert base == chunk_config_hash(512, 64, headers)
    assert base != chunk_config_hash(256, 64, headers)
    assert base != chunk_config_hash(512, 64, (("#", "h1"),))
    assert len(base) == 16
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_hashing.py -v`
Expected: FAIL (`ModuleNotFoundError: sportsscience_rag.hashing`)

- [ ] **Step 3: Implement `hashing.py`**

```python
"""Content and configuration hashing helpers."""

import hashlib


def content_hash(data: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def chunk_config_hash(
    chunk_size: int,
    chunk_overlap: int,
    headers: tuple[tuple[str, str], ...],
) -> str:
    """Return a short stable hash of the chunking configuration."""
    header_repr = "|".join(f"{marker}:{name}" for marker, name in headers)
    payload = f"{chunk_size}:{chunk_overlap}:{header_repr}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 4: Implement `models.py`**

```python
"""Immutable data structures passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

from PIL.Image import Image


@dataclass(frozen=True)
class RawPdf:
    source: str
    key: str
    data: bytes
    content_hash: str


@dataclass(frozen=True, eq=False)
class PageRender:
    page_number: int
    image: Image


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section_path: str
    page_numbers: tuple[int, ...]


@dataclass(frozen=True, eq=False)
class ParsedDocument:
    markdown: str
    page_count: int
    renders: tuple[PageRender, ...]
    is_empty: bool
    page_texts: tuple[tuple[int, str], ...]  # (page_no, normalized text) for page mapping


@dataclass(frozen=True)
class QuarantineEntry:
    source: str
    content_hash: str
    stage: str
    error_class: str
    message: str


@dataclass(frozen=True)
class IngestOutcome:
    source: str
    content_hash: str
    status: str  # "done" | "quarantined" | "skipped"
    n_chunks: int
```

- [ ] **Step 5: Write failing tests for config**

`tests/test_config.py`:
```python
import pytest

from sportsscience_rag.config import IngestionConfig


ENV = {
    "AWS_ACCESS_KEY_ID": "ak",
    "AWS_SECRET_ACCESS_KEY": "sk",
    "AWS_REGION": "eu-west-1",
    "S3_BUCKET": "bkt",
    "QDRANT_URL": "https://q",
    "QDRANT_API_KEY": "qk",
    "QDRANT_COLLECTION": "coll",
}


def test_from_env_reads_values(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = IngestionConfig.from_env(env_path=None)
    assert cfg.s3_bucket == "bkt"
    assert cfg.qdrant_collection == "coll"
    assert cfg.image_dpi == 150
    assert cfg.derived_prefix == "derived/"


def test_from_env_raises_on_missing(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="Missing required"):
        IngestionConfig.from_env(env_path=None)


def test_chunk_config_hash_is_property(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = IngestionConfig.from_env(env_path=None)
    assert len(cfg.chunk_config_hash) == 16
```

- [ ] **Step 6: Run config tests to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: sportsscience_rag.config`)

- [ ] **Step 7: Implement `config.py`**

```python
"""Environment-driven, validated ingestion configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from sportsscience_rag.hashing import chunk_config_hash

DEFAULT_HEADERS: tuple[tuple[str, str], ...] = (
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
)
_REQUIRED = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "S3_BUCKET",
    "QDRANT_URL",
    "QDRANT_API_KEY",
)


@dataclass(frozen=True)
class IngestionConfig:
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    s3_bucket: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "sport-science-documents"
    chunk_size: int = 256   # MiniLM truncates at 256 tokens
    chunk_overlap: int = 32
    headers: tuple[tuple[str, str], ...] = field(default=DEFAULT_HEADERS)
    image_dpi: int = 150
    derived_prefix: str = "derived/"

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "IngestionConfig":
        if env_path is not None:
            load_dotenv(env_path)
        values = {key: os.getenv(key, "") for key in _REQUIRED}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise ValueError("Missing required environment variables: " + ", ".join(missing))
        return cls(
            aws_access_key_id=values["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=values["AWS_SECRET_ACCESS_KEY"],
            aws_region=values["AWS_REGION"],
            s3_bucket=values["S3_BUCKET"],
            qdrant_url=values["QDRANT_URL"],
            qdrant_api_key=values["QDRANT_API_KEY"],
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "sport-science-documents"),
        )

    @property
    def chunk_config_hash(self) -> str:
        return chunk_config_hash(self.chunk_size, self.chunk_overlap, self.headers)
```

- [ ] **Step 8: Run all foundation tests**

Run: `pytest tests/test_hashing.py tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add src/sportsscience_rag/hashing.py src/sportsscience_rag/models.py src/sportsscience_rag/config.py tests/test_hashing.py tests/test_config.py
git commit -m "feat: config, models, and hashing foundation"
```

---

### Task 3: `s3_source.py` — list & fetch PDFs

**Files:**
- Create: `src/sportsscience_rag/s3_source.py`
- Test: `tests/test_s3_source.py`

**Interfaces:**
- Consumes: nothing from prior tasks (boto3 client injected).
- Produces:
  - `s3_source.parse_s3_url(url: str) -> tuple[str, str]` → `(bucket, key)`
  - `s3_source.S3Source(client, bucket: str)` with:
    - `list_pdfs(prefixes: Sequence[str], limit: int | None = None) -> list[tuple[str, str]]` → list of `(key, source_url)`, `.pdf` only, folder keys skipped
    - `fetch(key: str) -> bytes`

- [ ] **Step 1: Write failing tests**

`tests/test_s3_source.py`:
```python
from unittest.mock import MagicMock

from sportsscience_rag.s3_source import S3Source, parse_s3_url


def test_parse_s3_url():
    assert parse_s3_url("s3://bkt/a/b.pdf") == ("bkt", "a/b.pdf")
    assert parse_s3_url("s3://bkt") == ("bkt", "")


def _paginator_with(keys):
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    return paginator


def test_list_pdfs_filters_and_builds_urls():
    client = MagicMock()
    client.get_paginator.return_value = _paginator_with(
        ["papers/a.pdf", "papers/b.PDF", "papers/notes.txt", "papers/"]
    )
    src = S3Source(client, "bkt")
    result = src.list_pdfs(["papers/"])
    assert result == [
        ("papers/a.pdf", "s3://bkt/papers/a.pdf"),
        ("papers/b.PDF", "s3://bkt/papers/b.PDF"),
    ]


def test_list_pdfs_respects_limit():
    client = MagicMock()
    client.get_paginator.return_value = _paginator_with(["a.pdf", "b.pdf", "c.pdf"])
    src = S3Source(client, "bkt")
    assert len(src.list_pdfs([""], limit=2)) == 2


def test_fetch_reads_body():
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = b"PDFBYTES"
    client.get_object.return_value = {"Body": body}
    src = S3Source(client, "bkt")
    assert src.fetch("a.pdf") == b"PDFBYTES"
    client.get_object.assert_called_once_with(Bucket="bkt", Key="a.pdf")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_s3_source.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `s3_source.py`**

```python
"""List and fetch PDF objects from S3."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def parse_s3_url(url: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    parts = url.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


class S3Source:
    """Enumerates and downloads ``.pdf`` objects under given prefixes."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def list_pdfs(
        self,
        prefixes: Sequence[str],
        limit: int | None = None,
    ) -> list[tuple[str, str]]:
        """Return ``(key, source_url)`` for each PDF, capped at ``limit``."""
        found: list[tuple[str, str]] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for prefix in prefixes:
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/") or not key.lower().endswith(".pdf"):
                        continue
                    found.append((key, f"s3://{self._bucket}/{key}"))
                    if limit is not None and len(found) >= limit:
                        return found
        return found

    def fetch(self, key: str) -> bytes:
        """Download and return the raw bytes of an object."""
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        return response["Body"].read()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_s3_source.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/s3_source.py tests/test_s3_source.py
git commit -m "feat: S3 PDF listing and fetching"
```

---

### Task 4: `chunker.py` — heading-aware chunking with size guard

**Files:**
- Create: `src/sportsscience_rag/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: `config.IngestionConfig` (uses `headers`, `chunk_size`, `chunk_overlap`); produces `models.Chunk`.
- Produces:
  - `chunker.SectionChunker(config: IngestionConfig, token_length: Callable[[str], int] | None = None)` — `token_length` counts tokens; when `None`, lazily built from the MiniLM HF tokenizer. Tests inject a fast fake (e.g. word count).
  - `.chunk(markdown: str, page_texts: tuple[tuple[int, str], ...] = ()) -> list[Chunk]`
  - Behavior: split on markdown headers into sections; each section's text further split by a token-length `RecursiveCharacterTextSplitter` size guard; `section_path` joins active header values with `" > "`; a markdown table block (lines starting with `|`) is never split across chunks; `index` is a 0-based running counter across the document. `page_numbers` is best-effort: match the chunk's normalized leading text against each `(page_no, page_text)`; attribute every page whose text contains that signature; `()` if none match or `page_texts` is empty.

- [ ] **Step 1: Write failing tests**

`tests/test_chunker.py`:
```python
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


def test_sections_carry_hierarchical_path():
    chunks = _chunker().chunk(MD)
    paths = {c.section_path for c in chunks}
    assert "Introduction" in paths
    assert "Methods > Algorithm 3.2" in paths


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


def test_page_numbers_empty_when_no_match():
    md = "# Intro\n\ncontent that appears on no page at all here\n"
    page_texts = ((1, "totally different words about hydration"),)
    chunks = _chunker().chunk(md, page_texts)
    assert chunks[0].page_numbers == ()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_chunker.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `chunker.py`**

```python
"""Heading-aware markdown chunking with a token-budget size guard."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.models import Chunk

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_NORMALIZE = re.compile(r"[^a-z0-9 ]+")
_SIGNATURE_WORDS = 6  # leading words used to locate a chunk on a page

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _default_token_length() -> Callable[[str], int]:
    """Build a token-count function from the MiniLM tokenizer (lazy import)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)

    def _length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return _length


def _normalize(text: str) -> str:
    return _NORMALIZE.sub(" ", text.lower()).strip()


class SectionChunker:
    """Splits Docling markdown into heading-scoped, token-bounded chunks."""

    def __init__(
        self,
        config: IngestionConfig,
        token_length: Callable[[str], int] | None = None,
    ) -> None:
        self._headers = list(config.headers)
        self._header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers,
            strip_headers=True,
        )
        length_function = token_length or _default_token_length()
        self._size_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(
        self,
        markdown: str,
        page_texts: tuple[tuple[int, str], ...] = (),
    ) -> list[Chunk]:
        normalized_pages = [(no, _normalize(text)) for no, text in page_texts]
        chunks: list[Chunk] = []
        index = 0
        for section in self._header_splitter.split_text(markdown):
            section_path = " > ".join(
                section.metadata[name]
                for _, name in self._headers
                if name in section.metadata
            )
            for text in self._split_preserving_tables(section.page_content):
                cleaned = text.strip()
                if not cleaned:
                    continue
                chunks.append(
                    Chunk(
                        index=index,
                        text=cleaned,
                        section_path=section_path,
                        page_numbers=self._pages_for(cleaned, normalized_pages),
                    )
                )
                index += 1
        return chunks

    def _pages_for(
        self, chunk_text: str, normalized_pages: list[tuple[int, str]]
    ) -> tuple[int, ...]:
        if not normalized_pages:
            return ()
        signature = " ".join(_normalize(chunk_text).split()[:_SIGNATURE_WORDS])
        if not signature:
            return ()
        return tuple(no for no, page in normalized_pages if signature in page)

    def _split_preserving_tables(self, text: str) -> list[str]:
        """Split text by token budget but keep contiguous table blocks intact."""
        parts: list[str] = []
        buffer: list[str] = []
        in_table = False
        for line in text.splitlines():
            is_table = bool(_TABLE_LINE.match(line))
            if is_table and not in_table:
                parts.extend(self._flush_prose("\n".join(buffer)))
                buffer = [line]
                in_table = True
            elif is_table:
                buffer.append(line)
            elif in_table:
                parts.append("\n".join(buffer))  # emit whole table as one chunk
                buffer = [line]
                in_table = False
            else:
                buffer.append(line)
        if in_table:
            parts.append("\n".join(buffer))
        else:
            parts.extend(self._flush_prose("\n".join(buffer)))
        return parts

    def _flush_prose(self, prose: str) -> list[str]:
        if not prose.strip():
            return []
        return self._size_splitter.split_text(prose)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_chunker.py -v`
Expected: all PASS. If `MarkdownHeaderTextSplitter` metadata keys differ in the installed version, adjust `section.metadata` key access to match (`split_text` returns `Document` objects with `.page_content` and `.metadata`).

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/chunker.py tests/test_chunker.py
git commit -m "feat: heading-aware chunker with table-preserving size guard"
```

---

### Task 5: `embedder.py` — local FastEmbed vectors

**Files:**
- Create: `src/sportsscience_rag/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Produces:
  - `embedder.TextEmbedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2")`
  - `.dimension: int` (384)
  - `.embed(texts: list[str]) -> list[list[float]]` — one 384-float vector per input, order preserved

- [ ] **Step 1: Write failing test (integration — loads the model)**

`tests/test_embedder.py`:
```python
import pytest

from sportsscience_rag.embedder import TextEmbedder


@pytest.mark.integration
def test_embed_returns_384d_vectors():
    emb = TextEmbedder()
    vectors = emb.embed(["hello world", "second document"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
    assert emb.dimension == 384
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `embedder.py`**

```python
"""Local dense text embedding via FastEmbed."""

from __future__ import annotations

from fastembed import TextEmbedding

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


class TextEmbedder:
    """Computes 384-dim MiniLM vectors client-side."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self.dimension = DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]
```

- [ ] **Step 4: Run test to verify pass (first run downloads the model)**

Run: `pytest tests/test_embedder.py -v -m integration`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/embedder.py tests/test_embedder.py
git commit -m "feat: local FastEmbed MiniLM text embedder"
```

---

### Task 6: `persistence.py` — idempotent page-render upload to S3

**Files:**
- Create: `src/sportsscience_rag/persistence.py`
- Test: `tests/test_persistence.py`

**Interfaces:**
- Consumes: `models.PageRender`.
- Produces:
  - `persistence.RenderStore(client, bucket: str, derived_prefix: str = "derived/")`
  - `.render_key(content_hash: str, page_number: int) -> str` → `"<derived_prefix><hash>/screenshots/page-<N>.png"`
  - `.upload(content_hash: str, renders: Sequence[PageRender]) -> list[str]` — uploads PNGs, skips objects that already exist (via `head_object`), returns all keys

- [ ] **Step 1: Write failing tests**

`tests/test_persistence.py`:
```python
from unittest.mock import MagicMock

from botocore.exceptions import ClientError
from PIL import Image

from sportsscience_rag.models import PageRender
from sportsscience_rag.persistence import RenderStore


def _renders(n):
    return [PageRender(page_number=i + 1, image=Image.new("RGB", (2, 2))) for i in range(n)]


def _not_found():
    return ClientError({"Error": {"Code": "404"}}, "HeadObject")


def test_render_key_format():
    store = RenderStore(MagicMock(), "bkt", "derived/")
    assert store.render_key("abc", 3) == "derived/abc/screenshots/page-3.png"


def test_upload_puts_missing_objects():
    client = MagicMock()
    client.head_object.side_effect = _not_found()
    store = RenderStore(client, "bkt", "derived/")
    keys = store.upload("abc", _renders(2))
    assert keys == ["derived/abc/screenshots/page-1.png", "derived/abc/screenshots/page-2.png"]
    assert client.put_object.call_count == 2


def test_upload_skips_existing_objects():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 10}  # exists
    store = RenderStore(client, "bkt", "derived/")
    store.upload("abc", _renders(2))
    client.put_object.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_persistence.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `persistence.py`**

```python
"""Idempotent persistence of per-page renders to S3."""

from __future__ import annotations

import io
from collections.abc import Sequence
from typing import Any

from botocore.exceptions import ClientError

from sportsscience_rag.models import PageRender


class RenderStore:
    """Writes page PNGs to a derived-artifact S3 prefix, skipping duplicates."""

    def __init__(self, client: Any, bucket: str, derived_prefix: str = "derived/") -> None:
        self._client = client
        self._bucket = bucket
        self._prefix = derived_prefix

    def render_key(self, content_hash: str, page_number: int) -> str:
        return f"{self._prefix}{content_hash}/screenshots/page-{page_number}.png"

    def _exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    def upload(self, content_hash: str, renders: Sequence[PageRender]) -> list[str]:
        keys: list[str] = []
        for render in renders:
            key = self.render_key(content_hash, render.page_number)
            keys.append(key)
            if self._exists(key):
                continue
            buffer = io.BytesIO()
            render.image.save(buffer, format="PNG")
            buffer.seek(0)
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=buffer.getvalue(),
                ContentType="image/png",
            )
        return keys
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_persistence.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/persistence.py tests/test_persistence.py
git commit -m "feat: idempotent S3 page-render persistence"
```

---

### Task 7: `qdrant_store.py` — collection, IDs, skip-if-present, upsert

**Files:**
- Create: `src/sportsscience_rag/qdrant_store.py`
- Test: `tests/test_qdrant_store.py`

**Interfaces:**
- Consumes: `models.Chunk`.
- Produces:
  - `qdrant_store.NAMESPACE` (a fixed `uuid.UUID`)
  - `qdrant_store.point_id(content_hash: str, chunk_index: int) -> str`
  - `qdrant_store.QdrantStore(client, collection: str, dimension: int = 384)` with:
    - `.ensure_collection() -> None` — create only if absent, single named vector `text_embedding`
    - `.already_ingested(content_hash: str, parser_version: str, chunk_config_hash: str) -> bool`
    - `.upsert(content_hash: str, source: str, parser_version: str, chunk_config_hash: str, chunks: list[Chunk], vectors: list[list[float]]) -> int` — builds points, returns count

- [ ] **Step 1: Write failing tests**

`tests/test_qdrant_store.py`:
```python
from unittest.mock import MagicMock

from sportsscience_rag.models import Chunk
from sportsscience_rag.qdrant_store import QdrantStore, point_id


def test_point_id_is_deterministic_uuid():
    a = point_id("hash123", 0)
    b = point_id("hash123", 0)
    c = point_id("hash123", 1)
    assert a == b
    assert a != c
    assert len(a) == 36  # uuid string


def test_ensure_collection_creates_when_absent():
    client = MagicMock()
    client.collection_exists.return_value = False
    QdrantStore(client, "coll").ensure_collection()
    client.create_collection.assert_called_once()


def test_ensure_collection_skips_when_present():
    client = MagicMock()
    client.collection_exists.return_value = True
    QdrantStore(client, "coll").ensure_collection()
    client.create_collection.assert_not_called()


def test_already_ingested_true_when_points_found():
    client = MagicMock()
    client.count.return_value = MagicMock(count=3)
    assert QdrantStore(client, "coll").already_ingested("h", "pv", "cc") is True


def test_already_ingested_false_when_zero():
    client = MagicMock()
    client.count.return_value = MagicMock(count=0)
    assert QdrantStore(client, "coll").already_ingested("h", "pv", "cc") is False


def test_upsert_builds_points_with_payload():
    client = MagicMock()
    store = QdrantStore(client, "coll")
    chunks = [Chunk(index=0, text="t0", section_path="Intro", page_numbers=(1,))]
    n = store.upsert("h", "s3://b/k.pdf", "pv", "cc", chunks, [[0.1] * 384])
    assert n == 1
    args, kwargs = client.upsert.call_args
    points = kwargs["points"]
    assert points[0].payload["content_hash"] == "h"
    assert points[0].payload["section_path"] == "Intro"
    assert points[0].payload["text"] == "t0"
    assert "text_embedding" in points[0].vector
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_qdrant_store.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `qdrant_store.py`**

```python
"""Qdrant collection management and idempotent chunk upsert."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import models

from sportsscience_rag.models import Chunk

NAMESPACE = uuid.UUID("6f9b1d2a-3c4e-5a6b-8c9d-0e1f2a3b4c5d")
VECTOR_NAME = "text_embedding"


def point_id(content_hash: str, chunk_index: int) -> str:
    """Deterministic point ID from content hash + chunk index."""
    return str(uuid.uuid5(NAMESPACE, f"{content_hash}:{chunk_index}"))


class QdrantStore:
    """Creates the collection and upserts one point per chunk."""

    def __init__(self, client: Any, collection: str, dimension: int = 384) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension

    def ensure_collection(self) -> None:
        if self._client.collection_exists(self._collection):
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance.COSINE,
                )
            },
        )

    def _match_filter(
        self, content_hash: str, parser_version: str, chunk_config_hash: str
    ) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="content_hash", match=models.MatchValue(value=content_hash)),
                models.FieldCondition(key="parser_version", match=models.MatchValue(value=parser_version)),
                models.FieldCondition(key="chunk_config_hash", match=models.MatchValue(value=chunk_config_hash)),
            ]
        )

    def already_ingested(
        self, content_hash: str, parser_version: str, chunk_config_hash: str
    ) -> bool:
        result = self._client.count(
            collection_name=self._collection,
            count_filter=self._match_filter(content_hash, parser_version, chunk_config_hash),
            exact=True,
        )
        return result.count > 0

    def upsert(
        self,
        content_hash: str,
        source: str,
        parser_version: str,
        chunk_config_hash: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        points = [
            models.PointStruct(
                id=point_id(content_hash, chunk.index),
                vector={VECTOR_NAME: vector},
                payload={
                    "source": source,
                    "content_hash": content_hash,
                    "page_numbers": list(chunk.page_numbers),
                    "section_path": chunk.section_path,
                    "chunk_index": chunk.index,
                    "parser_version": parser_version,
                    "chunk_config_hash": chunk_config_hash,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)
        return len(points)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_qdrant_store.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/qdrant_store.py tests/test_qdrant_store.py
git commit -m "feat: Qdrant store with uuid5 IDs and idempotent upsert"
```

---

### Task 8: `parser.py` — Docling parse + page renders + empty detection

**Files:**
- Create: `src/sportsscience_rag/parser.py`
- Test: `tests/test_parser.py`

**Interfaces:**
- Consumes: `models.ParsedDocument`, `models.PageRender`.
- Produces:
  - `parser.PARSER_VERSION: str` (e.g. `"docling-<version>"`)
  - `parser.DoclingParser(image_dpi: int = 150, min_chars: int = 20)` with `.parse(data: bytes, name: str) -> ParsedDocument`
  - Behavior: OCR off; export markdown; generate per-page images at `image_dpi`; build `page_texts` = `(page_no, concatenated text of items on that page)` from the structured document for best-effort page mapping; `is_empty=True` when stripped markdown length < `min_chars`.

- [ ] **Step 1: Write failing/integration tests**

`tests/test_parser.py`:
```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_parser.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `parser.py`**

```python
"""Docling-based PDF parsing with page rendering and empty-text triage."""

from __future__ import annotations

import io

import docling
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from sportsscience_rag.models import PageRender, ParsedDocument

PARSER_VERSION = f"docling-{getattr(docling, '__version__', 'unknown')}"


class DoclingParser:
    """Converts PDF bytes to markdown + page renders (OCR disabled)."""

    def __init__(self, image_dpi: int = 150, min_chars: int = 20) -> None:
        self._min_chars = min_chars
        options = PdfPipelineOptions()
        options.do_ocr = False
        options.generate_page_images = True
        options.images_scale = image_dpi / 72.0  # Docling scale is relative to 72 DPI
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse(self, data: bytes, name: str) -> ParsedDocument:
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
        is_empty = len(markdown.strip()) < self._min_chars
        return ParsedDocument(
            markdown=markdown,
            page_count=len(pages),
            renders=renders,
            is_empty=is_empty,
            page_texts=self._page_texts(document),
        )

    @staticmethod
    def _page_texts(document) -> tuple[tuple[int, str], ...]:
        """Concatenate each text item's text under the page number it came from."""
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_parser.py -v -m integration`
Expected: PASS. If the installed Docling version exposes a different attribute for page numbers or images (e.g. `page.page_no` vs `page.page_number`, or `page.image.pil_image` vs `page.image`), adjust those two access points to match — run `python -c "from docling.document_converter import DocumentConverter; r=DocumentConverter().convert('assets/<one>.pdf'); p=list(r.document.pages.values())[0]; print(dir(p)); print(type(p.image))"` to inspect, then fix.

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/parser.py tests/test_parser.py
git commit -m "feat: Docling PDF parser with page renders and empty triage"
```

---

### Task 9: `logging_setup.py` — structured JSONL logging

**Files:**
- Create: `src/sportsscience_rag/logging_setup.py`
- Test: `tests/test_logging_setup.py`

**Interfaces:**
- Produces:
  - `logging_setup.JsonlLogger(path: Path | None = None, stream=None)` with `.event(**fields) -> dict` — writes one JSON object per line and returns the dict written.

- [ ] **Step 1: Write failing tests**

`tests/test_logging_setup.py`:
```python
import io
import json

from sportsscience_rag.logging_setup import JsonlLogger


def test_event_writes_one_json_line():
    buf = io.StringIO()
    logger = JsonlLogger(stream=buf)
    written = logger.event(
        source="s3://b/k.pdf", content_hash="h", stage="upsert",
        duration_ms=12, n_chunks=5, status="done", error_class=None,
    )
    line = buf.getvalue().strip()
    parsed = json.loads(line)
    assert parsed["status"] == "done"
    assert parsed["n_chunks"] == 5
    assert written == parsed
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_logging_setup.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `logging_setup.py`**

```python
"""Structured JSONL logging for per-document ingestion events."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO


class JsonlLogger:
    """Writes one JSON object per line to a file or stream."""

    def __init__(self, path: Path | None = None, stream: TextIO | None = None) -> None:
        if stream is not None:
            self._stream = stream
        elif path is not None:
            self._stream = open(path, "a", encoding="utf-8")  # noqa: SIM115
        else:
            self._stream = sys.stdout

    def event(self, **fields: Any) -> dict[str, Any]:
        self._stream.write(json.dumps(fields, default=str) + "\n")
        self._stream.flush()
        return fields
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_logging_setup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/logging_setup.py tests/test_logging_setup.py
git commit -m "feat: structured JSONL logger"
```

---

### Task 10: `visual_index.py` — dormant ColPali seed

**Files:**
- Create: `src/sportsscience_rag/visual_index.py`
- Test: `tests/test_visual_index.py`

**Interfaces:**
- Produces: `visual_index.VisualIndexPlaceholder` with `.index_from_renders(*args, **kwargs) -> None` raising `NotImplementedError`.

- [ ] **Step 1: Write failing test**

`tests/test_visual_index.py`:
```python
import pytest

from sportsscience_rag.visual_index import VisualIndexPlaceholder


def test_visual_index_is_not_implemented():
    with pytest.raises(NotImplementedError):
        VisualIndexPlaceholder().index_from_renders("derived/abc/screenshots/")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_visual_index.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `visual_index.py`**

```python
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
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_visual_index.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/visual_index.py tests/test_visual_index.py
git commit -m "feat: dormant ColPali visual-index extension point"
```

---

### Task 11: `pipeline.py` — orchestration, quarantine, resumability

**Files:**
- Create: `src/sportsscience_rag/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `IngestionConfig`, `S3Source`, `DoclingParser`, `SectionChunker`, `TextEmbedder`, `RenderStore`, `QdrantStore`, `JsonlLogger`, `PARSER_VERSION`, and all `models`.
- Produces:
  - `pipeline.IngestionResult(outcomes: list[IngestOutcome], quarantine: list[QuarantineEntry])` (frozen)
  - `pipeline.IngestionPipeline(config, source, parser, chunker, embedder, render_store, store, logger, parser_version=PARSER_VERSION)` with `.run(prefixes: list[str], limit: int | None = None, dry_run: bool = False) -> IngestionResult`
  - Behavior: iterate documents; skip if `store.already_ingested`; parse; empty → quarantine; upload renders; chunk; embed; upsert; per-doc try/except → quarantine + continue; `dry_run` lists PDFs and returns outcomes with status `skipped` and no writes.

- [ ] **Step 1: Write failing tests (all deps mocked)**

`tests/test_pipeline.py`:
```python
from unittest.mock import MagicMock

from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.models import Chunk, ParsedDocument
from sportsscience_rag.pipeline import IngestionPipeline

CFG = IngestionConfig(
    aws_access_key_id="x", aws_secret_access_key="x", aws_region="x",
    s3_bucket="bkt", qdrant_url="x", qdrant_api_key="x",
)


def _pipeline(**overrides):
    defaults = dict(
        config=CFG,
        source=MagicMock(),
        parser=MagicMock(),
        chunker=MagicMock(),
        embedder=MagicMock(),
        render_store=MagicMock(),
        store=MagicMock(),
        logger=MagicMock(),
        parser_version="docling-test",
    )
    defaults.update(overrides)
    return IngestionPipeline(**defaults), defaults


def test_happy_path_upserts_and_reports_done():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = False
    d["parser"].parse.return_value = ParsedDocument("# H\n\ntext", 1, (), False, ())
    d["chunker"].chunk.return_value = [Chunk(0, "text", "H", ())]
    d["embedder"].embed.return_value = [[0.0] * 384]
    d["store"].upsert.return_value = 1

    result = p.run([""])
    assert result.outcomes[0].status == "done"
    assert result.outcomes[0].n_chunks == 1
    assert not result.quarantine
    d["store"].upsert.assert_called_once()


def test_skips_already_ingested():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = True

    result = p.run([""])
    assert result.outcomes[0].status == "skipped"
    d["parser"].parse.assert_not_called()


def test_empty_document_is_quarantined():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = False
    d["parser"].parse.return_value = ParsedDocument("", 1, (), True, ())

    result = p.run([""])
    assert result.outcomes[0].status == "quarantined"
    assert result.quarantine[0].stage == "parse"
    d["store"].upsert.assert_not_called()


def test_exception_quarantines_and_continues():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("a.pdf", "s3://bkt/a.pdf"), ("b.pdf", "s3://bkt/b.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = False
    d["parser"].parse.side_effect = [RuntimeError("corrupt"), ParsedDocument("# H\n\nt", 1, (), False, ())]
    d["chunker"].chunk.return_value = [Chunk(0, "t", "H", ())]
    d["embedder"].embed.return_value = [[0.0] * 384]
    d["store"].upsert.return_value = 1

    result = p.run([""])
    statuses = {o.source: o.status for o in result.outcomes}
    assert statuses["s3://bkt/a.pdf"] == "quarantined"
    assert statuses["s3://bkt/b.pdf"] == "done"
    assert result.quarantine[0].error_class == "RuntimeError"


def test_dry_run_writes_nothing():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    result = p.run([""], dry_run=True)
    assert result.outcomes[0].status == "skipped"
    d["source"].fetch.assert_not_called()
    d["store"].upsert.assert_not_called()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `pipeline.py`**

```python
"""Document-level ingestion orchestration with quarantine and resumability."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.hashing import content_hash
from sportsscience_rag.models import IngestOutcome, QuarantineEntry


@dataclass(frozen=True)
class IngestionResult:
    outcomes: list[IngestOutcome]
    quarantine: list[QuarantineEntry]


class IngestionPipeline:
    """Runs the fetch → hash → skip → parse → render → chunk → embed → upsert flow."""

    def __init__(
        self,
        config: IngestionConfig,
        source,
        parser,
        chunker,
        embedder,
        render_store,
        store,
        logger,
        parser_version: str,
    ) -> None:
        self._config = config
        self._source = source
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._render_store = render_store
        self._store = store
        self._logger = logger
        self._parser_version = parser_version

    def run(
        self,
        prefixes: list[str],
        limit: int | None = None,
        dry_run: bool = False,
    ) -> IngestionResult:
        outcomes: list[IngestOutcome] = []
        quarantine: list[QuarantineEntry] = []
        pdfs = self._source.list_pdfs(prefixes, limit=limit)

        if not dry_run:
            self._store.ensure_collection()

        for key, source_url in pdfs:
            if dry_run:
                outcomes.append(IngestOutcome(source_url, "", "skipped", 0))
                self._logger.event(source=source_url, content_hash="", stage="list",
                                    duration_ms=0, n_chunks=0, status="skipped", error_class=None)
                continue
            outcome, entry = self._process_one(key, source_url)
            outcomes.append(outcome)
            if entry is not None:
                quarantine.append(entry)

        return IngestionResult(outcomes=outcomes, quarantine=quarantine)

    def _process_one(self, key: str, source_url: str):
        start = time.monotonic()
        chash = ""
        stage = "fetch"
        try:
            data = self._source.fetch(key)
            chash = content_hash(data)

            stage = "skip-check"
            if self._store.already_ingested(chash, self._parser_version, self._config.chunk_config_hash):
                return self._done(source_url, chash, "skipped", 0, start), None

            stage = "parse"
            parsed = self._parser.parse(data, key)
            if parsed.is_empty:
                return (
                    self._done(source_url, chash, "quarantined", 0, start),
                    QuarantineEntry(source_url, chash, "parse", "EmptyText", "empty text layer"),
                )

            stage = "render-upload"
            self._render_store.upload(chash, parsed.renders)

            stage = "chunk"
            chunks = self._chunker.chunk(parsed.markdown, parsed.page_texts)
            if not chunks:
                return (
                    self._done(source_url, chash, "quarantined", 0, start),
                    QuarantineEntry(source_url, chash, "chunk", "NoChunks", "no chunks produced"),
                )

            stage = "embed"
            vectors = self._embedder.embed([c.text for c in chunks])

            stage = "upsert"
            n = self._store.upsert(
                chash, source_url, self._parser_version,
                self._config.chunk_config_hash, chunks, vectors,
            )
            return self._done(source_url, chash, "done", n, start), None
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the batch
            entry = QuarantineEntry(source_url, chash, stage, type(exc).__name__, str(exc))
            return self._done(source_url, chash, "quarantined", 0, start, type(exc).__name__), entry

    def _done(self, source, chash, status, n_chunks, start, error_class=None) -> IngestOutcome:
        self._logger.event(
            source=source, content_hash=chash, stage=status,
            duration_ms=int((time.monotonic() - start) * 1000),
            n_chunks=n_chunks, status=status, error_class=error_class,
        )
        return IngestOutcome(source=source, content_hash=chash, status=status, n_chunks=n_chunks)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsscience_rag/pipeline.py tests/test_pipeline.py
git commit -m "feat: ingestion pipeline orchestration with quarantine"
```

---

### Task 12: `cli.py` + `__main__.py` — command-line surface + wiring

**Files:**
- Create: `src/sportsscience_rag/cli.py`, `src/sportsscience_rag/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: all modules; builds real boto3 + Qdrant clients only when not dry-run.
- Produces:
  - `cli.build_arg_parser() -> argparse.ArgumentParser`
  - `cli.main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing tests (arg parsing only — no network)**

`tests/test_cli.py`:
```python
from sportsscience_rag.cli import build_arg_parser


def test_defaults():
    args = build_arg_parser().parse_args([])
    assert args.prefixes is None
    assert args.dry_run is False
    assert args.limit is None
    assert args.derived_prefix == "derived/"


def test_repeatable_prefix_and_flags():
    args = build_arg_parser().parse_args(
        ["--prefix", "a/", "--prefix", "b/", "--collection", "c",
         "--limit", "2", "--dry-run", "-v", "--quarantine-report", "q.json"]
    )
    assert args.prefixes == ["a/", "b/"]
    assert args.collection == "c"
    assert args.limit == 2
    assert args.dry_run is True
    assert args.verbose is True
    assert args.quarantine_report == "q.json"


def test_no_multimodal_flag():
    parser = build_arg_parser()
    # --multimodal must not exist anymore
    import pytest
    with pytest.raises(SystemExit):
        parser.parse_args(["--multimodal"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `cli.py`**

```python
"""Command-line entry point for the ingestion pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import boto3
from qdrant_client import QdrantClient

from sportsscience_rag.chunker import SectionChunker
from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.embedder import TextEmbedder
from sportsscience_rag.logging_setup import JsonlLogger
from sportsscience_rag.parser import PARSER_VERSION, DoclingParser
from sportsscience_rag.persistence import RenderStore
from sportsscience_rag.pipeline import IngestionPipeline
from sportsscience_rag.qdrant_store import QdrantStore
from sportsscience_rag.s3_source import S3Source

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest S3 PDFs into Qdrant via Docling + FastEmbed.",
    )
    parser.add_argument("--prefix", action="append", dest="prefixes", metavar="FOLDER",
                        help="S3 prefix to ingest (repeatable). Defaults to bucket root.")
    parser.add_argument("--collection", help="Qdrant collection (overrides .env).")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N PDFs.")
    parser.add_argument("--quarantine-report", dest="quarantine_report",
                        help="Write failed/skipped list to this JSON path.")
    parser.add_argument("--derived-prefix", dest="derived_prefix", default="derived/",
                        help="S3 key prefix for persisted page renders.")
    parser.add_argument("--dry-run", action="store_true", help="List PDFs, write nothing.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    config = IngestionConfig.from_env(env_path=PROJECT_ROOT / ".env")
    collection = args.collection or config.qdrant_collection
    prefixes = args.prefixes or [""]

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region,
    )
    source = S3Source(s3_client, config.s3_bucket)

    if args.dry_run:
        pipeline = IngestionPipeline(
            config, source, parser=None, chunker=None, embedder=None,
            render_store=None, store=_DryStore(), logger=JsonlLogger(),
            parser_version=PARSER_VERSION,
        )
        result = pipeline.run(prefixes, limit=args.limit, dry_run=True)
        logger.info("Dry run: %d PDFs found.", len(result.outcomes))
        _write_quarantine(args, result)
        return 0

    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    pipeline = IngestionPipeline(
        config=config,
        source=source,
        parser=DoclingParser(image_dpi=config.image_dpi),
        chunker=SectionChunker(config),
        embedder=TextEmbedder(),
        render_store=RenderStore(s3_client, config.s3_bucket, args.derived_prefix),
        store=QdrantStore(qdrant_client, collection),
        logger=JsonlLogger(),
        parser_version=PARSER_VERSION,
    )
    result = pipeline.run(prefixes, limit=args.limit)
    done = sum(1 for o in result.outcomes if o.status == "done")
    logger.info("Ingestion finished: %d done, %d quarantined, %d skipped.",
                done,
                len(result.quarantine),
                sum(1 for o in result.outcomes if o.status == "skipped"))
    _write_quarantine(args, result)
    return 0


class _DryStore:
    """No-op store so dry-run needs no Qdrant connection."""

    def ensure_collection(self) -> None:  # pragma: no cover - trivial
        pass


def _write_quarantine(args, result) -> None:
    if not args.quarantine_report:
        return
    payload = [vars(entry) for entry in result.quarantine]
    Path(args.quarantine_report).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote quarantine report (%d entries) to %s",
                len(payload), args.quarantine_report)
```

- [ ] **Step 4: Implement `__main__.py`**

```python
"""``python -m sportsscience_rag`` entry point."""

import sys

from sportsscience_rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run CLI tests to verify pass**

Run: `pytest tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 6: Verify dry-run wiring end to end (no Qdrant, real S3 listing)**

Run: `python -m sportsscience_rag --dry-run --limit 3 -v`
Expected: logs `Dry run: N PDFs found.` (N ≥ 0) with no traceback. If credentials/prefix are wrong it will raise clearly — note the actual PDF prefix for Task 14.

- [ ] **Step 7: Commit**

```bash
git add src/sportsscience_rag/cli.py src/sportsscience_rag/__main__.py tests/test_cli.py
git commit -m "feat: CLI surface and pipeline wiring"
```

---

### Task 13: README + full unit-test/coverage gate

**Files:**
- Create: `README.md`
- Test: (runs the whole suite)

**Interfaces:** none new.

- [ ] **Step 1: Write `README.md`**

```markdown
# SportsScience RAG — S3 → Docling → Qdrant ingestion

Document-level, chunk-based ingestion of scientific-paper PDFs from S3 into a
Qdrant vector collection, with per-page renders persisted back to S3.

## Pipeline stages

`S3Source` (list/fetch PDFs) → `DoclingParser` (markdown + 150-DPI page renders,
OCR off, empty-text triage) → `RenderStore` (idempotent page-PNG upload to
`derived/<hash>/screenshots/`) → `SectionChunker` (heading-aware split + size
guard, tables kept intact) → `TextEmbedder` (local FastEmbed MiniLM, 384-d) →
`QdrantStore` (one point per chunk, `uuid5` content-hash IDs, skip-if-present).
`IngestionPipeline` iterates documents with per-document quarantine so one bad
PDF never aborts the batch. Visual/ColPali indexing is seeded (renders in S3)
but not built — see `visual_index.py`.

## Setup

    source .venv/bin/activate
    uv pip install -e ".[dev]"
    cp .env.example .env   # fill in AWS + Qdrant values

## Run

    python -m sportsscience_rag --dry-run --limit 3          # list only
    python -m sportsscience_rag --prefix papers/ --limit 2   # ingest 2
    python -m sportsscience_rag --collection my-test --quarantine-report q.json

## Test

    pytest -m "not integration"     # fast unit tests
    pytest                          # includes model/PDF integration tests
```

- [ ] **Step 2: Run full unit suite (no heavy integration)**

Run: `pytest -m "not integration" -v`
Expected: all PASS.

- [ ] **Step 3: Run integration suite (loads model + parses one PDF)**

Run: `pytest -m integration -v`
Expected: PASS (embedder + parser).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: pipeline README"
```

---

### Task 14: Live smoke test — 2 documents end to end

**Files:** none (verification task); may create a throwaway `q.json`.

**Interfaces:** none new.

- [ ] **Step 1: Discover the real PDF prefix**

Run: `python -m sportsscience_rag --dry-run -v --limit 5`
If it lists 0 PDFs at the root, probe prefixes:
```bash
python -c "
import boto3, os
from dotenv import load_dotenv
load_dotenv('.env')
c = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
r = c.list_objects_v2(Bucket=os.getenv('S3_BUCKET'), MaxKeys=20)
for o in r.get('Contents', []): print(o['Key'])
"
```
Note the prefix that holds `.pdf` objects; use it as `--prefix` below.

- [ ] **Step 2: Ingest 2 documents into a throwaway collection**

Run:
```bash
python -m sportsscience_rag --prefix <DISCOVERED_PREFIX> --limit 2 \
    --collection smoke-test-ingest --quarantine-report q.json -v
```
Expected: logs `Ingestion finished: 2 done, 0 quarantined, 0 skipped.` (or documents quarantined with a clear reason).

- [ ] **Step 3: Verify Qdrant points, payload, and 384-d vectors**

Run:
```bash
python -c "
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
load_dotenv('.env')
c = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
print('count:', c.count('smoke-test-ingest', exact=True).count)
pts = c.scroll('smoke-test-ingest', limit=1, with_payload=True, with_vectors=True)[0]
p = pts[0]
print('payload keys:', sorted(p.payload.keys()))
print('vector dim:', len(p.vector['text_embedding']))
print('section_path:', p.payload['section_path'])
"
```
Expected: `count` > 0; payload keys include `source, content_hash, page_numbers, section_path, chunk_index, parser_version, chunk_config_hash, text`; `vector dim: 384`.

- [ ] **Step 4: Verify page renders landed in S3**

Run:
```bash
python -c "
import boto3, os
from dotenv import load_dotenv
load_dotenv('.env')
c = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
r = c.list_objects_v2(Bucket=os.getenv('S3_BUCKET'), Prefix='derived/')
print('render objects:', r.get('KeyCount', 0))
for o in r.get('Contents', [])[:3]: print(o['Key'])
"
```
Expected: ≥ 1 object under `derived/…/screenshots/page-*.png`.

- [ ] **Step 5: Verify idempotency — rerun is a no-op**

Run:
```bash
python -m sportsscience_rag --prefix <DISCOVERED_PREFIX> --limit 2 --collection smoke-test-ingest -v
```
Expected: `2 skipped` (both already ingested), count unchanged in Step 3's query.

- [ ] **Step 6: Report results**

Summarize: docs ingested, point count, vector dim, render count, idempotency result, any quarantined docs with reasons. Leave the throwaway collection in place or delete it per user preference.

---

## Self-Review

**Spec coverage:** PDFs-only (Tasks 3,1) ✓; Docling OCR-off + markdown + renders + empty triage (Task 8) ✓; heading chunking + size guard + no mid-table (Task 4) ✓; local FastEmbed 384-d client-side, no server inference (Tasks 5,7) ✓; uuid5 content-hash IDs (Task 7) ✓; skip-if-present + resumability (Tasks 7,11) ✓; 150-DPI renders to S3 idempotently (Tasks 6,8) ✓; visual seed only (Task 10) ✓; per-doc quarantine + continue (Task 11) ✓; JSONL logging (Task 9,11) ✓; full CLI surface incl. `--limit`, `--quarantine-report`, `--derived-prefix`, no `--multimodal` (Task 12) ✓; single named vector + payload schema (Task 7) ✓; `.env`/config validation (Task 2) ✓; repo reshape + preserve `bulk_download_papers.py` (Task 1) ✓; live 2-doc verification (Task 14) ✓.

**Placeholder scan:** No TBD/TODO; every code step shows full code; version-sensitive Docling access points (Task 8 Step 4) and langchain metadata (Task 4 Step 4) carry concrete inspection commands rather than vague "adjust as needed."

**Type consistency:** `content_hash`/`chunk_config_hash` signatures consistent across Tasks 2/7/11; `Chunk(index,text,section_path,page_numbers)` consistent Tasks 4/7/11; `ParsedDocument(markdown,page_count,renders,is_empty)` consistent Tasks 2/8/11; `QdrantStore` method names (`ensure_collection`, `already_ingested`, `upsert`) consistent Tasks 7/11/12; `point_id` and `VECTOR_NAME="text_embedding"` consistent.
