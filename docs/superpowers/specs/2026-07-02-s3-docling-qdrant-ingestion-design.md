# Design: S3 → Docling → Qdrant ingestion pipeline

**Date:** 2026-07-02
**Branch:** `feature/s3-qdrant-ingestion`
**Status:** Approved (design), pending spec review

## Goal

Replace all prior ingestion attempts with a single, modular, class-oriented,
document-level, chunk-based RAG ingestion pipeline for scientific papers. It
loads born-digital PDFs from AWS S3, parses them with Docling, chunks them
along heading structure, embeds chunks locally with FastEmbed, and upserts one
Qdrant point per chunk. Per-page image renders are written back to S3 as a
dormant seed for a future visual (ColPali/Qwen-VL) collection. The pipeline is
the **core project**, not a loose script — it lives under `src/`.

### Corpus assumptions

- ~1000 PDFs, born-digital (no scans), processed sequentially, one-shot batch.
- **PDFs only.** No image-object inputs anywhere.

## Repo reshape

**Delete (git-tracked → recoverable from history):**

- `backend/` — outdated FastAPI/Celery/Unstructured attempt.
- `frontend/` — Next.js app; not the project we want.
- `handson/` — outdated hands-on scripts this replaces.

**Remove (untracked, being reimplemented):**

- `scripts/s3_qdrant_ingestion.py`, `scripts/requirements-s3-ingestion.txt`.

**Preserve / relocate:**

- `scripts/bulk_download_papers.py` → `tools/bulk_download_papers.py`
  (paper acquisition — a different concern from ingestion, untracked, too
  useful to lose).
- `assets/`, `papers/`, `examples/`, `docs/` stay.

**New layout:**

```
src/sportsscience_rag/        # the core package (editable install)
tools/bulk_download_papers.py # preserved acquisition helper
tests/                        # unit tests for pure-logic modules
pyproject.toml                # deps + editable install (uv pip install -e .)
.env.example                  # updated for this pipeline
README.md                     # pipeline-stage overview (or block)
```

## Non-negotiable architecture decisions

- **Parser:** Docling (local, layout-aware). Export markdown + structured JSON
  with bboxes. Preserve heading hierarchy, tables, equations. No cloud parser.
- **Triage:** born-digital, so OCR off by default (`do_ocr=False`). Detect and
  log documents that return an empty/near-empty text layer, route them to a
  quarantine list rather than embedding garbage. No full OCR-escalation path —
  flag and skip only.
- **Chunking:** LangChain `MarkdownHeaderTextSplitter` over Docling markdown
  (`#`, `##`, `###`), then a secondary `RecursiveCharacterTextSplitter` size
  guard so no chunk exceeds a token budget. Concrete default: child chunk max
  ~512 tokens with ~64-token overlap, measured with a tokenizer length
  function (not raw characters); values live in `IngestionConfig` and feed
  `chunk_config_hash`. Emit small child chunks for embedding plus a
  parent-section reference. Never split mid-table.
- **Embedding:** local FastEmbed (`sentence-transformers/all-MiniLM-L6-v2`,
  384-dim), vectors computed client-side; upsert raw vectors. No Qdrant
  server-side `models.Document` inference.
- **Idempotency:** point IDs are `uuid5(NAMESPACE, f"{content_hash}:{chunk_index}")`.
  `content_hash` = `sha256` of raw PDF bytes. Re-running is a no-op for
  unchanged files; a document already fully ingested for the current
  `(content_hash, parser_version, chunk_config_hash)` is skipped.
- **Page images:** per-page renders at 150 DPI written back to S3 under a
  derived-artifact prefix
  (`s3://<bucket>/derived/<content_hash>/screenshots/page-<N>.png`). Not
  embedded. Idempotent upload (skip if object exists). Dormant seed for a
  future visual collection — pre-wired, not built.
- **Visual collection:** created nowhere now; a clearly-commented extension
  point marks where a separate image-vector collection would later index the
  S3-persisted renders.

## Module decomposition (`src/sportsscience_rag/`)

One responsibility per file; frozen dataclasses; type hints; Google docstrings.

| Module | Responsibility |
|--------|----------------|
| `config.py` | Frozen `IngestionConfig`; load + validate `.env`; derive `chunk_config_hash`; hold `parser_version` |
| `models.py` | Frozen dataclasses: `RawPdf`, `ParsedDocument`, `PageRender`, `Chunk`, `IngestOutcome`, `QuarantineEntry` |
| `s3_source.py` | `S3Source` — list `.pdf` keys under prefix(es), fetch bytes; `parse_s3_url` |
| `parser.py` | `DoclingParser` — bytes → markdown + structured JSON (bboxes/tables/equations) + 150-DPI `PageRender`s; empty-text detection; OCR off |
| `chunker.py` | `SectionChunker` — markdown-header split → size guard; `section_path`, `chunk_index`, `page_numbers`; no mid-table splits |
| `embedder.py` | `TextEmbedder` — local FastEmbed MiniLM (384-d), client-side vectors |
| `persistence.py` | `RenderStore` — idempotent S3 upload of page PNGs to the derived prefix |
| `qdrant_store.py` | `QdrantStore` — create-if-absent collection, `uuid5` IDs, skip-if-present, upsert raw vectors |
| `hashing.py` | `content_hash` (sha256 of bytes), `parser_version`, `chunk_config_hash` helpers |
| `logging_setup.py` | Structured JSONL per-document logger |
| `pipeline.py` | `IngestionPipeline` — orchestrates stages, per-doc try/except, quarantine, resumability |
| `visual_index.py` | ColPali/Qwen-VL seed — commented `NotImplementedError` stub over S3 renders |
| `cli.py` + `__main__.py` | argparse CLI; `python -m sportsscience_rag` |

## Data flow (per PDF — iterate documents, never prefixes)

`S3Source.list` → for each `.pdf`:

1. fetch bytes → `content_hash = sha256(bytes)`
2. **skip** if already fully ingested for current `(content_hash, parser_version, chunk_config_hash)`
3. `DoclingParser` → markdown + JSON + page renders; empty text → quarantine, continue
4. `RenderStore.upload` renders to S3 (idempotent)
5. `SectionChunker` → chunks
6. `TextEmbedder` → 384-d vectors (client-side)
7. `QdrantStore.upsert` (uuid5 IDs, skip-if-present)
8. structured JSONL log line

Any exception at any step → structured error + append to quarantine list +
continue. One bad file must not abort the batch of 1000.

## Qdrant collection & payload

- Single named vector `text_embedding`, size 384, cosine.
- Create collection only if absent.
- Per-chunk payload: `source` (`s3://…`), `content_hash`, `page_numbers`,
  `section_path` (e.g. `Method > Algorithm 3.2`), `chunk_index`,
  `parser_version`, `chunk_config_hash`, and the chunk `text`.
- Point ID: `uuid5(NAMESPACE, f"{content_hash}:{chunk_index}")`.

## Logging

Structured JSONL per document: `source`, `content_hash`, `stage`,
`duration_ms`, `n_chunks`, `status` (`done` / `quarantined` / `skipped`),
`error_class`.

## CLI surface

- `--prefix` (repeatable), `--collection`, `--dry-run`, `--verbose`.
- `--limit N` — process only first N docs (testing).
- `--quarantine-report PATH` — write failed/skipped list as JSON.
- `--derived-prefix` — S3 key prefix for persisted renders (default `derived/`).
- No `--multimodal`.

## Configuration (`.env`)

Reuses the root `.env`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION`, `S3_BUCKET`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.
`.env.example` updated accordingly. Fail fast at startup if required vars are
missing.

## Dependencies

`boto3`, `python-dotenv`, `docling`, `langchain-text-splitters`, `fastembed`,
`qdrant-client`, `Pillow`. No LangChain document loaders. No server-side
inference. Docling pulls torch + layout models (large first-run download) —
expected.

## Testing

- **Unit (fast, no network):** hashing determinism, chunker heading/size
  behavior + no mid-table split, `parse_s3_url`, point-id determinism,
  config validation, quarantine-report serialization.
- **Live smoke (2 docs):** install deps into `.venv`, discover the S3 prefix
  holding the PDFs, run `--limit 2` against a **throwaway Qdrant Cloud
  collection** and renders under `derived/` in S3. Verify: point count > 0,
  payload shape matches schema, vectors are real 384-d client-side floats
  (not server-side `models.Document`), renders land in S3, and a second run is
  a no-op (idempotency).

## Risks & mitigations

- **Docling weight / first-run download:** expected; install into `.venv`.
- **Live writes to real S3 + Qdrant Cloud:** isolate via throwaway collection
  name and the dedicated `derived/` prefix; nothing else is touched.
- **Permanent loss of untracked files:** `bulk_download_papers.py` is
  preserved (relocated, not deleted); only the reimplemented ingestion script
  is removed.
- **`fast`-strategy empty-parse issue seen previously (Unstructured):**
  mitigated by moving to Docling + explicit empty-text quarantine.

## Out of scope (YAGNI)

- OCR escalation path.
- The visual/ColPali collection itself (only the seed + extension point).
- Retrieval / query side (a later sub-project).
- Any reuse of the deleted backend service.
