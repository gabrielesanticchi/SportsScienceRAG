# SportsScience RAG — S3 → Docling → Qdrant ingestion

Document-level, chunk-based ingestion of scientific-paper PDFs from AWS S3 into
a Qdrant vector collection, with per-page renders persisted back to S3. Built as
a modular, class-oriented package under `src/sportsscience_rag/`.

## Pipeline stages

`S3Source` (list/fetch `.pdf` objects) → `DoclingParser` (layout-aware markdown +
150-DPI per-page renders, OCR off, empty-text triage, per-page text for page
mapping) → `RenderStore` (idempotent page-PNG upload to
`s3://<bucket>/derived/<content_hash>/screenshots/page-<N>.png`) → `SectionChunker`
(heading-aware split on `#/##/###` → 256-token size guard using the MiniLM
tokenizer, tables kept intact, absolute `section_path` provenance, best-effort
`page_numbers`) → `TextEmbedder` (local FastEmbed `all-MiniLM-L6-v2`, 384-dim,
client-side vectors) → `QdrantStore` (one point per chunk, `uuid5(content_hash:chunk_index)`
IDs, skip-if-present).

`IngestionPipeline` iterates **documents** one at a time with per-document
try/except: any failure (corrupt PDF, empty text layer, parse/embed/upsert error)
is recorded as a structured quarantine entry and the batch continues. Idempotency
comes from content-hash-derived point IDs plus a skip check on
`(content_hash, parser_version, chunk_config_hash)`, so re-running is a no-op for
already-ingested files. Visual (ColPali/Qwen-VL) indexing is **seeded** — page
renders live in S3 — but not built; see `visual_index.py` for the extension point.

## Setup

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"       # deps incl. docling, fastembed, qdrant-client
cp .env.example .env             # fill in AWS + Qdrant Cloud values
```

Required `.env` keys: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`,
`S3_BUCKET`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`.

## Run

```bash
# List PDFs only — no writes
python -m sportsscience_rag --dry-run --limit 5

# Ingest the first 2 PDFs under a prefix into a named collection
python -m sportsscience_rag --prefix papers/ --limit 2 \
    --collection my-test --quarantine-report quarantine.json

# Full batch (all prefixes / bucket root)
python -m sportsscience_rag --prefix papers/
```

CLI flags: `--prefix` (repeatable), `--collection`, `--limit N`,
`--quarantine-report PATH`, `--derived-prefix` (default `derived/`), `--dry-run`,
`-v/--verbose`.

### Operational notes

- **Docling is CPU/GPU-bound** (~15–40s per PDF on Apple MPS). The full ~1000-paper
  corpus is a multi-hour, one-shot batch — run it detached (`nohup`/`tmux`).
- Run **one convert at a time** (the pipeline already does). Concurrent Docling
  converts can deadlock on Hugging Face model locks.
- Once Docling/FastEmbed models are cached, set `HF_HUB_OFFLINE=1` to skip HF
  network round-trips (faster, avoids rate-limit stalls).

## Test

```bash
pytest -m "not integration"      # fast unit tests (mocked S3/Qdrant, fake tokenizer)
HF_HUB_OFFLINE=1 pytest -m integration   # loads the embedding model + parses a real PDF
pytest --cov=sportsscience_rag --cov-report=term-missing
```

## Layout

```
src/sportsscience_rag/
  config.py         IngestionConfig (env-driven, validated) + chunk_config_hash
  models.py         frozen dataclasses (RawPdf, PageRender, Chunk, ParsedDocument, …)
  hashing.py        sha256 content hash + chunk-config hash
  s3_source.py      S3Source — list/fetch PDFs
  parser.py         DoclingParser — markdown + renders + page_texts (OCR off)
  chunker.py        SectionChunker — heading split + token size guard
  embedder.py       TextEmbedder — FastEmbed MiniLM (384-d)
  persistence.py    RenderStore — idempotent S3 page-render upload
  qdrant_store.py   QdrantStore — collection, uuid5 IDs, skip-if-present, upsert
  logging_setup.py  JsonlLogger — structured per-document JSONL events
  pipeline.py       IngestionPipeline — orchestration, quarantine, resumability
  visual_index.py   dormant ColPali/Qwen-VL extension point (not built)
  cli.py            argparse surface + wiring
  __main__.py       python -m sportsscience_rag
tools/bulk_download_papers.py   paper-acquisition helper (separate concern)
```
