# Multi-Tenant Hybrid Ingestion Engine

Production-grade document ingestion and hybrid retrieval backend colocated with Docker Compose.

## Architecture

```
POST /v1/ingest/upload (FastAPI + JWT)
  -> SHA-256 dup check (PostgreSQL + RLS)
  -> Raw PDF -> MinIO
  -> status=PENDING in Postgres
  -> Celery task (RabbitMQ)
       -> Unstructured PDF parse
       -> deterministic chunk IDs
       -> dense + sparse(BM25/IDF) + ColBERT multi embeddings
       -> Qdrant upsert (shared collection, tenant payload index)

POST /v1/search
  -> Stage 1: parallel sparse + dense prefetch (top 20 each, tenant filter)
  -> Stage 2: ColBERT MAX_SIM rerank over unified candidates (top 10)
```

## Stack

| Service | Purpose |
|---------|---------|
| FastAPI | Upload + search REST API |
| PostgreSQL | Document/chunk metadata with Row-Level Security on `tenant_id` |
| RabbitMQ | Celery broker |
| Redis | Celery result backend |
| Celery worker | Parsing, embedding, Qdrant upserts |
| MinIO | S3-compatible raw PDF storage |
| Qdrant | Shared hybrid collection (`dense`, `sparse`, `multi`) |

## Qdrant collection tuning

- Global HNSW disabled: `m=0`
- Payload-scoped graphs: `payload_m=16`
- Tenant keyword index: `tenant_id` with `is_tenant=true`
- ColBERT field: per-vector `hnsw_config.m=0` (rerank-only)
- Sparse field: server-side BM25 with `Modifier.IDF`
- Vectors and payload stored on disk

## Quick start

```bash
cd backend
cp .env.example .env
docker compose up --build
```

Services:

- API: http://localhost:8000
- Qdrant: http://localhost:6333
- MinIO console: http://localhost:9001
- RabbitMQ management: http://localhost:15672

Generate a dev JWT:

```bash
docker compose exec api python scripts/generate_dev_token.py
```

Upload a PDF:

```bash
TOKEN="<jwt>"
curl -X POST http://localhost:8000/v1/ingest/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/paper.pdf"
```

Search:

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"GPS tracking load monitoring in soccer"}'
```

## PDF parsing

PDFs are parsed with [Unstructured](https://github.com/Unstructured-IO/unstructured). Configure via `.env`:

| Variable | Values | Default |
|----------|--------|---------|
| `UNSTRUCTURED_STRATEGY` | `fast`, `hi_res`, `ocr_only`, `auto` | `fast` |
| `UNSTRUCTURED_INFER_TABLES` | `true` / `false` | `false` |

Set `UNSTRUCTURED_INFER_TABLES=true` for table-heavy PDFs (uses `hi_res` strategy).

## Celery reliability

```python
task_acks_late = True
task_reject_on_worker_lost = True
worker_prefetch_multiplier = 1
```

Embedder HTTP 429 retries use exponential backoff with jitter:

`delay = 2^attempt + uniform(0, jitter_bound)`

## Idempotent chunk IDs

Each Qdrant point ID is `sha256(f"{document_id}:{chunk_index}")`, so task retries overwrite vectors instead of duplicating them.

## Worktree

This backend lives on branch `enb` in worktree `SportsScienceRAG-enb`.
