# Infrastructure — current state

Canonical reference for the running infrastructure on the `development` branch.
The repository hosts **two independent retrieval systems**; this document maps
both and the services they depend on.

- **Core engine** (`sports_science_search/` + `examples/`) — single-process,
  research-corpus search over `assets/` against a single Qdrant collection.
- **Hybrid ingestion backend** (`backend/` + `frontend/`) — multi-tenant,
  Docker-Compose-orchestrated ingestion + hybrid retrieval service.

## Hybrid ingestion backend

```mermaid
flowchart TB
    subgraph client["Client"]
        FE["Next.js 16 / React 19<br/>Document Workspace<br/>upload · chat · citations"]
    end

    subgraph apilayer["API layer — FastAPI :8000 (JWT + tenant_id)"]
        UP["POST /v1/ingest/upload"]
        SR["POST /v1/search<br/>2-stage hybrid"]
        EV["SSE progress stream"]
    end

    subgraph asyncp["Async pipeline"]
        MQ["RabbitMQ :5672<br/>Celery broker"]
        WK["Celery worker<br/>parse → chunk → embed → upsert"]
        RD["Redis :6379<br/>result backend"]
        UNS["Unstructured<br/>PDF parse"]
        subgraph emb["Embedders"]
            DEN["Dense · all-MiniLM-L6-v2 (384d)"]
            SPA["Sparse · Qdrant/bm25 (IDF)"]
            COL["ColBERT · colbertv2.0 (128d)"]
        end
    end

    subgraph stores["Data stores"]
        PG[("PostgreSQL :5432<br/>doc/chunk metadata<br/>RLS on tenant_id")]
        S3[("MinIO / S3 :9000<br/>raw PDFs · ingestion-raw")]
        QD[("Qdrant :6333<br/>hybrid_ingestion<br/>dense · sparse · multi<br/>tenant-scoped HNSW")]
    end

    FE -->|upload PDF| UP
    UP -->|SHA-256 dedup + metadata| PG
    UP -->|store raw PDF| S3
    UP -->|enqueue task| MQ
    MQ --> WK
    WK -->|status / results| RD
    WK -->|fetch PDF| S3
    WK --> UNS
    UNS --> emb
    WK -->|upsert vectors| QD
    WK -->|status updates| PG
    WK -.->|progress events| EV
    EV -.->|live updates| FE

    FE -->|query + document_ids + JWT| SR
    SR -->|"Stage 1: parallel sparse + dense prefetch (tenant filter)"| QD
    SR -->|"Stage 2: ColBERT MAX_SIM rerank"| QD
    SR -->|ranked results| FE
```

### Services (from `backend/docker-compose.yml`)

| Service | Image | Ports | Role |
|---------|-------|-------|------|
| `api` | local `Dockerfile` (uvicorn) | 8000 | FastAPI upload + search |
| `worker` | local `Dockerfile` (celery, concurrency=2) | — | Parse / embed / upsert |
| `postgres` | `postgres:16-alpine` | 5432 | Metadata + RLS on `tenant_id` |
| `rabbitmq` | `rabbitmq:3.13-management-alpine` | 5672, 15672 | Celery broker (`amqp://`) |
| `redis` | `redis:7-alpine` | 6379 | Celery result backend |
| `minio` | `minio/minio` | 9000, 9001 | S3-compatible raw PDF store |
| `minio-init` | `minio/mc` | — | Creates `ingestion-raw` bucket |
| `qdrant` | `qdrant/qdrant:v1.13.4` | 6333, 6334 | Hybrid vector collection |
| `qdrant-init` | local `Dockerfile` | — | Runs `python -m app.qdrant.setup` |

### Key configuration (from `backend/app/config.py`)

| Setting | Default |
|---------|---------|
| `database_url` | `postgresql+asyncpg://ingest:***@localhost:5432/ingestion` |
| `database_url_sync` | `postgresql+psycopg2://…` (Celery worker) |
| `celery_broker_url` | `amqp://guest:guest@localhost:5672//` |
| `celery_result_backend` | `redis://localhost:6379/0` |
| `qdrant_collection` | `hybrid_ingestion` |
| `s3_bucket` / `s3_region` | `ingestion-raw` / `us-east-1` |
| `dense_model` / `dense_dimension` | `all-MiniLM-L6-v2` / 384 |
| `colbert_model` / `colbert_dimension` | `colbert-ir/colbertv2.0` / 128 |
| `UNSTRUCTURED_STRATEGY` / `UNSTRUCTURED_INFER_TABLES` | `fast` / `false` |

### Qdrant collection (`backend/app/qdrant/setup.py`)

- Named vectors: `dense` (cosine, 384), `sparse` (IDF modifier), `multi`
  (cosine, 128, `MultiVectorComparator.MAX_SIM`, `hnsw_config.m=0`).
- Global `hnsw_config.m=0`, `payload_m=16` → payload-scoped graphs only.
- `tenant_id` keyword payload index with `is_tenant=true`.
- Idempotent point IDs: `sha256(f"{document_id}:{chunk_index}")`.

## Core engine (standalone)

Independent of the backend stack — no Postgres/Celery/MinIO. Talks directly to
a Qdrant instance (Qdrant Cloud via `.env`, or in-memory for the `demo` command).

```mermaid
flowchart LR
    PDFs["assets/<br/>27 PDFs"] --> EXT["PDFExtractor<br/>PyMuPDF"]
    EXT --> CHK["TextChunker<br/>semantic · paragraph · fixed"]
    CHK --> VS["VectorStore<br/>all-MiniLM-L6-v2 (384d)"]
    VS --> QC[("Qdrant<br/>one collection,<br/>3 named vectors")]
    Q["query"] --> VS
    VS -->|search per strategy| RES["SearchResult comparison"]
```

| Module | Responsibility |
|--------|----------------|
| `pdf_extractor.py` | `PDFExtractor` — PyMuPDF text + section detection + metadata |
| `text_chunker.py` | `TextChunker` — semantic / paragraph / fixed chunking |
| `vector_store.py` | `VectorStore` — Qdrant collection, upload, search, stats |
| `search_engine.py` | `SemanticSearchEngine` — orchestrates the pipeline |

> Note: the **core engine uses PyMuPDF**; the **backend uses Unstructured**.
> They are separate parsers in separate pipelines — this is intentional, not a
> migration artifact.
