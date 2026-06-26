# Sports Science Semantic Search Engine

A semantic search engine over a corpus of 27 sports-science research papers
(PDFs in `assets/`). It extracts text and metadata from each PDF, chunks the
text using **three strategies**, embeds the chunks with Sentence Transformers,
stores them in **Qdrant** (one named vector per strategy), and lets you search
and compare the strategies side by side.

## Architecture

The code lives in the `sports_science_search/` package — one responsibility per
module:

| Module | Responsibility |
|--------|----------------|
| `constants.py` | Section patterns, chunking params, Qdrant/embedding config |
| `exceptions.py` | `PDFExtractionError`, `SectionDetectionError`, `VectorUploadError`, `PDFProcessingError` |
| `models.py` | Immutable dataclasses: `Paper`, `PaperChunk`, `SearchResult` |
| `pdf_extractor.py` | `PDFExtractor` — PyMuPDF text extraction, section detection, metadata |
| `text_chunker.py` | `TextChunker` — semantic / paragraph / fixed-size chunking |
| `vector_store.py` | `VectorStore` — Qdrant collection, upload, search, stats |
| `search_engine.py` | `SemanticSearchEngine` — orchestrates the full pipeline |

```
PDF -> PDFExtractor -> Paper
Paper -> TextChunker -> [PaperChunk] (semantic + paragraph + fixed)
[PaperChunk] -> VectorStore.upload_chunks -> Qdrant (named vectors)
query -> VectorStore.search(strategy) -> [SearchResult]
```

### Chunking strategies

| Strategy | How it splits | Named vector |
|----------|---------------|--------------|
| `semantic` | llama-index `SemanticSplitterNodeParser` (buffer=1, threshold=95) | `semantic` |
| `paragraph` | double-newline paragraphs (falls back to single newline) | `paragraph` |
| `fixed` | 200-word windows with 50-word overlap | `fixed` |

All three use `all-MiniLM-L6-v2` (384-dim, cosine distance).

## Setup

```bash
source .venv/bin/activate          # project virtualenv
# dependencies are already installed; for a fresh env use:
#   uv pip install pymupdf sentence-transformers llama-index \
#       llama-index-embeddings-huggingface qdrant-client python-dotenv
```

Create a `.env` with your Qdrant Cloud credentials (already present in this repo):

```
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io
QDRANT_API_KEY=<your-key>
```

## Usage (CLI)

The runnable entry point is `examples/run_search_demo.py`.

```bash
# 1. One-time ingest of all 27 PDFs into Qdrant Cloud (recreate from scratch)
PYTHONPATH=. python examples/run_search_demo.py ingest --recreate

# 2. Query — searches all three strategies and prints a comparison
PYTHONPATH=. python examples/run_search_demo.py search "high intensity interval training"
PYTHONPATH=. python examples/run_search_demo.py search "GPS tracking accuracy" --limit 5 --section Methods

# 3. Inspect the collection
PYTHONPATH=. python examples/run_search_demo.py stats
PYTHONPATH=. python examples/run_search_demo.py analyze

# Export a search result to JSON
PYTHONPATH=. python examples/run_search_demo.py search "muscle fatigue" --export out.json
```

### Local, no-cloud run

For a fully self-contained run on an in-memory Qdrant (ingest + one search in a
single process — nothing is persisted):

```bash
PYTHONPATH=. python examples/run_search_demo.py demo --query "muscle fatigue"
```

## Usage (library)

```python
from pathlib import Path
from qdrant_client import QdrantClient
from sports_science_search import SemanticSearchEngine

engine = SemanticSearchEngine(
    pdf_dir=Path("assets"),
    qdrant_client=QdrantClient(":memory:"),   # or QdrantClient(url=..., api_key=...)
)
engine.process_papers(recreate_collection=True)

results = engine.search_and_compare("acceleration load monitoring", limit=3)
print(results["analysis"]["top_strategy"])
```

## Notes

- The first run downloads the `all-MiniLM-L6-v2` model (~200 MB) and caches it
  in `~/.cache/huggingface`.
- Section detection auto-detects via regex; papers that fail auto-detection can
  be supplied a manual mapping JSON (`manual_mappings_path`).
- The legacy single-file module `12_handson_sportscience_semantic_engine.py` is
  now a backwards-compatibility shim that re-exports the package.

## Known limitations / technical debt

- `VectorStore.upload_chunks` encodes chunks one at a time and uses sequential
  integer point IDs — batch encoding and UUID IDs are pending optimizations.
- `avg_chunk_size` in the analysis output is an approximation.
