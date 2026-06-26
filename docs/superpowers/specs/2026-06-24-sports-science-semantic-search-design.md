# Sports Science Semantic Search Engine - Design Specification

**Date:** 2026-06-24  
**Author:** Claude Code + User  
**Project:** Qdrant Day 1 Hands-On Project  
**Target File:** `/Users/soccerment/Desktop/Projects/qdrant/12_handson_sportscience_semantic_engine.py`

---

## 1. Overview

Build a semantic search engine for 27 sports science PDF papers following OOP and DRY principles. The system will extract text from PDFs, chunk content using three different strategies, vectorize with sentence transformers, store in Qdrant, and provide advanced search with comparative analysis.

### Goals
- Convert PDFs to searchable vector embeddings automatically
- Compare three chunking strategies to identify optimal approach for academic papers
- Preserve paper structure (sections, metadata) for filtered search
- Provide advanced search analytics (relevance scores, section distribution, strategy comparison)

### Non-Goals
- Image/figure extraction from PDFs
- OCR for scanned documents (assumes text-based PDFs)
- Real-time updates or web interface
- Multi-language support

---

## 2. Architecture

### Component Diagram
```
PDFs (assets/) 
    ↓
PDFExtractor → Paper
    ↓
TextChunker → PaperChunk[] (3 strategies)
    ↓
SentenceTransformer → Vectors
    ↓
VectorStore → Qdrant Collection
    ↓
SemanticSearchEngine → SearchResults + Analysis
```

### Core Components

**PDFExtractor**
- Responsibility: Extract text and metadata from academic PDFs
- Technology: PyMuPDF (fitz)
- Outputs: `Paper` dataclass

**TextChunker**
- Responsibility: Implement three chunking strategies
- Strategies: Semantic (llama-index), Paragraph (double newline), Fixed-size (200 words, 50 overlap)
- Outputs: `PaperChunk[]` with metadata inheritance

**VectorStore**
- Responsibility: Manage Qdrant operations (create, upload, search)
- Technology: Qdrant Cloud, `qdrant-client`
- Collection: `sports_science_papers` with three named vectors

**SemanticSearchEngine**
- Responsibility: Orchestrate pipeline and search operations
- Role: Main entry point for processing and querying

---

## 3. Data Models

All data models are immutable (`@dataclass(frozen=True)`) following Python best practices.

### Paper
```python
@dataclass(frozen=True)
class Paper:
    filename: str                    # e.g., "malone2017.pdf"
    title: str                       # Extracted from PDF metadata or first page
    text: str                        # Full paper text
    sections: Dict[str, str]         # {"Abstract": "...", "Methods": "..."}
    metadata: Dict[str, Any]         # {authors, year, DOI, journal, page_count}
```

### PaperChunk
```python
@dataclass(frozen=True)
class PaperChunk:
    paper_filename: str              # Source paper
    paper_title: str                 # For display in search results
    section: str                     # Abstract, Methods, Results, etc.
    chunk_text: str                  # Actual content
    chunk_index: int                 # Position within paper (0-indexed)
    page_number: int                 # Original page in PDF
    metadata: Dict[str, Any]         # Inherited from Paper (authors, year, DOI, journal)
    chunk_strategy: str              # "semantic", "paragraph", "fixed"
```

### SearchResult
```python
@dataclass(frozen=True)
class SearchResult:
    paper_filename: str
    paper_title: str
    section: str
    chunk_text: str
    score: float                     # Cosine similarity score
    page_number: int
    metadata: Dict[str, Any]
    chunk_strategy: str
```

---

## 4. Processing Pipeline

### Stage 1: PDF Extraction

**Input:** 27 PDF files in `/Users/soccerment/Desktop/Projects/qdrant/assets/`

**Process:**
1. Load PDF with PyMuPDF
2. Extract full text (ignore images/figures)
3. Auto-detect sections using regex patterns (see below)
4. If auto-detection fails → check `section_mappings.json` for manual mapping
5. If no manual mapping exists → **FAIL FAST** with descriptive error
6. Extract metadata from PDF properties and text parsing

**Section Detection Patterns:**
```python
patterns = {
    "Abstract": r"(?i)^abstract\s*$",
    "Introduction": r"(?i)^(introduction|1\.?\s*introduction)$",
    "Methods": r"(?i)^(methods?|methodology|materials?\s+and\s+methods?)$",
    "Results": r"(?i)^(results?|results?\s+and\s+discussion)$",
    "Discussion": r"(?i)^(discussion|conclusions?)$",
    "References": r"(?i)^(references?|bibliography|literature\s+cited)$"
}
```

**Manual Section Mapping (`section_mappings.json`):**
```json
{
  "malone2017.pdf": {
    "Abstract": {"start_page": 1, "end_page": 1},
    "Introduction": {"start_page": 1, "end_page": 2},
    "Methods": {"start_page": 2, "end_page": 4},
    "Results": {"start_page": 4, "end_page": 6},
    "Discussion": {"start_page": 6, "end_page": 7},
    "References": {"start_page": 7, "end_page": 8}
  }
}
```

**Metadata Extraction:**
- `title`: PDF metadata or extracted from first page
- `authors`: Parsed from first page (best-effort)
- `year`: Extracted from filename pattern or text
- `DOI`: Regex search for DOI pattern
- `journal`: Parsed from header/footer if available
- `page_count`: From PDF properties

**Output:** `Paper` object with validated sections

---

### Stage 2: Chunking

**Input:** `Paper` object

**Process:** Apply all three strategies in parallel

**Strategy 1: Semantic Chunking**
- Use `SemanticSplitterNodeParser` from llama-index
- Respects semantic boundaries (doesn't split mid-concept)
- Optimal for academic papers with complex ideas
- Configuration: `buffer_size=1`, `breakpoint_percentile_threshold=95`

**Strategy 2: Paragraph Chunking**
- Split on double newlines (`\n\n`)
- Preserves natural paragraph boundaries in papers
- Simpler, more predictable than semantic
- Fallback: if no double newlines, split on single newlines

**Strategy 3: Fixed-Size Chunking**
- 200 words per chunk, 50-word overlap
- Baseline comparison strategy
- Ensures uniform chunk sizes across papers

**Metadata Inheritance:**
Each chunk inherits:
- Paper metadata (authors, year, DOI, journal)
- Section label (from section detection)
- Page number (tracked during extraction)
- Chunk index (position within paper)
- Chunk strategy identifier

**Output:** `List[PaperChunk]` (3x chunks per paper)

---

### Stage 3: Vectorization & Upload

**Input:** `List[PaperChunk]`

**Process:**
1. Encode `chunk_text` with `SentenceTransformer("all-MiniLM-L6-v2")` → 384-dim vector
2. Create Qdrant `PointStruct` with three named vectors:
   - `semantic`: vector if chunk_strategy == "semantic" else None
   - `paragraph`: vector if chunk_strategy == "paragraph" else None
   - `fixed`: vector if chunk_strategy == "fixed" else None
3. Upload to Qdrant collection in batches

**Qdrant Collection Schema:**
```python
Collection Name: "sports_science_papers"

Vectors Config:
- semantic: VectorParams(size=384, distance=Distance.COSINE)
- paragraph: VectorParams(size=384, distance=Distance.COSINE)
- fixed: VectorParams(size=384, distance=Distance.COSINE)

Payload Indexes:
- section: KEYWORD (for filtering by paper section)
- paper_filename: KEYWORD (for filtering by paper)
- year: INTEGER (for temporal filtering)
```

**Error Handling (Fail-Fast):**
- If Qdrant upload fails for any chunk → stop pipeline
- Log: which paper/chunk failed, error message, payload
- Exit with non-zero code

**Output:** Populated Qdrant collection

---

## 5. Search & Analysis

### Search Interface

**Method Signature:**
```python
def search_and_compare(
    query: str,
    limit: int = 3,
    section_filter: Optional[str] = None,
    year_filter: Optional[int] = None,
    paper_filter: Optional[str] = None
) -> Dict[str, Any]
```

**Process:**
1. Encode query with same model (`all-MiniLM-L6-v2`)
2. Query all three strategies in parallel using Qdrant's named vectors
3. Apply filters if specified (section, year, paper filename)
4. Collect results and compute analysis metrics

**Output Format:**
```python
{
    "query": "GPS tracking accuracy in soccer",
    "results_by_strategy": {
        "semantic": [SearchResult, ...],
        "paragraph": [SearchResult, ...],
        "fixed": [SearchResult, ...]
    },
    "analysis": {
        "chunk_count": {"semantic": 142, "paragraph": 198, "fixed": 215},
        "avg_chunk_size": {"semantic": 247, "paragraph": 183, "fixed": 200},
        "section_distribution": {"Methods": 0.45, "Results": 0.30, "Discussion": 0.25},
        "top_strategy": "semantic",
        "avg_scores": {"semantic": 0.821, "paragraph": 0.789, "fixed": 0.756}
    }
}
```

### Analysis Features

**1. Relevance Score Statistics**
- Min, max, average score per strategy
- Score distribution (histogram-ready data)

**2. Section Distribution**
- Which sections appear most in top results
- Breakdown per strategy

**3. Chunk Size Analysis**
- Average chunk size (characters) per strategy
- Min/max chunk sizes
- Total chunk count per strategy

**4. Strategy Comparison Matrix**
- For each query, which strategy had highest average score
- Rank strategies by performance

**5. Export Capability**
- Save search results to JSON for offline analysis
- Log queries and results for test case building

---

## 6. Class Design

### PDFExtractor

```python
class PDFExtractor:
    """Extracts text and metadata from academic PDFs using PyMuPDF."""
    
    def __init__(self, manual_mappings_path: Optional[Path] = None):
        """
        Args:
            manual_mappings_path: Path to section_mappings.json
        """
        self.manual_mappings = self._load_manual_mappings(manual_mappings_path)
    
    def extract(self, pdf_path: Path) -> Paper:
        """
        Extract text, detect sections, extract metadata.
        
        Raises:
            SectionDetectionError: If sections cannot be detected and no manual mapping exists
            PDFExtractionError: If PDF cannot be read or parsed
        """
    
    def _detect_sections(self, text: str, filename: str) -> Dict[str, str]:
        """
        Auto-detect paper sections using regex patterns.
        Falls back to manual mapping from section_mappings.json.
        
        Returns:
            Dict mapping section names to section text
            
        Raises:
            SectionDetectionError: If detection fails and no manual mapping available
        """
    
    def _load_manual_mappings(self, path: Optional[Path]) -> Dict:
        """Load manual section mappings from JSON file."""
    
    def _extract_metadata(self, pdf_path: Path, text: str) -> Dict[str, Any]:
        """
        Extract title, authors, year, DOI, journal from PDF metadata and text.
        
        Returns:
            Dict with keys: authors, year, DOI, journal, page_count
        """
```

### TextChunker

```python
class TextChunker:
    """Implements multiple chunking strategies for academic papers."""
    
    def __init__(self, embed_model: HuggingFaceEmbedding):
        """
        Args:
            embed_model: Used for semantic chunking
        """
        self.embed_model = embed_model
        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model
        )
    
    def chunk_semantic(self, paper: Paper) -> List[PaperChunk]:
        """
        Semantic chunking using SemanticSplitterNodeParser.
        Respects semantic boundaries, optimal for academic content.
        """
    
    def chunk_paragraph(self, paper: Paper) -> List[PaperChunk]:
        """
        Paragraph-based chunking on double newlines.
        Preserves natural paragraph structure.
        """
    
    def chunk_fixed_size(
        self, 
        paper: Paper, 
        chunk_size: int = 200, 
        overlap: int = 50
    ) -> List[PaperChunk]:
        """
        Fixed-size chunking with word overlap.
        Baseline strategy for comparison.
        """
    
    def chunk_all_strategies(self, paper: Paper) -> List[PaperChunk]:
        """
        Returns chunks from all three strategies.
        Combines results for parallel vectorization.
        """
    
    def _create_chunk(
        self,
        paper: Paper,
        chunk_text: str,
        chunk_index: int,
        section: str,
        page_number: int,
        strategy: str
    ) -> PaperChunk:
        """Helper to create PaperChunk with inherited metadata."""
```

### VectorStore

```python
class VectorStore:
    """Manages Qdrant vector database operations."""
    
    def __init__(self, client: QdrantClient, collection_name: str):
        """
        Args:
            client: Initialized QdrantClient
            collection_name: Name of collection to create/use
        """
        self.client = client
        self.collection_name = collection_name
    
    def create_collection(self, dimension: int = 384, recreate: bool = False):
        """
        Create Qdrant collection with three named vectors and payload indexes.
        
        Args:
            dimension: Vector dimension (384 for all-MiniLM-L6-v2)
            recreate: If True, delete existing collection first
        """
    
    def upload_chunks(
        self, 
        chunks: List[PaperChunk], 
        encoder: SentenceTransformer,
        batch_size: int = 100
    ):
        """
        Vectorize and upload chunks to Qdrant.
        
        Args:
            chunks: List of PaperChunk objects
            encoder: SentenceTransformer for vectorization
            batch_size: Number of points to upload per batch
            
        Raises:
            VectorUploadError: If upload fails
        """
    
    def search(
        self,
        query: str,
        encoder: SentenceTransformer,
        strategy: str,
        limit: int = 3,
        section_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        paper_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search using specific chunking strategy with optional filters.
        
        Args:
            query: Search query text
            encoder: SentenceTransformer to encode query
            strategy: "semantic", "paragraph", or "fixed"
            limit: Number of results to return
            section_filter: Filter by paper section
            year_filter: Filter by publication year
            paper_filter: Filter by paper filename
            
        Returns:
            List of SearchResult objects sorted by score
        """
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return collection statistics.
        
        Returns:
            Dict with keys:
            - total_points: Total number of chunks
            - chunks_by_strategy: Count per strategy
            - chunks_by_section: Count per section
            - unique_papers: Number of papers indexed
        """
```

### SemanticSearchEngine

```python
class SemanticSearchEngine:
    """Orchestrates the complete semantic search pipeline."""
    
    def __init__(
        self,
        pdf_dir: Path,
        qdrant_client: QdrantClient,
        collection_name: str = "sports_science_papers",
        manual_mappings_path: Optional[Path] = None
    ):
        """
        Initialize semantic search engine.
        
        Args:
            pdf_dir: Directory containing PDF papers
            qdrant_client: Initialized Qdrant client
            collection_name: Name for Qdrant collection
            manual_mappings_path: Path to section_mappings.json
        """
        self.pdf_dir = pdf_dir
        self.extractor = PDFExtractor(manual_mappings_path)
        self.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
        self.chunker = TextChunker(self.embed_model)
        self.vector_store = VectorStore(qdrant_client, collection_name)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    def process_papers(self, recreate_collection: bool = False):
        """
        Process all PDFs: extract → chunk → vectorize → upload.
        
        Implements FAIL-FAST error handling:
        - Stops on first error
        - Logs which paper failed and why
        - No partial uploads (rollback not implemented)
        
        Args:
            recreate_collection: If True, delete and recreate Qdrant collection
            
        Raises:
            PDFProcessingError: If any paper fails processing
        """
    
    def search_and_compare(
        self,
        query: str,
        limit: int = 3,
        section_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        paper_filter: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Search all strategies, compare results, return analysis.
        
        Args:
            query: Search query text
            limit: Number of results per strategy
            section_filter: Filter by section name
            year_filter: Filter by publication year
            paper_filter: Filter by paper filename
            verbose: If True, print formatted results
            
        Returns:
            Dict with keys:
            - query: Original query
            - results_by_strategy: Dict[strategy_name, List[SearchResult]]
            - analysis: Dict with statistics and comparisons
        """
    
    def analyze_chunking_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze chunk statistics per strategy from Qdrant collection.
        
        Returns:
            Dict with keys:
            - chunk_count: Chunks per strategy
            - avg_chunk_size: Average characters per strategy
            - section_distribution: Section breakdown per strategy
            - papers_indexed: Total unique papers
        """
    
    def export_results(
        self, 
        results: Dict[str, Any], 
        output_path: Path
    ):
        """Export search results and analysis to JSON file."""
```

---

## 7. Error Handling

### Fail-Fast Strategy

The system implements strict fail-fast error handling:

**Principles:**
- Stop processing on first error
- Provide detailed error messages with context
- No silent failures or partial state
- Exit with non-zero code

**Error Types:**

```python
class PDFExtractionError(Exception):
    """Raised when PDF cannot be read or parsed."""
    
class SectionDetectionError(Exception):
    """Raised when sections cannot be detected and no manual mapping exists."""
    
class VectorUploadError(Exception):
    """Raised when Qdrant upload fails."""
    
class PDFProcessingError(Exception):
    """General processing error with context."""
```

**Error Messages Include:**
- Paper filename that failed
- Processing stage (extraction, chunking, upload)
- Specific error details
- Suggested resolution (e.g., "Add manual mapping to section_mappings.json")

---

## 8. Configuration

### Environment Variables

Required in `.env` file:
```bash
QDRANT_URL="https://xxx.aws.cloud.qdrant.io"
QDRANT_API_KEY="eyJhbGci..."
```

### Manual Section Mappings

File: `section_mappings.json` (created as needed)

Format:
```json
{
  "paper_filename.pdf": {
    "Abstract": {"start_page": 1, "end_page": 1},
    "Introduction": {"start_page": 1, "end_page": 2},
    "Methods": {"start_page": 2, "end_page": 4},
    "Results": {"start_page": 4, "end_page": 6},
    "Discussion": {"start_page": 6, "end_page": 7},
    "References": {"start_page": 7, "end_page": 8}
  }
}
```

### Chunking Parameters

Configurable via class initialization:
```python
# Semantic chunking
buffer_size = 1
breakpoint_percentile_threshold = 95

# Fixed-size chunking
chunk_size = 200  # words
overlap = 50      # words
```

---

## 9. Dependencies

### Required Packages

```txt
qdrant-client>=1.7.0
sentence-transformers>=2.2.0
llama-index-core>=0.10.0
llama-index-embeddings-huggingface>=0.1.0
PyMuPDF>=1.23.0
python-dotenv>=1.0.0
transformers>=4.35.0
```

### Installation

```bash
source .venv/bin/activate
pip install qdrant-client sentence-transformers llama-index-core llama-index-embeddings-huggingface PyMuPDF python-dotenv transformers
```

---

## 10. Testing Strategy

### Unit Tests

- `test_pdf_extractor.py` - Test extraction, section detection, metadata parsing
- `test_text_chunker.py` - Test all three chunking strategies
- `test_vector_store.py` - Test Qdrant operations (mock client)
- `test_search_engine.py` - Integration tests with sample PDFs

### Test Coverage Target

80% minimum following project standards.

### Test Data

- 2-3 sample sports science PDFs in `test_assets/`
- Known section boundaries for validation
- Pre-computed expected chunk counts

---

## 11. Usage Example

```python
# Initialize
from pathlib import Path
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

engine = SemanticSearchEngine(
    pdf_dir=Path("assets"),
    qdrant_client=client,
    manual_mappings_path=Path("section_mappings.json")
)

# Process all papers
engine.process_papers(recreate_collection=True)

# Search and compare
results = engine.search_and_compare(
    query="GPS tracking accuracy in soccer",
    limit=3,
    section_filter="Methods",
    verbose=True
)

# Analyze effectiveness
analysis = engine.analyze_chunking_effectiveness()
print(analysis)

# Export results
engine.export_results(results, Path("search_results.json"))
```

---

## 12. Success Criteria

**Functional:**
- ✓ All 27 PDFs successfully processed
- ✓ Section detection working or manual mappings provided
- ✓ Three chunking strategies implemented and compared
- ✓ Search returns relevant results with metadata
- ✓ Analysis shows clear differences between strategies

**Non-Functional:**
- ✓ Code follows OOP principles (clear class boundaries, single responsibility)
- ✓ DRY principle applied (no code duplication)
- ✓ Type annotations on all functions
- ✓ Immutable data structures (frozen dataclasses)
- ✓ 80%+ test coverage
- ✓ Fail-fast error handling with descriptive messages

**Deliverables:**
- ✓ Working semantic search engine in `12_handson_sportscience_semantic_engine.py`
- ✓ Populated Qdrant collection with 27 papers
- ✓ Search comparison demonstrating strategy differences
- ✓ Analysis report showing which strategy works best for sports science papers
- ✓ Discord post with findings (per Day 1 project requirements)

---

## 13. Future Enhancements (Out of Scope)

These are explicitly **not** part of this implementation:

- Interactive web UI or CLI tool
- Real-time PDF uploads
- Multi-language support
- OCR for scanned PDFs
- Image/figure extraction and analysis
- Citation graph analysis
- Collaborative filtering or recommendation system
- API endpoint deployment
- Authentication/authorization
- Batch processing with parallelization
- Incremental updates (currently full reindex)

---

## 14. References

- Qdrant Day 1 Project Tutorial: https://qdrant.tech/documentation/qdrant-essentials/day-1/project/
- PyMuPDF Documentation: https://pymupdf.readthedocs.io/
- LlamaIndex Semantic Splitter: https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/#semanticsplitternodeparser
- Sentence Transformers: https://www.sbert.net/

---

**Design Status:** ✓ Approved by User  
**Next Step:** Invoke `writing-plans` skill to create implementation plan
