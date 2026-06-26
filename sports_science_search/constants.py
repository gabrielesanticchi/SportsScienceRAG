"""Constants for the sports science search engine."""

# Section detection patterns
SECTION_PATTERNS = {
    "Abstract": r"(?i)^abstract\s*$",
    "Introduction": r"(?i)^(introduction|1\.?\s*introduction)$",
    "Methods": r"(?i)^(methods?|methodology|materials?\s+and\s+methods?)$",
    "Results": r"(?i)^(results?|results?\s+and\s+discussion)$",
    "Discussion": r"(?i)^(discussion|conclusions?)$",
    "References": r"(?i)^(references?|bibliography|literature\s+cited)$",
}

# Chunking parameters
FIXED_CHUNK_SIZE = 200  # words
FIXED_CHUNK_OVERLAP = 50  # words
SEMANTIC_BUFFER_SIZE = 1
SEMANTIC_BREAKPOINT_THRESHOLD = 95
MIN_SECTIONS_FOR_AUTO_DETECTION = 3  # Minimum sections required for successful auto-detection

# Qdrant parameters
COLLECTION_NAME = "sports_science_papers"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
