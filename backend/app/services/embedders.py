"""Dense, sparse, and ColBERT multivector embedders."""

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import httpx
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import models
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings


class EmbedderRateLimitError(Exception):
    """Raised when an upstream embedder returns HTTP 429."""


@dataclass(frozen=True)
class HybridEmbeddings:
    dense: list[float]
    sparse: models.SparseVector
    multi: list[list[float]]


class HybridEmbedder:
    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self._dense_model = SentenceTransformer(cfg.dense_model)
        self._sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self._colbert_model = LateInteractionTextEmbedding(model_name=cfg.colbert_model)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _legacy_sparse(self, text: str) -> models.SparseVector:
        counts = Counter(self._tokenize(text))
        indices = list(counts.keys())
        values = [float(counts[idx]) for idx in indices]
        return models.SparseVector(indices=[hash(i) % 2_000_000 for i in indices], values=values)

    def embed_text(self, text: str) -> HybridEmbeddings:
        dense = self._dense_model.encode(text, normalize_embeddings=True).tolist()

        try:
            sparse_result = next(iter(self._sparse_model.embed([text])))
            sparse = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            )
        except Exception:
            sparse = self._legacy_sparse(text)

        try:
            colbert_tokens = list(self._colbert_model.embed([text]))
            multi = [token.tolist() for token in colbert_tokens[0]]
        except Exception as exc:
            if "429" in str(exc):
                raise EmbedderRateLimitError(str(exc)) from exc
            dense_tokens = self._dense_model.encode(
                self._tokenize(text) or [text],
                normalize_embeddings=True,
            )
            multi = [vector.tolist() for vector in dense_tokens]

        return HybridEmbeddings(dense=dense, sparse=sparse, multi=multi)

    def embed_query(self, query: str) -> HybridEmbeddings:
        return self.embed_text(query)


def check_embedder_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, EmbedderRateLimitError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    return "429" in str(exc)


@lru_cache
def get_hybrid_embedder() -> HybridEmbedder:
    return HybridEmbedder()
