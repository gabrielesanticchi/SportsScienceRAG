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
