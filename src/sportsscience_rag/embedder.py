"""Local dense text embedding via FastEmbed."""

from __future__ import annotations

from fastembed import TextEmbedding

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION = 384


class TextEmbedder:
    """Computes 384-dim MiniLM vectors client-side."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Loads the dense embedding model.

        Args:
            model_name: Name of the FastEmbed/sentence-transformers model to
                load. Defaults to ``sentence-transformers/all-MiniLM-L6-v2``,
                which produces 384-dimensional vectors.

        Returns:
            None.
        """
        self._model = TextEmbedding(model_name=model_name)
        self.dimension = DIMENSION

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Computes dense embedding vectors for a batch of texts.

        Args:
            texts: Texts to embed. An empty list short-circuits to an empty
                result without invoking the model.

        Returns:
            A list of embedding vectors (as plain ``float`` lists), one per
            input text, in the same order as ``texts``.
        """
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]
