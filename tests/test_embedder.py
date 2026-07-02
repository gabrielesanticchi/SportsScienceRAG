import pytest

from sportsscience_rag.embedder import TextEmbedder


@pytest.mark.integration
def test_embed_returns_384d_vectors():
    emb = TextEmbedder()
    vectors = emb.embed(["hello world", "second document"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
    assert emb.dimension == 384
