from sportsscience_rag.hashing import content_hash, chunk_config_hash


def test_content_hash_is_deterministic_and_hex():
    h1 = content_hash(b"hello world")
    h2 = content_hash(b"hello world")
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_content_hash_differs_on_different_bytes():
    assert content_hash(b"a") != content_hash(b"b")


def test_chunk_config_hash_stable_and_sensitive():
    headers = (("#", "h1"), ("##", "h2"))
    base = chunk_config_hash(512, 64, headers)
    assert base == chunk_config_hash(512, 64, headers)
    assert base != chunk_config_hash(256, 64, headers)
    assert base != chunk_config_hash(512, 64, (("#", "h1"),))
    assert len(base) == 16
