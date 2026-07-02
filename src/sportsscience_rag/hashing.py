"""Content and configuration hashing helpers."""

import hashlib


def content_hash(data: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def chunk_config_hash(
    chunk_size: int,
    chunk_overlap: int,
    headers: tuple[tuple[str, str], ...],
) -> str:
    """Return a short stable hash of the chunking configuration."""
    header_repr = "|".join(f"{marker}:{name}" for marker, name in headers)
    payload = f"{chunk_size}:{chunk_overlap}:{header_repr}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
