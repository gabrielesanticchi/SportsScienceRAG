import pytest

from sportsscience_rag.config import IngestionConfig


ENV = {
    "AWS_ACCESS_KEY_ID": "ak",
    "AWS_SECRET_ACCESS_KEY": "sk",
    "AWS_REGION": "eu-west-1",
    "S3_BUCKET": "bkt",
    "QDRANT_URL": "https://q",
    "QDRANT_API_KEY": "qk",
    "QDRANT_COLLECTION": "coll",
}


def test_from_env_reads_values(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = IngestionConfig.from_env(env_path=None)
    assert cfg.s3_bucket == "bkt"
    assert cfg.qdrant_collection == "coll"
    assert cfg.image_dpi == 150
    assert cfg.derived_prefix == "derived/"


def test_from_env_raises_on_missing(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="Missing required"):
        IngestionConfig.from_env(env_path=None)


def test_chunk_config_hash_is_property(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    cfg = IngestionConfig.from_env(env_path=None)
    assert len(cfg.chunk_config_hash) == 16
