from unittest.mock import MagicMock

from sportsscience_rag.config import IngestionConfig
from sportsscience_rag.models import Chunk, ParsedDocument
from sportsscience_rag.pipeline import IngestionPipeline

CFG = IngestionConfig(
    aws_access_key_id="x", aws_secret_access_key="x", aws_region="x",
    s3_bucket="bkt", qdrant_url="x", qdrant_api_key="x",
)


def _pipeline(**overrides):
    defaults = dict(
        config=CFG,
        source=MagicMock(),
        parser=MagicMock(),
        chunker=MagicMock(),
        embedder=MagicMock(),
        render_store=MagicMock(),
        store=MagicMock(),
        logger=MagicMock(),
        parser_version="docling-test",
    )
    defaults.update(overrides)
    return IngestionPipeline(**defaults), defaults


def test_happy_path_upserts_and_reports_done():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = False
    d["parser"].parse.return_value = ParsedDocument("# H\n\ntext", 1, (), False, ())
    d["chunker"].chunk.return_value = [Chunk(0, "text", "H", ())]
    d["embedder"].embed.return_value = [[0.0] * 384]
    d["store"].upsert.return_value = 1

    result = p.run([""])
    assert result.outcomes[0].status == "done"
    assert result.outcomes[0].n_chunks == 1
    assert not result.quarantine
    d["store"].upsert.assert_called_once()


def test_skips_already_ingested():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = True

    result = p.run([""])
    assert result.outcomes[0].status == "skipped"
    d["parser"].parse.assert_not_called()


def test_empty_document_is_quarantined():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = False
    d["parser"].parse.return_value = ParsedDocument("", 1, (), True, ())

    result = p.run([""])
    assert result.outcomes[0].status == "quarantined"
    assert result.quarantine[0].stage == "parse"
    d["store"].upsert.assert_not_called()


def test_exception_quarantines_and_continues():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("a.pdf", "s3://bkt/a.pdf"), ("b.pdf", "s3://bkt/b.pdf")]
    d["source"].fetch.return_value = b"PDF"
    d["store"].already_ingested.return_value = False
    d["parser"].parse.side_effect = [RuntimeError("corrupt"), ParsedDocument("# H\n\nt", 1, (), False, ())]
    d["chunker"].chunk.return_value = [Chunk(0, "t", "H", ())]
    d["embedder"].embed.return_value = [[0.0] * 384]
    d["store"].upsert.return_value = 1

    result = p.run([""])
    statuses = {o.source: o.status for o in result.outcomes}
    assert statuses["s3://bkt/a.pdf"] == "quarantined"
    assert statuses["s3://bkt/b.pdf"] == "done"
    assert result.quarantine[0].error_class == "RuntimeError"


def test_dry_run_writes_nothing():
    p, d = _pipeline()
    d["source"].list_pdfs.return_value = [("k.pdf", "s3://bkt/k.pdf")]
    result = p.run([""], dry_run=True)
    assert result.outcomes[0].status == "skipped"
    d["source"].fetch.assert_not_called()
    d["store"].upsert.assert_not_called()
