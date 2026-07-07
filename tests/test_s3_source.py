from unittest.mock import MagicMock

from sportsscience_rag.s3_source import S3Source, parse_s3_url


def test_parse_s3_url():
    assert parse_s3_url("s3://bkt/a/b.pdf") == ("bkt", "a/b.pdf")
    assert parse_s3_url("s3://bkt") == ("bkt", "")


def _paginator_with(keys):
    paginator = MagicMock()
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    return paginator


def test_list_pdfs_filters_and_builds_urls():
    client = MagicMock()
    client.get_paginator.return_value = _paginator_with(
        ["papers/a.pdf", "papers/b.PDF", "papers/notes.txt", "papers/"]
    )
    src = S3Source(client, "bkt")
    result = src.list_pdfs(["papers/"])
    assert result == [
        ("papers/a.pdf", "s3://bkt/papers/a.pdf"),
        ("papers/b.PDF", "s3://bkt/papers/b.PDF"),
    ]


def test_list_pdfs_respects_limit():
    client = MagicMock()
    client.get_paginator.return_value = _paginator_with(["a.pdf", "b.pdf", "c.pdf"])
    src = S3Source(client, "bkt")
    assert len(src.list_pdfs([""], limit=2)) == 2


def test_list_pdfs_limit_caps_across_prefixes():
    client = MagicMock()
    paginator = MagicMock()

    def _paginate(Bucket, Prefix):
        pages = {
            "a/": [{"Contents": [{"Key": "a/1.pdf"}, {"Key": "a/2.pdf"}]}],
            "b/": [{"Contents": [{"Key": "b/1.pdf"}, {"Key": "b/2.pdf"}]}],
        }
        return iter(pages[Prefix])

    paginator.paginate.side_effect = _paginate
    client.get_paginator.return_value = paginator

    src = S3Source(client, "bkt")
    result = src.list_pdfs(["a/", "b/"], limit=3)
    assert result == [
        ("a/1.pdf", "s3://bkt/a/1.pdf"),
        ("a/2.pdf", "s3://bkt/a/2.pdf"),
        ("b/1.pdf", "s3://bkt/b/1.pdf"),
    ]  # capped at 3 total, spanning both prefixes


def test_fetch_reads_body():
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = b"PDFBYTES"
    client.get_object.return_value = {"Body": body}
    src = S3Source(client, "bkt")
    assert src.fetch("a.pdf") == b"PDFBYTES"
    client.get_object.assert_called_once_with(Bucket="bkt", Key="a.pdf")
