from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from PIL import Image

from sportsscience_rag.models import PageRender
from sportsscience_rag.persistence import RenderStore, TextStore


def _renders(n):
    return [PageRender(page_number=i + 1, image=Image.new("RGB", (2, 2))) for i in range(n)]


def _not_found():
    return ClientError({"Error": {"Code": "404"}}, "HeadObject")


def test_render_key_format():
    store = RenderStore(MagicMock(), "bkt", "derived/")
    assert store.render_key("abc", 3) == "derived/abc/screenshots/page-3.png"


def test_upload_puts_missing_objects():
    client = MagicMock()
    client.head_object.side_effect = _not_found()
    store = RenderStore(client, "bkt", "derived/")
    keys = store.upload("abc", _renders(2))
    assert keys == ["derived/abc/screenshots/page-1.png", "derived/abc/screenshots/page-2.png"]
    assert client.put_object.call_count == 2


def test_upload_skips_existing_objects():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 10}  # exists
    store = RenderStore(client, "bkt", "derived/")
    store.upload("abc", _renders(2))
    client.put_object.assert_not_called()


def test_upload_reraises_non_404_errors():
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    store = RenderStore(client, "bkt", "derived/")
    with pytest.raises(ClientError):
        store.upload("abc", _renders(1))
    client.put_object.assert_not_called()


def test_upload_mixed_existing_and_missing():
    client = MagicMock()
    # page 1 exists, page 2 missing (404)
    client.head_object.side_effect = [{"ContentLength": 10}, _not_found()]
    store = RenderStore(client, "bkt", "derived/")
    keys = store.upload("abc", _renders(2))
    assert keys == [
        "derived/abc/screenshots/page-1.png",
        "derived/abc/screenshots/page-2.png",
    ]
    assert client.put_object.call_count == 1  # only the missing page uploaded


# --- TextStore -------------------------------------------------------------


_PAGE_TEXTS = ((1, "first page text"), (2, "second page text"))


def test_text_key_formats():
    store = TextStore(MagicMock(), "bkt", "derived/")
    assert store.markdown_key("abc") == "derived/abc/text/document.md"
    assert store.page_text_key("abc", 2) == "derived/abc/text/page-2.txt"


def test_text_upload_puts_markdown_and_pages_when_missing():
    client = MagicMock()
    client.head_object.side_effect = _not_found()
    store = TextStore(client, "bkt", "derived/")
    keys = store.upload("abc", "# Title\n\nbody", _PAGE_TEXTS)
    assert keys == [
        "derived/abc/text/document.md",
        "derived/abc/text/page-1.txt",
        "derived/abc/text/page-2.txt",
    ]
    assert client.put_object.call_count == 3
    # markdown body is UTF-8 encoded with a markdown content type
    md_call = client.put_object.call_args_list[0].kwargs
    assert md_call["Key"] == "derived/abc/text/document.md"
    assert md_call["Body"] == b"# Title\n\nbody"
    assert md_call["ContentType"].startswith("text/markdown")


def test_text_upload_skips_existing_objects():
    client = MagicMock()
    client.head_object.return_value = {"ContentLength": 10}  # everything exists
    store = TextStore(client, "bkt", "derived/")
    store.upload("abc", "# Title", _PAGE_TEXTS)
    client.put_object.assert_not_called()


def test_text_upload_reraises_non_404_errors():
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    store = TextStore(client, "bkt", "derived/")
    with pytest.raises(ClientError):
        store.upload("abc", "# Title", _PAGE_TEXTS)
    client.put_object.assert_not_called()


def test_text_upload_handles_no_page_texts():
    client = MagicMock()
    client.head_object.side_effect = _not_found()
    store = TextStore(client, "bkt", "derived/")
    keys = store.upload("abc", "# Title", ())
    assert keys == ["derived/abc/text/document.md"]
    assert client.put_object.call_count == 1
