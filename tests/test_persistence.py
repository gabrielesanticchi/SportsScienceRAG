from unittest.mock import MagicMock

from botocore.exceptions import ClientError
from PIL import Image

from sportsscience_rag.models import PageRender
from sportsscience_rag.persistence import RenderStore


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
