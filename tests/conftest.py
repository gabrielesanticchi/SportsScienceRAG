"""Shared test fixtures."""

import io

import pytest
from PIL import Image


@pytest.fixture
def one_px_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()
