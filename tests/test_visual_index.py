import pytest

from sportsscience_rag.visual_index import VisualIndexPlaceholder


def test_visual_index_is_not_implemented():
    with pytest.raises(NotImplementedError):
        VisualIndexPlaceholder().index_from_renders("derived/abc/screenshots/")
