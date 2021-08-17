import pytest
import os

from jina import Document, DocumentArray


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))
