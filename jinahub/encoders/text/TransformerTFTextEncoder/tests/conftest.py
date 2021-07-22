__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pytest
from jina import Document, DocumentArray


@pytest.fixture()
def docs_with_text() -> DocumentArray:
    return DocumentArray([
        Document(text='hello world') for _ in range(10)
    ])


@pytest.fixture()
def docs_with_chunk_text() -> DocumentArray:
    return DocumentArray([
        Document(
            chunks=[Document(text='hello world') for _ in range(10)]
        )
    ])


@pytest.fixture()
def docs_with_chunk_chunk_text() -> DocumentArray:
    return DocumentArray([
        Document(
            chunks=[Document(
                chunks=[Document(text='hello world') for _ in range(10)])])
    ])