__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pytest
from jina import Document, DocumentArray


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def data_generator(test_dir: str):
    def _generator():
        data_file_path = os.path.join(test_dir, 'data', 'test_data.txt')
        with open(data_file_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            yield Document(text=line.strip())
    return _generator


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
