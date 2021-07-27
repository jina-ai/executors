__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import shutil

import pytest
from jina import Document, DocumentArray


@pytest.fixture(scope="session", autouse=True)
def download_cache():
    os.system('scripts/download_full.sh')
    yield
    # shutil.rmtree('.cache') 


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
    return DocumentArray([Document(text='hello world') for _ in range(10)])


@pytest.fixture()
def docs_with_chunk_text() -> DocumentArray:
    chunks = [Document(text='hello world') for _ in range(10)]
    return DocumentArray([Document(chunks=chunks)])


@pytest.fixture()
def docs_with_chunk_chunk_text() -> DocumentArray:
    root = Document()
    chunks = [Document() for _ in range(10)]
    chunks_2 = [[Document(text='hello world') for _ in range(10)] for _ in range(10)]

    root.chunks.extend(chunks)
    for i, chunk in enumerate(chunks):
        chunk.chunks.extend(chunks_2[i])

    return DocumentArray([root])
