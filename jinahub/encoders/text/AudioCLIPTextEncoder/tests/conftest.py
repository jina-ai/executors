__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import shutil
import subprocess
from pathlib import Path

import pytest
from jina import Document, DocumentArray


@pytest.fixture(scope="session", autouse=True)
def download_cache():
    subprocess.run(
        'scripts/download_full.sh', cwd=Path(__file__).parents[1], check=True
    )
    yield
    shutil.rmtree('.cache')


@pytest.fixture()
def data_generator():
    def _generator():
        data_file_path = Path(__file__).parent / 'texts' / 'test_data.txt'
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
