__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict

import numpy as np
import pytest
from PIL import Image
from jina import DocumentArray, Document


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def docs_with_blobs() -> DocumentArray:
    return DocumentArray([
        Document(blob=np.ones((3, 10, 10), dtype=np.float32)) for _ in range(10)
    ])


@pytest.fixture()
def docs_with_chunk_blobs() -> DocumentArray:
    return DocumentArray([
        Document(
            chunks=[Document(blob=np.ones((3, 10, 10), dtype=np.float32))]) for _ in range(10)
    ])


@pytest.fixture()
def docs_with_chunk_chunk_blobs() -> DocumentArray:
    return DocumentArray([
        Document(
            chunks=[Document(
                chunks=[Document(blob=np.ones((3, 10, 10), dtype=np.float32)) for _ in range(10)])])
    ])


@pytest.fixture()
def test_images(test_dir: str) -> Dict[str, np.ndarray]:

    def get_path(file_name_no_suffix: str) -> str:
        return os.path.join(test_dir, 'data', file_name_no_suffix + '.png')

    return {
        file_name: np.array(Image.open(get_path(file_name)), dtype=np.float32)[:, :, 0:3] / 255 for file_name in [
            'airplane', 'banana1', 'banana2', 'satellite', 'studio'
        ]
    }