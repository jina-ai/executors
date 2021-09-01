__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict

import pytest
import numpy as np
from PIL import Image

from jina import DocumentArray, Document


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def test_images(test_dir: str) -> Dict[str, np.ndarray]:
    def get_path(file_name_no_suffix: str) -> str:
        return os.path.join(test_dir, "test_data", file_name_no_suffix + ".png")

    image_dict = {
        file_name: np.array(Image.open(get_path(file_name)))[:, :, 0:3]
        for file_name in ["airplane", "banana1", "banana2", "satellite", "studio"]
    }
    return image_dict


@pytest.fixture()
def docs_with_blobs() -> DocumentArray:
    return DocumentArray(
        [Document(blob=np.ones((10, 10, 3), dtype=np.uint8)) for _ in range(11)]
    )


@pytest.fixture()
def docs_with_chunk_blobs() -> DocumentArray:
    return DocumentArray(
        [
            Document(chunks=[Document(blob=np.ones((10, 10, 3), dtype=np.uint8))])
            for _ in range(11)
        ]
    )


@pytest.fixture()
def docs_with_chunk_chunk_blobs() -> DocumentArray:
    return DocumentArray(
        [
            Document(
                chunks=[
                    Document(
                        chunks=[
                            Document(blob=np.ones((10, 10, 3), dtype=np.uint8))
                            for _ in range(11)
                        ]
                    )
                ]
            )
        ]
    )
