__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
from jina import Document, DocumentArray
from PIL import Image


@pytest.fixture()
def test_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def docs_with_blobs() -> DocumentArray:
    return DocumentArray(
        [Document(blob=np.ones((336, 336, 3), dtype=np.float32)) for _ in range(10)]
    )


@pytest.fixture()
def docs_with_chunk_blobs() -> DocumentArray:
    return DocumentArray(
        [
            Document(chunks=[Document(blob=np.ones((336, 336, 3), dtype=np.float32))])
            for _ in range(10)
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
                            Document(blob=np.ones((336, 336, 3), dtype=np.float32))
                            for _ in range(10)
                        ]
                    )
                ]
            )
        ]
    )


@pytest.fixture()
def test_images(test_dir: str) -> Dict[str, np.ndarray]:
    def get_path(file_name_no_suffix: str) -> str:
        return os.path.join(test_dir, 'test_data', file_name_no_suffix + '.png')

    return {
        file_name: np.array(Image.open(get_path(file_name)), dtype=np.float32)[
            :, :, 0:3
        ]
        / 255
        for file_name in ['airplane', 'banana1', 'banana2', 'satellite', 'studio']
    }


@pytest.fixture(scope='session')
def docker_image_name() -> str:
    return Path(__file__).parents[1].stem.lower()


@pytest.fixture(scope='session')
def build_docker_image(docker_image_name: str) -> str:
    subprocess.run(['docker', 'build', '-t', docker_image_name, '.'], check=True)
    return docker_image_name


@pytest.fixture(scope='session')
def build_docker_image_gpu(docker_image_name: str) -> str:
    image_name = f'{docker_image_name}:gpu'
    subprocess.run(
        ['docker', 'build', '-t', image_name, '-f', 'Dockerfile.gpu', '.'], check=True
    )
    return image_name
