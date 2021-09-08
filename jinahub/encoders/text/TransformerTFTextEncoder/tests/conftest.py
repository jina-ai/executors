__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from pathlib import Path

import pytest
from jina import Document, DocumentArray


@pytest.fixture()
def docs_with_text() -> DocumentArray:
    return DocumentArray([Document(text='hello world') for _ in range(10)])


@pytest.fixture()
def docs_with_chunk_text() -> DocumentArray:
    return DocumentArray(
        [Document(chunks=[Document(text='hello world') for _ in range(10)])]
    )


@pytest.fixture()
def docs_with_chunk_chunk_text() -> DocumentArray:
    return DocumentArray(
        [
            Document(
                chunks=[
                    Document(chunks=[Document(text='hello world') for _ in range(10)])
                ]
            )
        ]
    )


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
