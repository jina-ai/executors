__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess
from pathlib import Path

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session', autouse=True)
def create_model_weights():
    path_to_embedding_array = os.path.join(TEST_DIR, 'unit', 'expected.npz')
    path_to_embedding_batch = os.path.join(TEST_DIR, 'unit', 'expected_batch.npz')

    if not os.path.isfile(path_to_embedding_array) or not os.path.isfile(
        path_to_embedding_batch
    ):
        os.system(
            f'python {os.path.join(TEST_DIR, "unit", "encoding_with_original_tfidf.py")}'
        )

    yield

    if os.path.isfile(path_to_embedding_array):
        os.remove(path_to_embedding_array)

    if os.path.isfile(path_to_embedding_batch):
        os.remove(path_to_embedding_batch)


@pytest.fixture(scope='session')
def docker_image_name() -> str:
    return Path(__file__).parents[1].stem.lower()


@pytest.fixture(scope='session')
def build_docker_image(docker_image_name: str) -> str:
    subprocess.run(['docker', 'build', '-t', docker_image_name, '.'], check=True)
    return docker_image_name
