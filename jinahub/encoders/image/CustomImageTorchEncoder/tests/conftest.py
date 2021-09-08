__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess
from pathlib import Path

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session', autouse=True)
def create_model_weights():
    path_to_model = os.path.join(TEST_DIR, 'model', 'model_state_dict.pth')
    if not os.path.isfile(path_to_model):
        os.system(f'python {os.path.join(TEST_DIR, "model", "external_model.py")}')

    yield

    if os.path.exists(path_to_model):
        os.remove(path_to_model)


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
