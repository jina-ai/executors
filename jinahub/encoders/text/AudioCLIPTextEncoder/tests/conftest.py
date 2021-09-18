__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def download_cache():
    subprocess.run(
        'scripts/download_full.sh',
        cwd=Path(__file__).parents[1],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    yield
    shutil.rmtree(Path(__file__).parents[1] / '.cache')


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
