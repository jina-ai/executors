import glob
import os
import subprocess
from pathlib import Path

import cv2
import pytest
from jina import Document, DocumentArray

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='package')
def build_da():
    def _build_da():
        return DocumentArray(
            [
                Document(blob=cv2.imread(path), tags={'filename': path.split('/')[-1]})
                for path in glob.glob(os.path.join(cur_dir, 'data/img/*.jpg'))
            ]
        )

    return _build_da


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
