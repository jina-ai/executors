import subprocess
from pathlib import Path

import pytest
from executor import VideoLoader


@pytest.fixture()
def encoder(tmp_path) -> VideoLoader:
    workspace = str(tmp_path / 'workspace')
    encoder = VideoLoader(metas={'workspace': workspace})
    return encoder


@pytest.fixture(scope='session')
def docker_image_name() -> str:
    return Path(__file__).parents[1].stem.lower()


@pytest.fixture(scope='session')
def build_docker_image(docker_image_name: str) -> str:
    subprocess.run(['docker', 'build', '-t', docker_image_name, '.'], check=True)
    return docker_image_name
