__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def download_cache():
    subprocess.run(
        'scripts/download_full.sh', cwd=Path(__file__).parents[1], check=True
    )
    yield
    shutil.rmtree('.cache')


@pytest.fixture(scope='session')
def build_docker_image() -> str:
    img_name = Path(__file__).parents[1].stem.lower()
    subprocess.run(['docker', 'build', '-t', img_name, '.'], check=True)

    return img_name
