import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def download_models():
    subprocess.run(
        ['python', '-m' 'laserembeddings', 'download-models'],
        cwd=Path(__file__).parents[1],
        check=True,
    )
    yield
