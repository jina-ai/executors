import os
import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def download_cache():
    os.system('scripts/download_full.sh')
    yield
    shutil.rmtree('.cache', ignore_errors=True)
