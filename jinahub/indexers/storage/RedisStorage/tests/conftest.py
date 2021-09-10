import os
import time

import pytest
from jina import Document, DocumentArray

from ..redis_storage import RedisStorage


@pytest.fixture(scope='function')
def indexer():
    return RedisStorage()


@pytest.fixture()
def docker_compose(request):
    os.system(
        f'docker-compose -f {request.param} --project-directory . up  --build -d --remove-orphans'
    )
    time.sleep(5)
    yield
    os.system(
        f'docker-compose -f {request.param} --project-directory . down --remove-orphans'
    )


@pytest.fixture(scope='function')
def docs():
    return DocumentArray(
        [
            Document(content=value)
            for value in ['cat', 'dog', 'crow', 'pikachu', 'magikarp']
        ]
    )
