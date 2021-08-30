import os
import time

import pytest
import numpy as np
from jina import Document, DocumentArray

from .. import MongoDBStorage

NUM_DOCS = 10


@pytest.fixture
def storage():
    return MongoDBStorage()


@pytest.fixture
def docs_to_index():
    docu_array = DocumentArray()
    for idx in range(0, NUM_DOCS):
        d = Document(text=f'hello {idx}')
        d.embedding = np.random.random(20)
        docu_array.append(d)
    return docu_array


@pytest.fixture
def docs_to_index_no_embedding():
    docu_array = DocumentArray()
    for idx in range(0, NUM_DOCS):
        d = Document(text=f'hello {idx}')
        docu_array.append(d)
    return docu_array


@pytest.fixture
def docker_compose(request):
    os.system(
        f"docker-compose -f {request.param} --project-directory . up  --build -d --remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {request.param} --project-directory . down --remove-orphans"
    )
