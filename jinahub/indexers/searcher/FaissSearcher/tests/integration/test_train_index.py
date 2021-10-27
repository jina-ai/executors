__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow
from jina.executors.metas import get_default_metas

from ...faiss_searcher import FaissSearcher


def _get_docs_from_vecs(queries):
    docs = DocumentArray()
    for q in queries:
        doc = Document(embedding=q)
        docs.append(doc)
    return docs


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    metas['name'] = 'faiss_idx'
    yield metas
    del os.environ['TEST_WORKSPACE']


def test_train_and_index(metas):
    query = np.array(np.random.random([10, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(query)

    train_data = np.array(np.random.random([512, 10]), dtype=np.float32)
    train_docs = _get_docs_from_vecs(train_data)

    f = Flow().add(
        uses=FaissSearcher,
        timeout_ready=-1,
        uses_with={
            'index_key': 'IVF10_HNSW32,PQ2',
        },
    )
    with f:
        # train and index docs first
        f.post(on='/train', data=train_docs, request_size=0)

        result = f.post(
            on='/search', data=query_docs, return_results=True, parameters={'top_k': 4}
        )[0].docs
        assert len(result[0].matches) == 4
        for d in result:
            assert (
                d.matches[0].scores['cosine'].value
                <= d.matches[1].scores['cosine'].value
            )


@pytest.mark.gpu
@pytest.mark.docker
def test_docker_runtime_gpu(build_docker_image_gpu: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'executor',
                f'--uses=docker://{build_docker_image_gpu}',
                '--gpus',
                'all',
            ],
            timeout=30,
            check=True,
        )
