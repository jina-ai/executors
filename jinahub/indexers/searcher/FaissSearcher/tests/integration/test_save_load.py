__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow
from jina.executors.metas import get_default_metas
from jina_commons.indexers.dump import export_dump_streaming

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


def test_save(metas, tmpdir):
    vec_idx = np.random.randint(0, high=512, size=[512]).astype(str)
    vec = np.array(np.random.random([512, 10]), dtype=np.float32)

    query = np.array(np.random.random([10, 10]), dtype=np.float32)
    query_docs = _get_docs_from_vecs(query)

    export_dump_streaming(
        os.path.join(tmpdir, 'dump'),
        1,
        len(vec_idx),
        zip(vec_idx, vec, [b'' for _ in range(len(vec))]),
    )
    dump_path = os.path.join(tmpdir, 'dump')

    f = Flow().add(
        uses=FaissSearcher,
        timeout_ready=-1,
        uses_with={
            'index_key': 'Flat',
            'dump_path': dump_path,
        },
        uses_meta=metas,
    )
    with f:
        f.post(on='/save')

    new_f = Flow().add(
        uses=FaissSearcher,
        timeout_ready=-1,
        uses_with={
            'index_key': 'Flat',
        },
        uses_meta=metas,
    )
    with new_f:
        result = new_f.post(
            on='/search', data=query_docs, return_results=True, parameters={'top_k': 4}
        )[0].docs
        assert len(result[0].matches) == 4
        for d in result:
            assert (
                d.matches[0].scores['cosine'].value
                <= d.matches[1].scores['cosine'].value
            )
