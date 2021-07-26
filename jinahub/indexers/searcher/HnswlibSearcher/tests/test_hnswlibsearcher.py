__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow
from jina.executors.metas import get_default_metas
from jina_commons.indexers.dump import import_vectors

from .. import HnswlibSearcher

# fix the seed here
np.random.seed(500)

docs = DocumentArray([Document(embedding=np.random.random(10)) for i in range(10)])
search_doc = DocumentArray([Document(embedding=np.random.random(10))])
DUMP_PATH = 'tests/dump1'
TOP_K = 5


@pytest.fixture(scope='function', autouse=True)
def metas(tmpdir):
    os.environ['TEST_WORKSPACE'] = str(tmpdir)
    metas = get_default_metas()
    metas['workspace'] = os.environ['TEST_WORKSPACE']
    yield metas
    del os.environ['TEST_WORKSPACE']


@pytest.mark.parametrize(['metric', 'is_distance'],
                         [('l2', True), ('ip', True), ('cosine', True),
                          ('l2', False), ('ip', False), ('cosine', False)])
def test_metric(tmpdir, metric, is_distance):
    metas = {'workspace': str(tmpdir), 'name': 'searcher', 'pea_id': 0, 'replica_id': 0}

    indexer = HnswlibSearcher(dump_path=DUMP_PATH, default_top_k=TOP_K, metas=metas, metric=metric, is_distance=is_distance)
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    indexer.search(docs, {})

    assert len(docs[0].matches) == TOP_K
    for i in range(len(docs[0].matches) - 1):
        if not is_distance:
            assert docs[0].matches[i].scores[metric].value >= docs[0].matches[i + 1].scores[metric].value
        else:
            assert docs[0].matches[i].scores[metric].value <= docs[0].matches[i + 1].scores[metric].value


def test_query_vector(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'searcher', 'pea_id': 0, 'replica_id': 0}

    indexer = HnswlibSearcher(dump_path=DUMP_PATH, default_top_k=TOP_K, metas=metas)
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    indexer.search(docs, {})

    ids, vecs = import_vectors(DUMP_PATH, str(0))
    ids = np.array(list(ids))
    vecs = np.array(list(vecs))

    assert len(docs) == 1
    assert len(docs[0].matches) == TOP_K
    assert docs[0].matches[0].id in ids
    assert len(docs[0].matches[0].embedding) == 7
    assert docs[0].matches[0].embedding in vecs

    da = DocumentArray([Document(id=0), Document(id=1), Document(id=2)])
    indexer.fill_embedding(da)
    for i, doc in enumerate(da):
        assert list(doc.embedding)


def test_none_doc(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'searcher', 'pea_id': 0, 'replica_id': 0}
    indexer = HnswlibSearcher(dump_path=DUMP_PATH, default_top_k=TOP_K, metas=metas)
    indexer.search(None, {})

    indexer.fill_embedding(None)


def test_query_vector_empty(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'searcher', 'pea_id': 0, 'replica_id': 0}

    indexer = HnswlibSearcher(default_top_k=TOP_K, metas=metas)
    docs = DocumentArray([Document(embedding=np.random.random(7))])
    indexer.search(docs, {})
    assert len(docs[0].matches) == 0


def test_flow(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'searcher', 'pea_id': 0, 'replica_id': 0}

    flow = Flow().add(uses=HnswlibSearcher, override_with={'dump_path': DUMP_PATH, 'default_top_k': TOP_K},
                      override_metas=metas)
    with flow:
        resp = flow.post(
            on='/search',
            inputs=DocumentArray([Document(embedding=np.random.random(7))]),
            return_results=True
        )
    assert len(resp[0].data.docs[0].matches) == TOP_K

    doc_array = DocumentArray([Document(id=0), Document(id=1), Document(id=2)])
    with flow:
        resp = flow.post(
            on='/fill_embedding',
            inputs=doc_array,
            return_results=True
        )
    for i, doc in enumerate(resp[0].data.docs):
        assert doc.embedding
        assert doc.embedding.dense.shape == [7]
