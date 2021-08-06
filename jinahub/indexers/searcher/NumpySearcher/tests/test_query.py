import os

import numpy as np
import pytest
from jina import Document, DocumentArray

from ..numpy_searcher import NumpySearcher

TOP_K = 5
cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture()
def query_docs():
    chunks = DocumentArray([Document(embedding=np.random.random(7))])
    root_doc = Document(embedding=np.random.random(7))
    root_doc.chunks.extend(chunks)
    docs = DocumentArray()
    docs.append(root_doc)
    return docs


@pytest.mark.parametrize('default_traversal_paths', [['r'], ['c']])
def test_query_vector(tmpdir, query_docs, default_traversal_paths):
    runtime = {
        'workspace': str(tmpdir),
        'name': 'searcher',
        'pea_id': 0,
        'replica_id': 0,
    }
    dump_path = os.path.join(cur_dir, 'dump1')
    indexer = NumpySearcher(dump_path=dump_path, default_traversal_paths=default_traversal_paths, runtime_args=runtime)
    indexer.search(query_docs, {'top_k': TOP_K})
    assert len(query_docs) == 1
    doc_traversal = query_docs.traverse_flat(default_traversal_paths)
    assert len(doc_traversal[0].matches) == TOP_K
    assert len(doc_traversal[0].matches[0].embedding) == 7


@pytest.mark.parametrize(['metric', 'is_distance'],
                         [('cosine', True), ('euclidean', True),
                          ('cosine', False), ('euclidean', False)])
def test_metric(tmpdir, query_docs, metric, is_distance):
    runtime = {
        'workspace': str(tmpdir),
        'name': 'searcher',
        'pea_id': 0,
        'replica_id': 0,
    }

    dump_path = os.path.join(cur_dir, 'dump1')
    indexer = NumpySearcher(dump_path=dump_path, default_top_k=TOP_K, runtime_args=runtime, metric=metric,
                            is_distance=is_distance)

    indexer.search(query_docs, {})
    assert len(query_docs[0].matches) == TOP_K

    for i in range(len(query_docs[0].matches) - 1):
        if not is_distance:
            assert query_docs[0].matches[i].scores[metric].value >= query_docs[0].matches[i + 1].scores[metric].value
        else:
            assert query_docs[0].matches[i].scores[metric].value <= query_docs[0].matches[i + 1].scores[metric].value


def test_empty_shard(tmpdir, query_docs):
    runtime = {
        'workspace': str(tmpdir),
        'name': 'searcher',
        'pea_id': 0,
        'replica_id': 0,
    }
    indexer = NumpySearcher(dump_path='tests/dump_empty', runtime_args=runtime)
    indexer.search(query_docs, {'top_k': TOP_K})
    assert len(query_docs) == 1
    assert len(query_docs[0].matches) == 0


def test_empty_documents(tmpdir):
    runtime = {
        'workspace': str(tmpdir),
        'name': 'searcher',
        'pea_id': 0,
        'replica_id': 0,
    }
    indexer = NumpySearcher(dump_path='tests/dump1', runtime_args=runtime)
    docs = DocumentArray([Document()])
    indexer.search(docs, {'top_k': TOP_K})
    assert len(docs) == 1
    assert len(docs[0].matches) == 0

    docs2 = DocumentArray()
    indexer.search(docs2, {'top_k': TOP_K})
    assert len(docs2) == 0
