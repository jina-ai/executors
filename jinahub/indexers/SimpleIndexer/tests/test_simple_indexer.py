from copy import deepcopy
from pathlib import Path

import pytest
import numpy as np
from jina import Document, DocumentArray, Executor, Flow

from ..simple_indexer import SimpleIndexer


def assert_document_arrays_equal(arr1, arr2):
    assert len(arr1) == len(arr2)
    for d1, d2 in zip(arr1, arr2):
        assert d1.id == d2.id
        assert d1.content == d2.content
        assert d1.chunks == d2.chunks
        assert d1.matches == d2.matches


@pytest.fixture
def docs():
    doc1 = Document(id='doc1', embedding=np.array([0, 0, 0, 0]))
    doc1.chunks.append(Document(id='doc1-chunk1', embedding=np.array([1, 0, 0, 0])))
    doc1.chunks.append(Document(id='doc1-chunk2', embedding=np.array([0, 1, 0, 0])))
    doc1.chunks.append(Document(id='doc1-chunk3', embedding=np.array([0, 1, 0, 1])))

    doc2 = Document(id='doc2', embedding=np.array([1, 1, 1, 1]))
    doc2.chunks.append(Document(id='doc2-chunk1', embedding=np.array([0, 0, 1, 0])))
    doc2.chunks.append(Document(id='doc2-chunk2', embedding=np.array([0, 0, 0, 1])))
    doc2.chunks.append(Document(id='doc2-chunk3', embedding=np.array([0, 1, 0, 1])))
    return DocumentArray([doc1, doc2])


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[1] / 'config.yml'))
    assert ex.default_top_k == 5


def test_simple_indexer_flow(tmpdir):
    f = Flow().add(
        uses=SimpleIndexer,
        uses_with={'index_file_name': 'name'},
        uses_metas={'workspace': str(tmpdir)},
    )

    with f:
        resp = f.post(
            on='/index',
            inputs=[Document(id='a', embedding=np.array([1]))],
            return_results=True,
        )
        print(f'{resp}')
        resp = f.post(
            on='/search',
            inputs=[Document(embedding=np.array([1]))],
            return_results=True,
            parameters={'top_k': 5},
        )
        assert resp[0].docs[0].matches[0].id == 'a'


def test_simple_indexer(tmpdir):
    metas = {'workspace': str(tmpdir)}
    indexer = SimpleIndexer(index_file_name='name', metas=metas)

    assert indexer._flush
    index_docs = DocumentArray([Document(id='a', embedding=np.array([1]))])
    indexer.index(index_docs, {})
    assert indexer._flush

    search_docs = DocumentArray(([Document(embedding=np.array([1]))]))
    indexer.search(
        docs=search_docs,
        parameters={'top_k': 5},
    )
    assert not indexer._flush
    assert search_docs[0].matches[0].id == 'a'

    search_docs_id = DocumentArray([Document(id='a')])
    assert search_docs_id[0].embedding is None
    indexer.fill_embedding(search_docs_id)
    assert search_docs_id[0].embedding is not None


def test_simple_indexer_loading(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}
    docs_indexer1 = SimpleIndexer(index_file_name='docs_indexer', metas=metas)
    docs_indexer1.index(docs)
    docs_indexer2 = SimpleIndexer(index_file_name='docs_indexer', metas=metas)
    assert_document_arrays_equal(docs_indexer2._docs, docs)


def test_simple_indexer_index(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}

    # test general/normal case
    docs_indexer = SimpleIndexer(index_file_name='normal', metas=metas)
    docs_indexer.index(docs)
    assert_document_arrays_equal(docs_indexer._docs, docs)

    # test index empty docs
    docs_indexer = SimpleIndexer(index_file_name='empty', metas=metas)
    docs_indexer.index(DocumentArray())
    assert not docs_indexer._docs

    # test index empty flat docs
    docs_indexer.index(docs, parameters={'traversal_paths': ['cc']})
    assert not docs_indexer._docs

    # test index docs using default traversal paths
    docs_indexer = SimpleIndexer(
        index_file_name='default_traversal_paths',
        default_traversal_paths=['c'],
        metas=metas,
    )

    docs_indexer.index(docs)
    assert_document_arrays_equal(docs_indexer._docs, docs.traverse_flat(['c']))

    # test index docs by passing traversal paths as a parameter
    docs_indexer = SimpleIndexer(index_file_name='params_traversal_paths', metas=metas)
    docs_indexer.index(docs, parameters={'traversal_paths': ['c']})
    assert_document_arrays_equal(docs_indexer._docs, docs.traverse_flat(['c']))


def test_simple_indexer_search(tmpdir, docs):
    metas = {'workspace': str(tmpdir)}

    # test general/normal case
    indexer = SimpleIndexer(index_file_name='search_normal', metas=metas)
    indexer.index(docs)
    search_docs = deepcopy(docs)
    indexer.search(search_docs)
    assert search_docs[0].matches[0].id == 'doc1'
    assert search_docs[1].matches[0].id == 'doc2'

    # test index empty docs
    indexer = SimpleIndexer(
        index_file_name='search_empty', default_traversal_paths=['c'], metas=metas
    )

    indexer.index(docs)
    search_docs = DocumentArray()
    indexer.search(search_docs)
    assert not search_docs

    # test search empty flat docs
    search_docs = deepcopy(docs)
    indexer.search(search_docs, parameters={'traversal_paths': ['cc']})
    assert_document_arrays_equal(search_docs, docs)

    # test search with default traversal_paths
    search_docs = deepcopy(docs)
    indexer.search(search_docs)
    assert search_docs[0].chunks[0].matches[0].id == 'doc1-chunk1'
    assert search_docs[0].chunks[1].matches[0].id == 'doc1-chunk2'
    assert search_docs[1].chunks[0].matches[0].id == 'doc2-chunk1'
    assert search_docs[1].chunks[1].matches[0].id == 'doc2-chunk2'

    # test search by passing traversal_paths as a parameter
    indexer = SimpleIndexer(index_file_name='params_traversal_paths', metas=metas)
    indexer.index(docs, parameters={'traversal_paths': ['c']})
    search_docs = deepcopy(search_docs)
    indexer.search(search_docs, parameters={'traversal_paths': ['c']})
    assert search_docs[0].chunks[0].matches[0].id == 'doc1-chunk1'
    assert search_docs[0].chunks[1].matches[0].id == 'doc1-chunk2'
    assert search_docs[1].chunks[0].matches[0].id == 'doc2-chunk1'
    assert search_docs[1].chunks[1].matches[0].id == 'doc2-chunk2'
