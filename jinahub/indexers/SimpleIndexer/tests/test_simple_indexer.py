import numpy as np
from jina import Flow, Document, DocumentArray

from .. import SimpleIndexer


def test_simple_indexer_flow(tmpdir):
    f = Flow().add(
        uses=SimpleIndexer,
        override_with={'index_file_name': 'name'},
        override_metas={'workspace': str(tmpdir)},
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


def test_aggregate(tmpdir):
    metas = {'workspace': str(tmpdir)}
    indexer = SimpleIndexer(index_file_name='name', aggregate=True, default_traversal_paths=['c'], metas=metas)

    idx1 = Document(id='idx1')
    idx1.chunks.append(Document(id='idx1-chunk1'), embedding=np.array([1, 0, 0, 0]))
    idx1.chunks.append(Document(id='idx1-chunk2'), embedding=np.array([0, 1, 0, 0]))

    idx2 = Document(id='idx2')
    idx2.chunks.append(Document(id='idx2-chunk1'), embedding=np.array([0, 0, 1, 0]))
    idx2.chunks.append(Document(id='idx2-chunk2'), embedding=np.array([0, 0, 0, 1]))

    q1 = Document(id='q1')
    q1.chunks.append(Document(id='q1-chunk1'), embedding=np.array([1, 0, 0, 0]))
    q1.chunks.append(Document(id='q1-chunk2'), embedding=np.array([0, 1, 0, 0]))

    q2 = Document(id='q2')
    q2.chunks.append(Document(id='q2-chunk1'), embedding=np.array([0, 0, 1, 0]))
    q2.chunks.append(Document(id='q2-chunk2'), embedding=np.array([0, 0, 0, 1]))

    indexer.index(DocumentArray([idx1, idx2]), {})
    indexer.search(DocumentArray([q1, q2]), {})

    assert q1.matches[0].id == 'idx1'
    assert q2.matches[0].id == 'idx2'
