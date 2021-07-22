import os

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow
from jina.logging.profile import TimeContext

from jina_commons.indexers.dump import import_metas, import_vectors
from .. import LMDBStorage

np.random.seed(0)
d_embedding = np.array([1, 1, 1, 1, 1, 1, 1])
c_embedding = np.array([2, 2, 2, 2, 2, 2, 2])


def get_documents(nr=10, index_start=0, emb_size=7, text='hello world'):
    docs = []
    for i in range(index_start, nr + index_start):
        with Document() as d:
            d.id = i
            d.text = f'{text} {i}'
            d.embedding = np.random.random(emb_size)
            d.tags['field'] = f'tag data {i}'
        docs.append(d)
    return DocumentArray(docs)


def test_lmdb_crud(tmpdir, nr_docs=10):
    docs = get_documents(nr=nr_docs)

    metas = {'workspace': str(tmpdir), 'name': 'storage', 'pea_id': 0}

    # indexing
    indexer = LMDBStorage(map_size=10485760 * 1000, metas=metas)
    indexer.index(docs, {})
    assert indexer.size == len(docs)

    query_docs = DocumentArray([Document(id=id) for id in [d.id for d in docs]])
    indexer.search(query_docs, {})
    for q, d in zip(query_docs, docs):
        assert d.id == q.id
        assert d.text == q.text
        np.testing.assert_equal(d.embedding, q.embedding)

    # getting size
    items = indexer.size

    # updating
    update_docs = get_documents(nr=nr_docs, text='hello there')
    indexer.update(update_docs, {})

    query_docs = DocumentArray([Document(id=id) for id in [d.id for d in docs]])
    indexer.search(query_docs, {})
    for q, d in zip(query_docs, update_docs):
        assert d.id == q.id
        assert d.text == q.text
        np.testing.assert_equal(d.embedding, q.embedding)

    # asserting...
    assert indexer.size == items

    indexer.delete(docs, {})
    assert indexer.size == 0


def test_lmdb_crud_flow(tmpdir):
    metas = {'workspace': str(tmpdir), 'name': 'storage'}
    runtime_args = {'pea_id': 0, 'replica_id': None}

    def _get_flow() -> Flow:
        return Flow().add(
            uses={
                'jtype': 'LMDBStorage',
                'with': {},
                'metas': metas,
            }
        )

    docs = get_documents(nr=10)
    update_docs = get_documents(nr=10, text='hello there')

    # indexing
    with _get_flow() as f:
        f.index(
            inputs=docs,
            parameters={
                'storage': {'traversal_paths': ['r']},
            },
        )

    # getting size
    with LMDBStorage(metas=metas, runtime_args=runtime_args) as indexer:
        items = indexer.size

    assert items == len(docs)

    # updating
    with _get_flow() as f:
        f.post(
            on='/update',
            inputs=update_docs,
            parameters={
                'storage': {'traversal_paths': ['r']},
            },
        )

    # asserting...
    with LMDBStorage(metas=metas, runtime_args=runtime_args) as indexer:
        assert indexer.size == items


def _in_docker():
    """ Returns: True if running in a Docker container, else False """
    with open('/proc/1/cgroup', 'rt') as ifh:
        if 'docker' in ifh.read():
            print('in docker, skipping benchmark')
            return True
        return False


# benchmark only
@pytest.mark.skipif(
    _in_docker() or ('GITHUB_WORKFLOW' in os.environ),
    reason='skip the benchmark test on github workflow or docker',
)
def test_lmdb_bm(tmpdir):
    nr = 100000
    # Cristian: running lmdb benchmark with 100000 docs ...	running lmdb benchmark with 100000 docs takes 1 minute and 53 seconds
    with TimeContext(f'running lmdb benchmark with {nr} docs'):
        test_lmdb_crud(tmpdir, nr)


def _doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


def _assert_dump_data(dump_path, docs, shards, pea_id):
    size_shard = len(docs) // shards
    size_shard_modulus = len(docs) % shards
    ids_dump, vectors_dump = import_vectors(
        dump_path,
        str(pea_id),
    )
    if pea_id == shards - 1:
        docs_expected = docs[
            (pea_id) * size_shard : (pea_id + 1) * size_shard + size_shard_modulus
        ]
    else:
        docs_expected = docs[(pea_id) * size_shard : (pea_id + 1) * size_shard]
    print(f'### pea {pea_id} has {len(docs_expected)} docs')

    # TODO these might fail if we implement any ordering of elements on dumping / reloading
    ids_dump = list(ids_dump)
    vectors_dump = list(vectors_dump)
    np.testing.assert_equal(ids_dump, [d.id for d in docs_expected])
    np.testing.assert_allclose(vectors_dump, [d.embedding for d in docs_expected])

    _, metas_dump = import_metas(
        dump_path,
        str(pea_id),
    )
    metas_dump = list(metas_dump)
    np.testing.assert_equal(
        metas_dump,
        [_doc_without_embedding(d) for d in docs_expected],
    )


@pytest.mark.parametrize('shards', [2, 3, 7])
def test_dump(tmpdir, shards):
    metas = {'workspace': str(tmpdir), 'name': 'storage'}
    dump_path = os.path.join(tmpdir, 'dump_dir')

    def _get_flow() -> Flow:
        return Flow().add(
            uses={
                'jtype': 'LMDBStorage',
                'with': {},
                'metas': metas,
            }
        )

    docs = get_documents(nr=10)

    # indexing
    with _get_flow() as f:
        f.index(inputs=docs)
        f.post(on='/dump', parameters={'dump_path': dump_path, 'shards': shards})

    for pea_id in range(shards):
        _assert_dump_data(dump_path, docs, shards, pea_id)
