import os
import time
from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor, Flow
from jina.logging.profile import TimeContext
from jina_commons.indexers.dump import import_metas, import_vectors

from ..postgres_indexer import PostgreSQLStorage
from ..postgreshandler import doc_without_embedding


@pytest.fixture()
def docker_compose(request):
    os.system(
        f"docker-compose -f {request.param} --project-directory . up  --build -d "
        f"--remove-orphans"
    )
    time.sleep(5)
    yield
    os.system(
        f"docker-compose -f {request.param} --project-directory . down "
        f"--remove-orphans"
    )


d_embedding = np.array([1, 1, 1, 1, 1, 1, 1])
c_embedding = np.array([2, 2, 2, 2, 2, 2, 2])

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.fixture(scope='function', autouse=True)
def patched_random_port(mocker):
    used_ports = set()
    from jina.helper import random_port

    def _random_port():

        for i in range(10):
            _port = random_port()

            if _port is not None and _port not in used_ports:
                used_ports.add(_port)
                return _port
        raise Exception('no port available')

    mocker.patch('jina.helper.random_port', new_callable=lambda: _random_port)


def get_documents(chunks, same_content, nr=10, index_start=0, same_tag_content=None):
    next_chunk_id = nr + index_start
    for i in range(index_start, nr + index_start):
        d = Document()
        d.id = i
        if same_content:
            d.text = 'hello world'
            d.embedding = np.random.random(d_embedding.shape)
        else:
            d.text = f'hello world {i}'
            d.embedding = np.random.random(d_embedding.shape)
        if same_tag_content:
            d.tags['field'] = 'tag data'
        elif same_tag_content is False:
            d.tags['field'] = f'tag data {i}'
        for j in range(chunks):
            c = Document()
            c.id = next_chunk_id
            if same_content:
                c.text = 'hello world from chunk'
                c.embedding = np.random.random(c_embedding.shape)
            else:
                c.text = f'hello world from chunk {j}'
                c.embedding = np.random.random(c_embedding.shape)
            if same_tag_content:
                c.tags['field'] = 'tag data'
            elif same_tag_content is False:
                c.tags['field'] = f'tag data {next_chunk_id}'
            next_chunk_id += 1
            d.chunks.append(c)
        yield d


def validate_db_side(postgres_indexer, expected_data):
    ids, vecs, metas = zip(*expected_data)
    with postgres_indexer.handler as handler:
        cursor = handler.connection.cursor()
        cursor.execute(
            f'SELECT doc_id, embedding, doc from {postgres_indexer.table} ORDER BY '
            f'doc_id::int'
        )
        record = cursor.fetchall()
        for i in range(len(expected_data)):
            np.testing.assert_equal(ids[i], str(record[i][0]))
            embedding = np.frombuffer(record[i][1], dtype=postgres_indexer.dump_dtype)
            np.testing.assert_equal(vecs[i], embedding)
            np.testing.assert_equal(metas[i], bytes(record[i][2]))


def test_config():
    ex = Executor.load_config(
        str(Path(__file__).parents[1] / 'config.yml'), override_with={'dry_run': True}
    )
    assert ex.username == 'postgres'


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_postgres(tmpdir, docker_compose):
    postgres_indexer = PostgreSQLStorage()
    NR_DOCS = 10000
    original_docs = DocumentArray(
        list(get_documents(nr=NR_DOCS, chunks=0, same_content=False))
    )

    postgres_indexer.delete(original_docs, {})

    with TimeContext(f'### indexing {len(original_docs)} docs'):
        postgres_indexer.add(original_docs, {})
    np.testing.assert_equal(postgres_indexer.size, NR_DOCS)

    info_original_docs = [
        (doc.id, doc.embedding, doc_without_embedding(doc)) for doc in original_docs
    ]
    validate_db_side(postgres_indexer, info_original_docs)

    new_docs = DocumentArray(
        list(get_documents(chunks=False, nr=10, same_content=True))
    )
    postgres_indexer.update(new_docs, {})

    info_new_docs = [
        (doc.id, doc.embedding, doc_without_embedding(doc)) for doc in new_docs
    ]
    ids, vecs, metas = zip(*info_new_docs)
    expected_info = [(ids[0], vecs[0], metas[0])]
    validate_db_side(postgres_indexer, expected_info)

    postgres_indexer.delete(new_docs, {})
    np.testing.assert_equal(postgres_indexer.size, len(original_docs) - len(new_docs))


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_mwu_empty_dump(tmpdir, docker_compose):
    f = Flow().add(uses=PostgreSQLStorage)

    with f:
        resp = f.post(
            on='/index', inputs=DocumentArray([Document()]), return_results=True
        )
        print(f'{resp}')

    dump_path = os.path.join(tmpdir, 'dump')

    with f:
        f.post(
            on='/dump',
            parameters={'dump_path': os.path.join(tmpdir, 'dump'), 'shards': 1},
        )

    # assert dump contents
    ids, vecs = import_vectors(dump_path, pea_id='0')
    assert ids is not None
    ids, metas = import_metas(dump_path, pea_id='0')
    assert vecs is not None
    assert metas is not None


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_return_embeddings(docker_compose):
    indexer = PostgreSQLStorage()
    doc = Document(embedding=np.random.random(10))
    da = DocumentArray([doc])
    query1 = DocumentArray([Document(id=doc.id)])
    indexer.add(da, parameters={})
    indexer.search(query1, parameters={})
    assert query1[0].embedding is not None
    assert query1[0].embedding.shape == (10,)

    query2 = DocumentArray([Document(id=doc.id)])
    indexer.search(query2, parameters={"return_embeddings": False})
    assert query2[0].embedding is None


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
@pytest.mark.parametrize('psql_virtual_shards', [44, 128])
@pytest.mark.parametrize('real_shards', [1, 5])
def test_snapshot(docker_compose, psql_virtual_shards, real_shards):
    postgres_indexer = PostgreSQLStorage(virtual_shards=psql_virtual_shards)

    def _assert_snapshot_shard_distribution(func, nr_shards, total_docs_expected):
        total_docs = 0
        for i in range(nr_shards):
            data = func(shard_id=i, total_shards=nr_shards)
            docs_this_shard = len(list(data))
            assert docs_this_shard >= postgres_indexer.virtual_shards // real_shards
            total_docs += docs_this_shard

        np.testing.assert_equal(total_docs, total_docs_expected)

    NR_SHARDS = real_shards
    NR_DOCS = postgres_indexer.virtual_shards * 2 + 3
    original_docs = DocumentArray(
        list(get_documents(nr=NR_DOCS, chunks=0, same_content=False))
    )

    NR_NEW_DOCS = 30
    new_docs = DocumentArray(
        list(
            get_documents(
                nr=NR_NEW_DOCS, index_start=NR_DOCS, chunks=0, same_content=False
            )
        )
    )

    # make sure to cleanup if the PSQL instance is kept running
    postgres_indexer.delete(original_docs, {})
    postgres_indexer.delete(new_docs, {})

    # indexing the documents
    postgres_indexer.add(original_docs, {})
    np.testing.assert_equal(postgres_indexer.size, NR_DOCS)

    # create a snapshot
    postgres_indexer.snapshot()

    # data added the snapshot will not be part of the export
    postgres_indexer.add(new_docs, {})

    np.testing.assert_equal(postgres_indexer.size, NR_DOCS + NR_NEW_DOCS)
    np.testing.assert_equal(postgres_indexer.snapshot_size, NR_DOCS)

    _assert_snapshot_shard_distribution(
        postgres_indexer.get_snapshot, NR_SHARDS, NR_DOCS
    )

    # create another snapshot
    postgres_indexer.snapshot()
    timestamp = postgres_indexer.last_snapshot_timestamp

    # docs for the delta resolving
    NR_DOCS_DELTA = 33
    docs_delta = DocumentArray(
        list(
            get_documents(
                nr=NR_DOCS_DELTA,
                index_start=NR_DOCS + NR_NEW_DOCS,
                chunks=0,
                same_content=False,
            )
        )
    )
    time.sleep(3)
    postgres_indexer.add(docs_delta, {})

    np.testing.assert_equal(
        postgres_indexer.size, NR_DOCS + NR_NEW_DOCS + NR_DOCS_DELTA
    )
    np.testing.assert_equal(postgres_indexer.snapshot_size, NR_DOCS + NR_NEW_DOCS)

    NR_DOCS_DELTA_DELETED = 10
    docs_delta_deleted = DocumentArray(
        list(
            get_documents(
                nr=NR_DOCS_DELTA_DELETED, index_start=0, chunks=0, same_content=False
            )
        )
    )
    postgres_indexer.delete(docs_delta_deleted, {'soft_delete': True})

    _assert_snapshot_shard_distribution(
        postgres_indexer.get_snapshot,
        NR_SHARDS,
        NR_DOCS + NR_NEW_DOCS,
    )

    # we use total_shards=1 in order to guarantee getting all the data in the delta
    deltas = postgres_indexer._get_delta(
        shard_id=0, total_shards=1, timestamp=timestamp
    )
    deltas = list(deltas)
    np.testing.assert_equal(len(deltas), NR_DOCS_DELTA + NR_DOCS_DELTA_DELETED)

    # TODO delta is then passed to the indexer
    # after soft-deletion is added to Faiss


def test_postgres_shard_distribution():
    assert ['0'] == PostgreSQLStorage._vshards_to_get(0, 3, 5)
    assert ['1'] == PostgreSQLStorage._vshards_to_get(1, 3, 5)
    assert ['2', '3', '4'] == PostgreSQLStorage._vshards_to_get(2, 3, 5)
    assert [str(s) for s in range(5)] == PostgreSQLStorage._vshards_to_get(0, 1, 5)
    with pytest.raises(ValueError):
        PostgreSQLStorage._vshards_to_get(1, 1, 5)
