import os

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow

from jinahub.indexers.compound.FaissPostgresIndexer import FaissPostgresIndexer

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.join(cur_dir, 'docker-compose.yml')


def get_documents(nr=10, index_start=0, emb_size=256):
    random_batch = np.random.random([nr, emb_size]).astype(np.float32)
    for i in range(index_start, nr + index_start):
        d = Document()
        d.id = f'aa{i}'  # to test it supports non-int ids
        d.embedding = random_batch[i - index_start]
        yield d


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
@pytest.mark.parametrize('index_key', ['IVF64,PQ32', 'IVF8,PQ8', 'HNSW32'])
def test_online_train(docker_compose, index_key):
    docs = get_documents(10240)
    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': index_key}
    ) as f:
        f.post(on='/index', inputs=docs)
        f.post(on='/train')
        f.post(on='/sync')
        result = f.post(on='/search', inputs=get_documents(10), return_results=True)[0]
        for doc in result.docs:
            assert len(doc.matches) == 10


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_online_train_2_flows(docker_compose):
    docs = get_documents(10240)
    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/index', inputs=docs)

    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/train')
        f.post(on='/sync')
        result = f.post(on='/search', inputs=get_documents(10), return_results=True)[0]
        for doc in result.docs:
            assert len(doc.matches) == 10


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_online_train_3_flows(docker_compose):
    docs = get_documents(10240)
    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/index', inputs=docs)

    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/train')

    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/sync')
        result = f.post(on='/search', inputs=get_documents(10), return_results=True)[0]
        for doc in result.docs:
            assert len(doc.matches) == 10


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_offline_train(tmpdir, docker_compose):
    docs = get_documents(10240)
    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/index', inputs=docs)

    train_docs = DocumentArray(get_documents(10240))
    index_filepath = os.path.join(tmpdir, 'trained_faiss.bin')

    with Flow().add(
        uses='FaissPostgresIndexer',
        uses_with={'index_key': 'IVF64,PQ32', 'trained_index_file': index_filepath},
    ) as f:
        import faiss

        index = faiss.index_factory(256, 'IVF64,PQ32', faiss.METRIC_INNER_PRODUCT)
        index.train(train_docs.embeddings)
        faiss.write_index(index, index_filepath)

        f.post(on='/sync')
        result = f.post(on='/search', inputs=get_documents(10), return_results=True)[0]
        for doc in result.docs:
            assert len(doc.matches) == 10
