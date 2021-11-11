import os

import numpy as np
import pytest
from jina import Document, Flow

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
def test_oneline_train(docker_compose):
    docs = get_documents(512)
    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        f.post(on='/index', inputs=docs)
        f.post(on='/train')

    with Flow().add(
        uses='FaissPostgresIndexer', uses_with={'index_key': 'IVF64,PQ32'}
    ) as f:
        result = f.post(on='/search', inputs=docs, return_results=True)
        print(result[0])
