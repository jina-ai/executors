import os

import pytest as pytest
from jina import Document, DocumentArray, Flow

from ..redis_storage import RedisStorage

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(os.path.join(cur_dir, 'docker-compose.yml'))


@pytest.mark.parametrize('docker_compose', [compose_yml], indirect=['docker_compose'])
def test_flow(docs, docker_compose):
    f = Flow().add(uses=RedisStorage)
    with f:
        f.post(on='/index', inputs=docs)
        resp = f.post(
            on='/search',
            inputs=DocumentArray([Document(id=doc.id) for doc in docs]),
            return_results=True,
        )

    assert len(resp[0].docs) == len(docs)
    assert all(doc_a.id == doc_b.id for doc_a, doc_b in zip(resp[0].docs, docs))
