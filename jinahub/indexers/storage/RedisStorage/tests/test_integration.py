from jina import Flow, DocumentArray, Document
from pytest_mock_resources import create_redis_fixture

from .. import RedisStorage

redis = create_redis_fixture()


def test_flow(docs, redis_kwargs):
    f = Flow().add(uses=RedisStorage, uses_with={'hostname': redis_kwargs['host'], 'port': redis_kwargs['port']})
    with f:
        f.post(on='/index', inputs=docs)
        resp = f.post(on='/search', inputs=DocumentArray([Document(id=doc.id) for doc in docs]), return_results=True)

    assert len(resp[0].docs) == len(docs)
    assert all(doc_a.id == doc_b.id for doc_a, doc_b in zip(resp[0].docs, docs))
