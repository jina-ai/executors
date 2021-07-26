import pytest as pytest
from jina import Document, DocumentArray

from pytest_mock_resources import create_redis_fixture

from redis_storage import RedisStorage

redis = create_redis_fixture()


@pytest.fixture
def indexer(redis):
    kwargs = redis.pmr_credentials.as_redis_kwargs()
    return RedisStorage(hostname=kwargs['host'], port=kwargs['port'])


@pytest.fixture
def docs():
    return DocumentArray([
        Document(content=value)
        for value in ['cat', 'dog', 'crow', 'pikachu', 'magikarp']
    ])


def test_connection(indexer):
    assert indexer.hostname == 'localhost'
    assert indexer.get_query_handler().ping()


def test_upsert(indexer, docs):
    indexer.upsert(docs, parameters={})
    qh = indexer.get_query_handler()
    redis_keys = qh.keys()
    assert all(doc.id.encode() in redis_keys for doc in docs)


def test_search(indexer, docs):
    indexer.upsert(docs, parameters={})
    query = DocumentArray([Document(id=doc.id) for doc in docs])
    indexer.search(query, parameters={})
    assert all(query_doc.content == doc.content for query_doc, doc in zip(query, docs))


def test_upsert_with_duplicates(indexer, docs):
    # insert same docs twice
    indexer.upsert(docs, parameters={})
    indexer.upsert(docs, parameters={})

    qh = indexer.get_query_handler()
    assert len(qh.keys()) == 5


def test_search_not_found(indexer, docs):
    indexer.upsert(docs, parameters={})

    query = DocumentArray([Document(id=docs[0].id), Document()])
    indexer.search(query, parameters={})
    assert query[0].content == docs[0].content
    assert query[1].content is None


def test_delete(indexer, docs):
    indexer.upsert(docs, parameters={})
    indexer.delete(docs[:2], parameters={})
    query = DocumentArray([Document(id=doc.id) for doc in docs])
    indexer.search(query, parameters={})
    assert all(query_doc.content is None for query_doc in query[:2])
    assert all(query_doc.content == doc.content for query_doc, doc in zip(query[2:], docs[2:]))

    qh = indexer.get_query_handler()
    assert len(qh.keys()) == 3
