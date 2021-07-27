import logging

from jina import Document, DocumentArray

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


def test_add(indexer, docs):
    indexer.add(docs, parameters={})

    with indexer.get_query_handler() as redis_handler:
        assert all(doc.id.encode() in redis_handler.keys() for doc in docs)


def test_add_existing(indexer, docs):
    indexer.add(docs, parameters={})
    existing_doc = Document(id=docs[0].id, content="new content")
    indexer.add(DocumentArray([existing_doc]), parameters={})

    with indexer.get_query_handler() as redis_handler:
        result = redis_handler.get(existing_doc.id)
        data = bytes(result)
        retrieved_doc = Document(data)
        assert retrieved_doc.content != existing_doc.content


def test_update(indexer, docs):
    indexer.add(docs, parameters={})

    for doc in docs:
        doc.content = 'new ' + doc.content

    indexer.update(docs, parameters={})

    with indexer.get_query_handler() as redis_handler:
        for doc in docs:
            result = redis_handler.get(doc.id)
            data = bytes(result)
            retrieved_doc = Document(data)
            assert retrieved_doc.content == doc.content


def test_update_non_existing(indexer, docs):
    indexer.update(docs, parameters={})

    with indexer.get_query_handler() as redis_handler:
        assert all(doc.id.encode() not in redis_handler.keys() for doc in docs)


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
