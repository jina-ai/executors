from jina import Document, DocumentArray
from ..

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


def test_add(indexer, docs, caplog):
    indexer.add(docs, parameters={})
    qh = indexer.get_query_handler()
    redis_keys = qh.keys()
    assert all(doc.id.encode() in redis_keys for doc in docs)

    new_docs = DocumentArray()
    new_docs.append(Document())
    new_docs.append(docs[0])
    indexer.add(new_docs, parameters={})
    assert f'The following IDs already exist: {docs[0].id}' in caplog.text


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
