import random

import pytest
from jina import DocumentArray, Document
from minranker import MinRanker


@pytest.fixture
def documents_chunk():
    document_array = DocumentArray()
    document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for i in range(0, 10):
        chunk = Document()
        for j in range(0, 10):
            match = Document(
                tags={
                    'level': 'chunk',
                }
            )
            match.scores['cosine'] = random.random()
            match.parent_id = i
            chunk.matches.append(match)
        document.chunks.append(chunk)

    document_array.extend([document])
    return document_array


@pytest.fixture
def documents_chunk_chunk():
    document_array = DocumentArray()
    document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for i in range(0, 10):
        chunk = Document()
        for j in range(0, 10):
            chunk_chunk = Document()
            for k in range(0,10):

                match = Document(
                    tags={
                        'level': 'chunk',
                    }
                )
                match.scores['cosine'] = random.random()
                match.parent_id = j
                chunk_chunk.matches.append(match)
            chunk.chunks.append(chunk_chunk)
        document.chunks.append(chunk)

    document_array.extend([document])
    return document_array


@pytest.mark.parametrize('default_traversal_paths', [['r'], ['c']])
def test_ranker(documents_chunk, documents_chunk_chunk, default_traversal_paths):
    ranker = MinRanker(metric='cosine', default_traversal_paths=default_traversal_paths)
    if default_traversal_paths == ['r']:
        ranking_docs = documents_chunk
    else:
        ranking_docs = documents_chunk_chunk

    ranker.rank(ranking_docs, parameters={})
    assert ranking_docs

    for doc in ranking_docs.traverse_flat(default_traversal_paths):
        assert doc.matches
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            assert match.scores['cosine'].value >= doc.matches[i + 1].scores['cosine'].value
