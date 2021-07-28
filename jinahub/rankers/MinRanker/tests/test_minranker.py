import random

import pytest
from jina import DocumentArray, Document
from minranker import MinRanker


@pytest.fixture
def documents_chunk():
    document_array = DocumentArray()
    document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for j in range(0, 10):
        chunk = Document()
        for i in range(0, 10):
            match = Document(
                tags={
                    'level': 'chunk',
                }
            )
            match.scores['cosine'] = random.random()
            match.parent_id = j
            chunk.matches.append(match)
        document.chunks.append(chunk)

    document_array.extend([document])
    return document_array


@pytest.mark.parametrize('default_traversal_paths', [['r'], ['c']])
def test_ranker(documents_chunk, default_traversal_paths):
    ranker = MinRanker(metric='cosine', default_traversal_paths=default_traversal_paths)
    ranker.rank(documents_chunk, parameters={})
    assert documents_chunk

    for doc in documents_chunk:
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            assert match.scores['cosine'].value >= doc.matches[i + 1].scores['cosine'].value
