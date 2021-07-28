import numpy as np
from jina import Executor, requests, DocumentArray, Document
import pytest
import random
from minranker import MinRanker


@pytest.fixture
def documents():
    document_array = DocumentArray()
    document = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for j in range(0, 10):
        chunk = Document()
        for i in range(0, 10):
            match = Document(
                tags={
                    'match_size': 35,
                    'match_price': 31 + 2 * i,
                    'match_brand': 1,
                    'relevance': int((100 - i) / 10),
                }
            )
            match.scores['cosine'] = random.random()
            match.parent_id = j
            chunk.matches.append(match)
        document.chunks.append(chunk)

    document_array.extend([document])
    return document_array


def test_ranker(documents):
    ranker = MinRanker(metric='cosine')
    ranker.rank(documents, parameters={})
    assert documents

    for doc in documents:
        for i in range(len(doc.matches)-1):
            match = doc.matches[i]
            assert match.tags
            assert match.scores['cosine'].value >= doc.matches[i+1].scores['cosine'].value
