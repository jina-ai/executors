__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path
from typing import List

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor
from simpleranker import SimpleRanker


def test_config():
    encoder = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert encoder.metric == 'cosine'


def test_no_docs():
    SimpleRanker().rank(None, {})


def test_empty_docs():
    docs = DocumentArray([])
    SimpleRanker().rank(docs, {})
    assert len(docs) == 0


def test_docs_no_matched_chunks():
    docs = DocumentArray([Document()])
    SimpleRanker().rank(docs, {})
    assert len(docs) == 1
    assert len(docs[0].matches) == 0


def test_wrong_ranking():
    with pytest.raises(ValueError, match='ranking should be'):
        SimpleRanker(ranking='wrong')


@pytest.mark.parametrize('ranking', ['min', 'max', 'mean_min', 'mean_max'])
def test_ranking_metrics(ranking: str):
    query = Document()
    chunks = [Document(), Document()]
    query.chunks = chunks

    query.chunks[0].matches = [
        Document(scores={'cosine': 0.1}, parent_id='1'),
        Document(scores={'cosine': 0.3}, parent_id='2'),
    ]
    query.chunks[1].matches = [
        Document(scores={'cosine': 0.2}, parent_id='1'),
        Document(scores={'cosine': 0.5}, parent_id='2'),
    ]

    ranker = SimpleRanker(ranking=ranking)
    ranker.rank(DocumentArray([query]), {})

    expected_id = {'min': '1', 'mean_min': '1', 'max': '2', 'mean_max': '2'}
    assert query.matches[0].id == expected_id[ranking]

    expected_scores_1 = {'min': 0.1, 'max': 0.5, 'mean_min': 0.15, 'mean_max': 0.4}
    expected_scores_2 = {'min': 0.3, 'max': 0.2, 'mean_min': 0.4, 'mean_max': 0.15}
    np.testing.assert_almost_equal(
        query.matches[0].scores['cosine'].value, expected_scores_1[ranking]
    )
    np.testing.assert_almost_equal(
        query.matches[1].scores['cosine'].value, expected_scores_2[ranking]
    )


@pytest.mark.parametrize('traversal_paths', (['r'], ['c']))
def test_traversal_paths(traversal_paths: List[str]):
    docs = DocumentArray([Document()])
    docs[0].chunks = [Document()]
    matches = [
        Document(scores={'cosine': 0.1}, parent_id='1'),
        Document(scores={'cosine': 0.3}, parent_id='2'),
        Document(scores={'cosine': 0.2}, parent_id='1'),
        Document(scores={'cosine': 0.5}, parent_id='2'),
    ]

    if traversal_paths == ['r']:
        docs[0].chunks[0].matches = matches
    elif traversal_paths == ['c']:
        docs[0].chunks[0].chunks = [Document()]
        docs[0].chunks[0].chunks[0].matches = matches

    ranker = SimpleRanker(
        matches_path='m', ranking='min', traversal_paths=traversal_paths
    )
    ranker.rank(docs, {})

    for doc in docs.traverse_flat(traversal_paths):
        assert doc.matches[0].id == '1'
        np.testing.assert_almost_equal(doc.matches[0].scores['cosine'].value, 0.1)
        np.testing.assert_almost_equal(doc.matches[1].scores['cosine'].value, 0.3)
