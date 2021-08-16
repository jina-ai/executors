__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
from ...simpleranker import SimpleRanker


@pytest.mark.parametrize('default_traversal_paths', [['r'], ['c']])
def test_min_ranking(documents_chunk, documents_chunk_chunk, default_traversal_paths):
    ranker = SimpleRanker(metric='cosine', default_traversal_paths=default_traversal_paths)
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
            assert (
                match.scores['cosine'].value
                <= doc.matches[i + 1].scores['cosine'].value
            )


@pytest.mark.parametrize('default_traversal_paths', [['r'], ['c']])
def test_max_ranking(documents_chunk, documents_chunk_chunk, default_traversal_paths):
    ranker = SimpleRanker(metric='cosine', ranking='max',
                          default_traversal_paths=default_traversal_paths)
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
            assert (
                match.scores['cosine'].value
                >= doc.matches[i + 1].scores['cosine'].value
            )


def test_mean_max_ranking(documents_chunk):
    default_traversal_paths = ['r']
    ranker = SimpleRanker(metric='cosine', ranking='mean_max',
                          default_traversal_paths=default_traversal_paths)
    ranking_docs = documents_chunk

    mean_scores = []
    for doc in ranking_docs[0].chunks:
        scores = []
        for match in doc.matches:
            scores.append(match.scores['cosine'].value)
        mean_scores.append(sum(scores)/10)
    mean_scores.sort(reverse=True)
    ranker.rank(ranking_docs, parameters={})
    assert ranking_docs

    for doc in ranking_docs.traverse_flat(default_traversal_paths):
        assert doc.matches
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            assert match.scores['cosine'].value == pytest.approx(mean_scores[i], 1e-5)


def test_mean_min_ranking(documents_chunk):
    default_traversal_paths = ['r']
    ranker = SimpleRanker(metric='cosine', ranking='mean_min',
                          default_traversal_paths=default_traversal_paths)
    ranking_docs = documents_chunk

    mean_scores = []
    for doc in ranking_docs[0].chunks:
        scores = []
        for match in doc.matches:
            scores.append(match.scores['cosine'].value)
        mean_scores.append(sum(scores)/10)
    mean_scores.sort()
    ranker.rank(ranking_docs, parameters={})
    assert ranking_docs

    for doc in ranking_docs.traverse_flat(default_traversal_paths):
        assert doc.matches
        for i in range(len(doc.matches) - 1):
            match = doc.matches[i]
            assert match.tags
            assert match.scores['cosine'].value == pytest.approx(mean_scores[i], 1e-5)