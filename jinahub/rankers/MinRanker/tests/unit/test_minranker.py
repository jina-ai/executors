__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
from ...minranker import MinRanker


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
