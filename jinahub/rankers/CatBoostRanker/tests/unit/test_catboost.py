import pytest

from ...catboost_ranker import CatBoostRanker


def test_dump_load():
    assert 1
    ran = CatBoostRanker(
        query_features=['brand', 'price'],
        document_features=['brand', 'price'],
        label='relevance',
    )
