import os

import pytest
from jina import Flow

from ...catboost_ranker import CatboostRanker


@pytest.fixture
def flow():
    return Flow().add(
        uses=CatboostRanker,
        uses_with={
            'query_features': ['brand', 'price'],
            'match_features': ['brand', 'price'],
            'relevance_label': 'relevance',
        },
    )


def test_train_dump_load_search_flow(
    flow,
    documents_to_train_price_sensitive_model,
    tmpdir,
    documents_without_label_random_brand,
):
    model_path = str(tmpdir) + '/model.cbm'
    with flow as f:
        f.post('/train', inputs=documents_to_train_price_sensitive_model)
        rv = f.search(documents_without_label_random_brand, return_results=True)
        relevances_before_dump = []
        for doc in rv[0].data.docs:
            for match in doc.matches:
                assert isinstance(match.scores['relevance'].value, float)
                relevances_before_dump.append(match.scores['relevance'].value)
        f.post('/dump', parameters={'model_path': model_path})
        assert os.path.exists(model_path)
        f.post('/load', parameters={'model_path': model_path})
        # ensure after load produce the same result
        rv = f.search(documents_without_label_random_brand, return_results=True)
        relevances_after_dump = []
        for doc in rv[0].data.docs:
            for match in doc.matches:
                assert isinstance(match.scores['relevance'].value, float)
                relevances_after_dump.append(match.scores['relevance'].value)
        assert relevances_before_dump == relevances_after_dump
