import random

import pytest
from jina import Document, DocumentArray
from lightgbm_ranker import LightGBMRanker

NUM_DOCS = 1000
NUM_MATCHES = 5


@pytest.fixture
def ranker():
    return LightGBMRanker(
        query_features=['brand_query', 'price_query'],
        match_features=['brand_match', 'price_match'],
        relevance_label='relevance',
    )


@pytest.fixture
def ranker_with_categorical_features():
    return LightGBMRanker(
        query_features=['price_query'],
        match_features=['price_match'],
        relevance_label='relevance',
        categorical_query_features=['brand_query'],
        categorical_match_features=['brand_match'],
    )


@pytest.fixture
def documents_to_train_price_sensitive_model():
    """features: color, brand, price. Label relevance"""
    # price sensitive, relevance based on pure price, cheaper relevance higher.
    da = DocumentArray()
    for _ in range(NUM_DOCS):
        root = Document(
            tags={'price_query': random.randint(200, 500), 'brand_query': 1}
        )
        for _ in range(NUM_MATCHES):
            root_price = root.tags['price_query']
            root.matches.extend(
                [
                    Document(
                        tags={
                            'price_match': root_price - 100,
                            'brand_match': 3,
                            'relevance': 10,
                        }
                    ),
                    Document(
                        tags={
                            'price_match': root_price,
                            'brand_match': 3,
                            'relevance': 6,
                        }
                    ),
                    Document(
                        tags={
                            'price_match': root_price + 100,
                            'brand_match': 3,
                            'relevance': 4,
                        }
                    ),
                    Document(
                        tags={
                            'price_match': root_price + 200,
                            'brand_match': 3,
                            'relevance': 2,
                        }
                    ),
                ]
            )
        da.append(root)
    return da


@pytest.fixture
def documents_random_brand():
    """features: color, brand, price. Label relevance"""
    # expect price
    da = DocumentArray()
    d1 = Document(tags={'brand_query': 2, 'price_query': 200})
    d1.matches.append(
        Document(id=1, tags={'brand_match': 2, 'price_match': 405, 'relevance': 3})
    )
    d1.matches.append(
        Document(id=2, tags={'brand_match': 2, 'price_match': 305, 'relevance': 3})
    )
    d1.matches.append(
        Document(id=3, tags={'brand_match': 2, 'price_match': 96, 'relevance': 3})
    )
    d1.matches.append(
        Document(id=4, tags={'brand_match': 2, 'price_match': 200, 'relevance': 3})
    )
    da.append(d1)
    return da
