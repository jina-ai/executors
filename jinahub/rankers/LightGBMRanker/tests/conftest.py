import random

import pytest
from jina import Document, DocumentArray

from ..lightgbm_ranker import LightGBMRanker

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
        root = Document(tags={'price': random.randint(200, 500), 'brand': 1})
        for _ in range(NUM_MATCHES):
            root_price = root.tags['price']
            root.matches.extend(
                [
                    Document(
                        tags={'price': root_price - 100, 'brand': 3, 'relevance': 10}
                    ),
                    Document(tags={'price': root_price, 'brand': 3, 'relevance': 6}),
                    Document(
                        tags={'price': root_price + 100, 'brand': 3, 'relevance': 4}
                    ),
                    Document(
                        tags={'price': root_price + 200, 'brand': 3, 'relevance': 2}
                    ),
                ]
            )
        da.append(root)
    return da


@pytest.fixture
def documents_without_label_random_price():
    """features: color, brand, price. Label relevance"""
    # expect 5 > 3 > 1
    # expect price
    da = DocumentArray()
    d1 = Document(tags={'brand': random.randint(0, 5), 'price': 200})
    d1.matches.append(Document(tags={'brand': random.randint(0, 5), 'price': 196}))
    d1.matches.append(Document(tags={'brand': random.randint(0, 5), 'price': 100}))
    d1.matches.append(Document(tags={'brand': random.randint(0, 5), 'price': 50}))
    da.append(d1)
    return da


@pytest.fixture
def documents_without_label_random_brand():
    """features: color, brand, price. Label relevance"""
    # expect price
    da = DocumentArray()
    d1 = Document(tags={'brand': 2, 'price': 200})
    d1.matches.append(Document(id=1, tags={'brand': 2, 'price': 405}))
    d1.matches.append(Document(id=2, tags={'brand': 2, 'price': 305}))
    d1.matches.append(Document(id=3, tags={'brand': 2, 'price': 96}))
    d1.matches.append(Document(id=4, tags={'brand': 2, 'price': 200}))
    da.append(d1)
    return da
