import random

import pytest
import numpy as np
from jina import Document, DocumentArray

from ..lightgbm_ranker import LightGBMRanker

NUM_DOCS = 1000
NUM_MATCHES = 5


@pytest.fixture
def ranker():
    return LightGBMRanker(
        query_features=['brand', 'price'],
        match_features=['brand', 'price'],
        relevance_label='relevance',
    )


@pytest.fixture
def relevances():
    return np.random.uniform(0, 1, [1, NUM_DOCS]).flatten()


@pytest.fixture
def documents_to_train_stub_model(relevances):
    """features: color, brand, price. Label relevance"""
    # initial stub model, relevance purely dependent on brand, not price.
    # brand relevance 5 > 4 > 3 > 2 > 1.
    da = DocumentArray()
    bins = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    inds = np.digitize(relevances, bins)
    for brand, relevance in zip(inds, relevances):
        doc = Document(
            tags={
                'brand': int(brand),
                'price': random.randint(50, 200),
                'weight': random.uniform(0, 1),
            }
        )
        for _ in range(NUM_MATCHES):
            # each match has an extra relevance field indicates score.
            doc.matches.append(
                Document(
                    tags={
                        'brand': int(brand),
                        'price': random.randint(50, 200),
                        'relevance': float(relevance),
                    }
                )
            )
        da.append(doc)
    return da


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
                        tags={'price': root_price - 100, 'brand': 3, 'relevance': 0.8}
                    ),
                    Document(tags={'price': root_price, 'brand': 3, 'relevance': 0.6}),
                    Document(
                        tags={'price': root_price + 100, 'brand': 3, 'relevance': 0.4}
                    ),
                    Document(
                        tags={'price': root_price + 200, 'brand': 3, 'relevance': 0.2}
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
