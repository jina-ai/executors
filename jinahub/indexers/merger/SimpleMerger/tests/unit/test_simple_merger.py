__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
from jina import Flow, Document

from jinahub.indexers.merger.SimpleMerger.simple_merger import SimpleMerger


@pytest.fixture
def docs_matrix():
    docs_matrix = [
        Document(
            text=f'doc {i}',
            matches=[
                Document(text=f'doc {i}, match {j}')
                for j in range(3)
            ],
            chunks=[
                Document(
                    text=f'doc {i}, chunk {j}',
                    matches=[
                        Document(
                            text=f'doc {i}, chunk {j}, match {k}'
                        )
                        for k in range(2)
                    ]
                )
                for j in range(3)
            ])
        for i in range(2)
    ]


def test_simple_merger(docs_matrix):
    executor = SimpleMerger(default_traversal_paths=('r',))
    executor.merge(docs_matrix=docs_matrix)
