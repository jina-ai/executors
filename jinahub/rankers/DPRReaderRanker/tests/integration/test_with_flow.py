__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
from jina import Document, DocumentArray, Flow

from ...dpr_reader import DPRReaderRanker


@pytest.mark.parametrize('request_size', [1, 8, 50])
def test_integration(request_size: int):
    docs = DocumentArray(
        [
            Document(
                text='just some random text here',
                matches=[Document(text='random text', tags={'title': 'random title'})],
            )
            for _ in range(50)
        ]
    )
    with Flow(return_results=True).add(uses=DPRReaderRanker) as flow:
        resp = flow.post(
            on='/search',
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 50
