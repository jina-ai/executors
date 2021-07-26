__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Callable, List
import pytest

from jina import Flow, DocumentArray
from jinahub.text.encoders.sentence_encoder import TransformerSentenceEncoder


@pytest.mark.parametrize(
    'request_size', [1, 10, 50, 100]
)
def test_integration(
    data_generator: Callable,
    request_size: int
):
    with Flow().add(uses=TransformerSentenceEncoder) as flow:
        resp = flow.post(on='/index', inputs=data_generator(), request_size=request_size, return_results=True)

    assert min(len(resp) * request_size, 50) == 50

    for r in resp:
        for doc in r.docs:
            assert doc.embedding is not None
