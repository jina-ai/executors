__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os

import pytest
from jina import Flow, Document, DocumentArray

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize('_type', ['wav', 'mp3', 'blob'])
def test_chunks_exists(build_da, _type):
    da = build_da(_type)
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.post(on='segment', inputs=da, return_results=True)

    locations = [
        [0, 56500],
        [69500, 92000],
        [94500, 213000],
        [223500, 270500],
    ]

    assert len(responses[0].docs) == 1
    for doc in responses[0].docs:
        assert len(doc.chunks) == 4
        for chunk, location in zip(doc.chunks, locations):
            assert chunk.location == location
