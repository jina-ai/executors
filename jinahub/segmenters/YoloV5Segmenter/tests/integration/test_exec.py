__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

from jina import Flow, Document, DocumentArray

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_chunks_exists(build_da):
    da = build_da()
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.post(on='segment', inputs=da, return_results=True)

    assert len(responses[0].docs) == 3
    for doc in responses[0].docs:
        assert len(doc.chunks) > 0
        for chunk in doc.chunks:
            assert chunk.blob.ndim == 3
            assert chunk.tags.get('label')
            assert chunk.tags.get('conf')
