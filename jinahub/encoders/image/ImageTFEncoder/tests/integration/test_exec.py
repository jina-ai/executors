__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import List

import numpy as np
import pytest
from jina import Flow, Document, DocumentArray
from jinahub.encoder.image_tf_encoder import ImageTFEncoder

input_dim = 336
target_output_dim = 1280

@pytest.mark.parametrize('arr_in', [
    (np.ones((input_dim, input_dim, 3), dtype=np.float32)),
])
def test_tf_no_batch(arr_in: np.ndarray):
    flow = Flow().add(uses=ImageTFEncoder)
    with flow:
        results = flow.post(
            on='/test',
            inputs=DocumentArray([Document(blob=arr_in)]),
            return_results=True
        )

        assert len(results[0].docs) == 1
        assert results[0].docs[0].embedding.shape == (target_output_dim,)


def test_tf_batch():
    flow = Flow().add(uses=ImageTFEncoder)

    with flow:
        results = flow.post(
            on='/test',
            inputs=(Document(blob=np.ones((input_dim, input_dim, 3), dtype=np.float32)) for _ in range(25)),
            return_results=True
        )
        assert len(results[0].docs.get_attributes('embedding')) == 25
        assert results[0].docs.get_attributes('embedding')[0].shape == (target_output_dim,)


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'traversal_paths'],
    [
        (pytest.lazy_fixture('docs_with_blobs'), [[['r'], 10], [['c'], 0], [['cc'], 0]], ['r']),
        (pytest.lazy_fixture('docs_with_chunk_blobs'), [[['r'], 0], [['c'], 10], [['cc'], 0]], ['c']),
        (pytest.lazy_fixture('docs_with_chunk_chunk_blobs'), [[['r'], 0], [['c'], 0], [['cc'], 10]], ['cc'])
    ]
)
def test_traversal_path(docs: DocumentArray, docs_per_path: List[List[str]], traversal_paths: List[str]):
    flow = Flow().add(uses=ImageTFEncoder)
    with flow:
        results = flow.post(
            on='/test',
            inputs=docs,
            parameters={'traversal_paths': traversal_paths},
            return_results=True
        )
        for path, count in docs_per_path:
            assert len(DocumentArray(results[0].docs).traverse_flat(path).get_attributes('embedding')) == count

