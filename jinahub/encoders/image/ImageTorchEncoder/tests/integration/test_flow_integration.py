__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from typing import List

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow

from ...torch_encoder import ImageTorchEncoder


@pytest.mark.parametrize(
    'arr_in',
    [
        (np.ones((224, 224, 3), dtype=np.uint8)),
        (np.ones((100, 100, 3), dtype=np.uint8)),
        (np.ones((50, 40, 3), dtype=np.uint8)),
    ],
)
def test_no_batch(arr_in: np.ndarray):
    flow = Flow().add(uses=ImageTorchEncoder)
    with flow:
        resp = flow.post(
            on='/test', inputs=[Document(blob=arr_in)], return_results=True
        )

    results_arr = DocumentArray(resp[0].data.docs)
    assert len(results_arr) == 1
    assert results_arr[0].embedding is not None
    assert results_arr[0].embedding.shape == (512,)


def test_with_batch():
    flow = Flow().add(uses=ImageTorchEncoder)

    with flow:
        resp = flow.post(
            on='/test',
            inputs=(
                Document(blob=np.ones((224, 224, 3), dtype=np.uint8)) for _ in range(25)
            ),
            return_results=True,
        )

    assert len(resp[0].docs.get_attributes('embedding')) == 25


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'traversal_paths'],
    [
        (pytest.lazy_fixture('docs_with_blobs'), [['r', 11], ['c', 0], ['cc', 0]], 'r'),
        (
            pytest.lazy_fixture('docs_with_chunk_blobs'),
            [['r', 0], ['c', 11], ['cc', 0]],
            'c',
        ),
        (
            pytest.lazy_fixture('docs_with_chunk_chunk_blobs'),
            [['r', 0], ['c', 0], ['cc', 11]],
            'cc',
        ),
    ],
)
def test_traversal_paths(
    docs: DocumentArray, docs_per_path: List[List[str]], traversal_paths: str
):
    def validate_traversal(expected_docs_per_path: List[List[str]]):
        def validate(res):
            for path, count in expected_docs_per_path:
                embeddings = DocumentArray(res[0].docs).traverse_flat(
                    [path]).get_attributes('embedding')
                return len([em for em in embeddings if em is not None]) == count

        return validate

    flow = Flow().add(uses=ImageTorchEncoder)

    with flow:
        resp = flow.post(
            on='/test',
            inputs=docs,
            parameters={'traversal_paths': [traversal_paths]},
            return_results=True,
        )

    assert validate_traversal(docs_per_path)(resp)


@pytest.mark.gpu
@pytest.mark.docker
def test_docker_runtime_gpu():
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'pea',
                '--uses=docker://imagetorchencoder:gpu',
                '--gpus',
                'all',
                '--uses-with',
                'device:cuda',
            ],
            timeout=30,
            check=True,
        )
