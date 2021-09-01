__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
from typing import List

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow

from ...paddle_image import ImagePaddlehubEncoder


@pytest.mark.parametrize(
    'arr_in',
    [
        (np.ones((3, 224, 224), dtype=np.float32)),
        (np.ones((3, 100, 100), dtype=np.float32)),
        (np.ones((3, 50, 40), dtype=np.float32)),
    ],
)
def test_paddle_no_batch(arr_in: np.ndarray):
    flow = Flow().add(uses=ImagePaddlehubEncoder)
    with flow:
        results = flow.post(
            on='/test',
            inputs=DocumentArray([Document(blob=arr_in)]),
            return_results=True,
        )

        assert len(results[0].docs) == 1
        assert results[0].docs[0].embedding.shape == (2048,)


def test_paddle_batch():
    flow = Flow().add(uses=ImagePaddlehubEncoder)

    with flow:
        results = flow.post(
            on='/test',
            inputs=(
                Document(blob=np.ones((3, 224, 224), dtype=np.float32))
                for _ in range(25)
            ),
            return_results=True,
        )
        assert len(results[0].docs.get_attributes('embedding')) == 25
        assert results[0].docs.get_attributes('embedding')[0].shape == (2048,)


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'traversal_paths'],
    [
        (pytest.lazy_fixture('docs_with_blobs'), [['r', 10], ['c', 0], ['cc', 0]], 'r'),
        (
            pytest.lazy_fixture('docs_with_chunk_blobs'),
            [['r', 0], ['c', 10], ['cc', 0]],
            'c',
        ),
        (
            pytest.lazy_fixture('docs_with_chunk_chunk_blobs'),
            [['r', 0], ['c', 0], ['cc', 10]],
            'cc',
        ),
    ],
)
def test_traversal_path(
    docs: DocumentArray, docs_per_path: List[List[str]], traversal_paths: str
):
    flow = Flow().add(uses=ImagePaddlehubEncoder)
    with flow:
        results = flow.post(
            on='/test',
            inputs=docs,
            parameters={'traversal_paths': [traversal_paths]},
            return_results=True,
        )
        for path, count in docs_per_path:
            embeddings = (
                DocumentArray(results[0].docs)
                .traverse_flat([path])
                .get_attributes('embedding')
            )
            assert len([em for em in embeddings if em is not None]) == count


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ['jina', 'executor', f'--uses=docker://{build_docker_image}'],
            timeout=30,
            check=True,
        )


@pytest.mark.gpu
@pytest.mark.docker
def test_docker_runtime_gpu(build_docker_image_gpu: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'pea',
                f'--uses=docker://{build_docker_image_gpu}',
                '--gpus all',
                '--uses-with',
                'device:cuda',
            ],
            timeout=30,
            check=True,
        )
