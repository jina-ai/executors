__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess

import pytest
from jina import Document, DocumentArray, Flow
from sentence_encoder import TransformerSentenceEncoder

_EMBEDDING_DIM = 384


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_integration(request_size: int):
    docs = DocumentArray(
        [Document(text='just some random text here') for _ in range(50)]
    )
    with Flow(return_results=True).add(uses=TransformerSentenceEncoder) as flow:
        resp_docs = flow.post(on='/index', inputs=docs, request_size=request_size)

    assert len(resp_docs) == 50
    for doc in resp_docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


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
                'executor',
                f'--uses=docker://{build_docker_image_gpu}',
                '--gpus',
                'all',
                '--uses-with',
                'device:cuda',
            ],
            timeout=30,
            check=True,
        )
