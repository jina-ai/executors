__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess

import pytest
from jina import Document, DocumentArray, Flow
from spacy_text_encoder import SpacyTextEncoder

_EMBEDDING_DIM = 96


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_integration(request_size: int):
    docs = DocumentArray(
        [Document(text='just some random text here') for _ in range(50)]
    )
    with Flow(return_results=True).add(uses=SpacyTextEncoder) as flow:
        resp = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size,
        )

    assert len(resp) == 50
    for doc in resp:
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
                'device:"/GPU:0"',
            ],
            timeout=30,
            check=True,
        )


def test_spacy_text_encoder():
    docs = DocumentArray(
        [
            Document(text='Han likes eating pizza'),
            Document(text='Han likes pizza'),
            Document(text='Jina rocks'),
        ]
    )
    f = Flow().add(uses=SpacyTextEncoder)
    with f:
        docs = f.post(on='/test', inputs=docs)
        assert len(docs) == 3
        for doc in docs:
            assert doc.embedding.shape == (96,)
