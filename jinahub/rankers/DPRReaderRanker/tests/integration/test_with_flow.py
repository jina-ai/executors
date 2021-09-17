__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess

import pytest
from dpr_reader import DPRReaderRanker
from jina import Document, DocumentArray, Flow


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
                '--gpus',
                'all',
                '--uses-with',
                'device:cuda',
            ],
            timeout=30,
            check=True,
        )
