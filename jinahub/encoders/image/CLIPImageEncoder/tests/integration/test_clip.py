import subprocess

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow

from ...clip_image import CLIPImageEncoder


@pytest.mark.parametrize("request_size", [1, 10, 50, 100])
def test_integration(request_size: int):
    docs = DocumentArray(
        [
            Document(blob=np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            for _ in range(50)
        ]
    )
    with Flow(return_results=True).add(uses=CLIPImageEncoder) as flow:
        resp = flow.post(
            on="/index",
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 50
    for r in resp:
        for doc in r.docs:
            assert doc.embedding is not None
            assert doc.embedding.shape == (512,)


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
