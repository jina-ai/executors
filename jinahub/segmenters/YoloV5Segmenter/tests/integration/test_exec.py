__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess

import pytest
from jina import Flow

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
