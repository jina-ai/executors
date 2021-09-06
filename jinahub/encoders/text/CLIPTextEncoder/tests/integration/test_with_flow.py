import subprocess

import pytest
from jina import Document, DocumentArray, Flow

from ...clip_text import CLIPTextEncoder

_EMBEDDING_DIM = 512


@pytest.mark.parametrize('request_size', [1, 10, 50, 100])
def test_integration(request_size: int):
    docs = DocumentArray(
        [Document(text='just some random text here') for _ in range(50)]
    )
    with Flow(return_results=True).add(uses=CLIPTextEncoder) as flow:
        resp = flow.post(
            on='/index',
            inputs=docs,
            request_size=request_size,
            return_results=True,
        )

    assert sum(len(resp_batch.docs) for resp_batch in resp) == 50
    for r in resp:
        for doc in r.docs:
            assert doc.embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                'jina',
                'executor',
                f'--uses=docker://{build_docker_image}',
                '--volumes=.cache:/workspace/.cache',
            ],
            timeout=30,
            check=True,
        )
