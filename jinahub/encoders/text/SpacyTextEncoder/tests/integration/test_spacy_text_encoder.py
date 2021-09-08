__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess
import pytest

from jina import Document, Flow, DocumentArray

try:
    from spacy_text_encoder import SpacyTextEncoder
except:
    from ...spacy_text_encoder import SpacyTextEncoder


def test_spacy_text_encoder():
    docs = DocumentArray([Document(text='Han likes eating pizza'), Document(text='Han likes pizza'),
                          Document(text='Jina rocks')])
    f = Flow().add(uses=SpacyTextEncoder)
    with f:
        resp = f.post(on='/test', inputs=docs, return_results=True)
        docs = resp[0].docs
        assert len(docs) == 3
        for doc in docs:
            assert doc.embedding.shape == (96,)


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
                'device:"/GPU:0"',
            ],
            timeout=30,
            check=True,
        )
