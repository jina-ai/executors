__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import subprocess

import pytest
from executor.audio_loader import AudioLoader
from jina import Document, DocumentArray, Flow


def test_integration():
    docs = DocumentArray([Document(uri='tests/test_data/example_mp3.mp3')])
    with Flow().add(uses=AudioLoader) as flow:
        resp = flow.post(
            on="/index",
            inputs=docs,
            return_results=True,
        )

    for r in resp:
        assert len(r.docs) == 1
        for doc in r.docs:
            assert doc.blob is not None


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ['jina', 'executor', f'--uses=docker://{build_docker_image}'],
            timeout=30,
            check=True,
        )
