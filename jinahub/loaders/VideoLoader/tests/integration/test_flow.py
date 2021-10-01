__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import subprocess

import pytest
from executor import VideoLoader
from jina import Document, DocumentArray, Flow


def test_integration(tmp_path):
    da = DocumentArray(
        [Document(id='2c2OmN49cj8.mp4', uri='tests/toy_data/2c2OmN49cj8.mp4')]
    )
    workspace = str(tmp_path / 'workspace')
    with Flow().add(uses=VideoLoader, uses_metas={'workspace': workspace}) as flow:
        resp = flow.post(on='/index', inputs=da, return_results=True)

    assert len(resp[0].docs) == 1
    for doc in resp[0].docs:
        assert len(doc.chunks) == 16
        for image_chunk in filter(lambda x: x.modality == 'image', doc.chunks):
            assert len(image_chunk.blob.shape) == 3

        for audio_chunk in filter(lambda x: x.modality == 'audio', doc.chunks):
            assert audio_chunk.blob is not None


@pytest.mark.docker
def test_docker_runtime(build_docker_image: str):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            ['jina', 'executor', f'--uses=docker://{build_docker_image}'],
            timeout=30,
            check=True,
        )
