__copyright__ = 'Copyright (c) 2020-2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import os
import subprocess

import librosa
import pytest
from executor.audio_clip_encoder import AudioCLIPEncoder
from jina import Document, DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_flow_from_yml():
    doc = DocumentArray([Document()])
    with Flow(return_results=True).add(uses=AudioCLIPEncoder) as f:
        resp = f.post(on='/test', inputs=doc, return_results=True)
        assert resp is not None


def test_embedding_exists():
    x_audio, sr = librosa.load(os.path.join(cur_dir, '../test_data/sample.mp3'))
    doc = DocumentArray([Document(tensor=x_audio, tags={'sample_rate': sr})])

    with Flow().add(uses=AudioCLIPEncoder) as f:
        responses = f.post(on='index', inputs=doc, return_results=True)
        assert responses[0].docs[0].embedding is not None
        assert responses[0].docs[0].embedding.shape == (1024,)


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
                'download_model:True',
            ],
            timeout=30,
            check=True,
        )
