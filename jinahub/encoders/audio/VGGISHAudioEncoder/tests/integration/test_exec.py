__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess

import librosa
import pytest
from executor.vggish import vggish_input
from jina import Document, DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_flow_from_yml():

    doc = DocumentArray([Document()])
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        resp = f.post(on='test', inputs=doc, return_results=True)

    assert resp is not None


def test_embedding_exists():

    x_audio, sample_rate = librosa.load(
        os.path.join(cur_dir, '../test_data/sample.wav')
    )
    log_mel_examples = vggish_input.waveform_to_examples(x_audio, sample_rate)
    doc = DocumentArray([Document(blob=log_mel_examples)])

    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        responses = f.post(on='index', inputs=doc, return_results=True)

    assert responses[0].docs[0].embedding is not None


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
