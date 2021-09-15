__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import subprocess

import numpy as np
import pytest
from jina import Document, DocumentArray, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_video_torch_encoder():
    model_state_dict_path = os.path.join(cur_dir, '../model/model_state_dict.pth')
    input_dim = 224
    test_img = np.random.rand(3, input_dim, input_dim)
    docs = DocumentArray([Document(blob=test_img), Document(blob=test_img)])
    f = Flow().add(
        uses={
            'jtype': 'CustomImageTorchEncoder',
            'with': {
                'model_state_dict_path': model_state_dict_path,
                'layer_name': 'conv1',
                'model_definition_file': os.path.join(
                    cur_dir, '../model/external_model.py'
                ),
                'model_class_name': 'ExternalModel',
            },
        }
    )
    with f:
        resp = f.post(on='/test', inputs=docs, return_results=True)
        assert resp[0].docs[0].embedding.shape == (10,)
        assert resp[0].docs[1].embedding.shape == (10,)


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
            ],
            timeout=30,
            check=True,
        )
