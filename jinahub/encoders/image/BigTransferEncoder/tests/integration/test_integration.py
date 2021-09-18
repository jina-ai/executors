import os
import shutil
import subprocess

import numpy as np
import PIL.Image as Image
import pytest
from jina import Document, Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


def data_generator(num_docs):
    for i in range(num_docs):
        doc = Document(uri=os.path.join(cur_dir, '..', 'imgs', 'cat.jpg'))
        doc.convert_image_uri_to_blob()
        img = Image.fromarray(doc.blob.astype('uint8'))
        img = img.resize((96, 96))
        img = np.array(img).astype('float32') / 255
        doc.blob = img
        yield doc


@pytest.mark.parametrize(
    'model_name', ['R50x1', 'R101x1', 'R50x3', 'R101x3']  # , 'R152x4']
)
@pytest.mark.parametrize('dataset', ['Imagenet1k', 'Imagenet21k'])
def test_all_models(model_name: str, dataset: str):
    shutil.rmtree('pretrained', ignore_errors=True)
    os.environ['TRANSFER_MODEL_NAME'] = f'{dataset}/{model_name}'
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as flow:
        data = flow.post(
            on='/index',
            inputs=data_generator(100),
            request_size=10,
            return_results=True,
        )
        docs = data[0].docs
        for doc in docs:
            assert doc.embedding is not None


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
