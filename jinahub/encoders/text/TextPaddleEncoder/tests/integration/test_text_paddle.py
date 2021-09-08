import subprocess

import pytest
from jina import Document, DocumentArray, Flow

from ...text_paddle import TextPaddleEncoder


@pytest.fixture(scope='function')
def flow():
    return Flow().add(uses=TextPaddleEncoder)


@pytest.fixture(scope='function')
def content():
    return 'hello world'


@pytest.fixture(scope='function')
def document_array(content):
    return DocumentArray([Document(content=content)])


def validate_callback(mock, validate_func):
    for args, kwargs in mock.call_args_list:
        validate_func(*args, **kwargs)
    mock.assert_called()


@pytest.mark.parametrize(
    'parameters',
    [
        {'traverse_paths': ['r'], 'batch_size': 10},
        {'traverse_paths': ['m'], 'batch_size': 10},
        {'traverse_paths': ['r', 'c'], 'batch_size': 5},
    ],
)
def test_text_paddle(flow, content, document_array, parameters, mocker):
    def validate(resp):
        for doc in resp.docs:
            assert doc.embedding.shape == (1024,)
            assert doc.embedding.all()

    mock_on_done = mocker.Mock()
    with flow as f:
        f.index(inputs=document_array, on_done=mock_on_done)
    validate_callback(mock_on_done, validate)


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
                'use_gpu:True',
            ],
            timeout=30,
            check=True,
        )
