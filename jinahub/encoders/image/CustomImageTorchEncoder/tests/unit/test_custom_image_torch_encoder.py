from pathlib import Path

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor

from ...custom_image_torch_encoder import CustomImageTorchEncoder


@pytest.fixture
def encoder():
    model_dir = Path(__file__).parents[1] / 'model'
    return CustomImageTorchEncoder(
        model_definition_file=str(model_dir / 'external_model.py'),
        model_state_dict_path=str(model_dir / 'model_state_dict.pth'),
        layer_name='conv1',
        model_class_name='ExternalModel',
    )


def test_config():
    with pytest.raises(ValueError, match='model_state_dict_path option requires'):
        ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))


def test_encoder(encoder):
    output_dim = 10
    input_dim = 224
    test_img = np.random.rand(3, input_dim, input_dim)
    docs = DocumentArray([Document(blob=test_img), Document(blob=test_img)])
    encoder.encode(docs, {})
    assert len(docs) == 2
    for doc in docs:
        assert doc.embedding.shape == (output_dim,)


def test_encoder_traversal_paths(encoder):
    output_dim = 10
    input_dim = 224
    test_img = np.random.rand(3, input_dim, input_dim)
    docs = DocumentArray(
        [
            Document(chunks=[Document(blob=test_img), Document(blob=test_img)]),
            Document(chunks=[Document(blob=test_img), Document(blob=test_img)]),
        ]
    )
    encoder.encode(docs, {'traversal_paths': ['c']})
    assert len(docs) == 2
    assert len(docs.traverse_flat(['c'])) == 4
    for chunk in docs.traverse_flat(['c']):
        assert chunk.embedding.shape == (output_dim,)
