import pytest
import os

import numpy as np

from jina import Document, DocumentArray

try:
    from custom_image_torch_encoder import CustomImageTorchEncoder
except:
    from jinahub.encoder.custom_image_torch_encoder import CustomImageTorchEncoder

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def encoder(tmpdir):
    model_state_dict_path = os.path.join(cur_dir, '../model/model_state_dict.pth')
    return CustomImageTorchEncoder(model_definition_file=os.path.join(cur_dir, '../model/external_model.py'),
                                   model_state_dict_path=model_state_dict_path, layer_name='conv1',
                                   model_class_name='ExternalModel')


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
    docs = DocumentArray([Document(chunks=[Document(blob=test_img), Document(blob=test_img)]),
                          Document(chunks=[Document(blob=test_img), Document(blob=test_img)])])
    encoder.encode(docs, {'traversal_paths': ['c']})
    assert len(docs) == 2
    assert len(docs.traverse_flat(['c'])) == 4
    for chunk in docs.traverse_flat(['c']):
        assert chunk.embedding.shape == (output_dim,)
