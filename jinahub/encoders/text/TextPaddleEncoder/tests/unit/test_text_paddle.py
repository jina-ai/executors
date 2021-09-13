from pathlib import Path

import pytest
import numpy as np
import paddlehub as hub
from jina import Document, DocumentArray, Executor

from ...text_paddle import TextPaddleEncoder


@pytest.fixture(scope='function')
def model():
    return hub.Module(name='ernie_tiny')


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.default_batch_size == 32


def test_text_paddle(model, document_array, content, parameters):
    ex = TextPaddleEncoder()
    assert ex.on_gpu is False
    ex.encode(document_array, parameters)
    for doc in document_array:
        assert isinstance(doc.embedding, np.ndarray)
        assert doc.embedding.shape == (1024,)
    embeds = model.get_embedding([[content]])
    pooled_features = []
    for embed in embeds:
        pooled_feature, _ = embed
        pooled_features.append(pooled_feature)
    assert (pooled_features == document_array[0].embedding).all()
