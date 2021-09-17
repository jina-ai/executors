__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path

import numpy as np
import pytest
from image_tf_encoder import ImageTFEncoder
from PIL import Image
from jina import Document, DocumentArray, Executor

_INPUT_DIM = 336
_EMBEDDING_DIM = 1280


@pytest.fixture(scope="module")
def encoder() -> ImageTFEncoder:
    return ImageTFEncoder()


@pytest.fixture(scope="function")
def nested_docs() -> DocumentArray:
    blob = np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8)
    docs = DocumentArray([Document(id="root1", blob=blob)])
    docs[0].chunks = [
        Document(id="chunk11", blob=blob),
        Document(id="chunk12", blob=blob),
        Document(id="chunk13", blob=blob),
    ]
    docs[0].chunks[0].chunks = [
        Document(id="chunk111", blob=blob),
        Document(id="chunk112", blob=blob),
    ]

    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.model_name == 'MobileNetV2'


def test_no_documents(encoder: ImageTFEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: ImageTFEncoder):
    encoder.encode(docs=None, parameters={})


def test_docs_no_blobs(encoder: ImageTFEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_single_image(encoder: ImageTFEncoder):
    docs = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)
    assert docs[0].embedding.dtype == np.float32


def test_encoding_cpu():
    encoder = ImageTFEncoder(device='cpu')
    input_data = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
def test_encoding_gpu():
    encoder = ImageTFEncoder(device='/GPU:0')
    input_data = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    'img_shape',
    [224, 512],
)
def test_encode_any_image_shape(img_shape):
    encoder = ImageTFEncoder(img_shape=img_shape)
    docs = DocumentArray(
        [Document(blob=np.ones((img_shape, img_shape, 3), dtype=np.uint8))]
    )

    encoder.encode(docs=docs, parameters={})
    assert len(docs.get_attributes('embedding')) == 1


def test_encode_any_image_shape_mismatch():
    encoder = ImageTFEncoder(img_shape=224)
    docs = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )

    with pytest.raises(ValueError):
        encoder.encode(docs=docs, parameters={})


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(encoder: ImageTFEncoder, batch_size: int):
    blob = np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8)
    docs = DocumentArray([Document(blob=blob) for _ in range(32)])
    encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)
