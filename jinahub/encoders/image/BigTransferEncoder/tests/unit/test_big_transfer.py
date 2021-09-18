import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from big_transfer import BigTransferEncoder
from jina import Document, DocumentArray, Executor
from PIL import Image

directory = os.path.dirname(os.path.realpath(__file__))


_INPUT_DIM = 512
_EMBEDDING_DIM = 2048


@pytest.fixture(scope="function")
def encoder() -> BigTransferEncoder:
    yield BigTransferEncoder()
    shutil.rmtree('pretrained', ignore_errors=True)


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
    assert ex.model_path == 'pretrained'
    assert ex.model_name == 'Imagenet21k/R50x1'


def test_no_documents(encoder: BigTransferEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: BigTransferEncoder):
    encoder.encode(docs=None, parameters={})


def test_docs_no_blobs(encoder: BigTransferEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_single_image(encoder: BigTransferEncoder):
    docs = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)
    assert docs[0].embedding.dtype == np.float32


def test_encoding_cpu(encoder: BigTransferEncoder):
    input_data = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
def test_encoding_gpu():
    encoder = BigTransferEncoder(device='/GPU:0')
    input_data = DocumentArray(
        [Document(blob=np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8))]
    )

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    'img_shape',
    [224, 512],
)
def test_encode_any_image_shape(img_shape: int, encoder: BigTransferEncoder):
    docs = DocumentArray(
        [Document(blob=np.ones((img_shape, img_shape, 3), dtype=np.uint8))]
    )

    encoder.encode(docs=docs, parameters={})
    assert len(docs.get_attributes('embedding')) == 1


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(encoder: BigTransferEncoder, batch_size: int):
    blob = np.ones((_INPUT_DIM, _INPUT_DIM, 3), dtype=np.uint8)
    docs = DocumentArray([Document(blob=blob) for _ in range(32)])
    encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    "traversal_paths, counts",
    [
        [('c',), (('r', 0), ('c', 3), ('cc', 0))],
        [('cc',), (("r", 0), ('c', 0), ('cc', 2))],
        [('r',), (('r', 1), ('c', 0), ('cc', 0))],
        [('cc', 'r'), (('r', 1), ('c', 0), ('cc', 2))],
    ],
)
def test_traversal_path(
    traversal_paths: Tuple[str],
    counts: Tuple[str, int],
    nested_docs: DocumentArray,
    encoder: BigTransferEncoder,
):
    encoder.encode(nested_docs, parameters={"traversal_paths": traversal_paths})
    for path, count in counts:
        embeddings = nested_docs.traverse_flat([path]).get_attributes('embedding')
        assert len([em for em in embeddings if em is not None]) == count
