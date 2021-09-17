import os
import shutil
from pathlib import Path

import numpy as np
import PIL.Image as Image
import pytest
from big_transfer import BigTransferEncoder
from jina import Document, DocumentArray, Executor

directory = os.path.dirname(os.path.realpath(__file__))


_INPUT_DIM = 512
_EMBEDDING_DIM = 2048


@pytest.fixture(scope="module")
def encoder() -> BigTransferEncoder:
    return BigTransferEncoder()


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
