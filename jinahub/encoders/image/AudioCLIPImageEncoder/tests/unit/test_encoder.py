__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor
from PIL import Image

from ...audioclip_image import AudioCLIPImageEncoder


@pytest.fixture(scope="module")
def basic_encoder() -> AudioCLIPImageEncoder:
    return AudioCLIPImageEncoder()


@pytest.fixture(scope="module")
def basic_encoder_no_pre() -> AudioCLIPImageEncoder:
    return AudioCLIPImageEncoder(use_default_preprocessing=False)


@pytest.fixture(scope="function")
def nested_docs() -> DocumentArray:
    blob = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(id='root1', blob=blob)])
    docs[0].chunks = [
        Document(id='chunk11', blob=blob),
        Document(id='chunk12', blob=blob),
        Document(id='chunk13', blob=blob),
    ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', blob=blob),
        Document(id='chunk112', blob=blob),
    ]

    return docs


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.default_batch_size == 32
    assert ex.default_traversal_paths == ['r']
    assert ex.use_default_preprocessing == True


def test_no_documents(basic_encoder):
    docs = DocumentArray()
    basic_encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 0


def test_none_docs(basic_encoder):
    basic_encoder.encode(docs=None, parameters={})


def test_docs_no_blobs(basic_encoder):
    docs = DocumentArray([Document()])
    basic_encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_err_preprocessing(basic_encoder):
    docs = DocumentArray([Document(blob=np.ones((3, 100, 100), dtype=np.uint8))])

    with pytest.raises(ValueError, match='If `use_default_preprocessing=True`'):
        basic_encoder.encode(docs, {})


def test_err_no_preprocessing(basic_encoder_no_pre):
    docs = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])

    with pytest.raises(ValueError, match='If `use_default_preprocessing=False`'):
        basic_encoder_no_pre.encode(docs, {})


def test_single_image(basic_encoder):
    docs = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])
    basic_encoder.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32


def test_single_image_no_preprocessing(basic_encoder_no_pre):
    docs = DocumentArray([Document(blob=np.ones((3, 224, 224), dtype=np.uint8))])
    basic_encoder_no_pre.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32


def test_batch_different_size(basic_encoder):
    docs = DocumentArray(
        [
            Document(blob=np.ones((100, 100, 3), dtype=np.uint8)),
            Document(blob=np.ones((200, 100, 3), dtype=np.uint8)),
        ]
    )
    basic_encoder.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


def test_batch_no_preprocessing(basic_encoder_no_pre):
    docs = DocumentArray(
        [
            Document(blob=np.ones((3, 224, 224), dtype=np.uint8)),
            Document(blob=np.ones((3, 224, 224), dtype=np.uint8)),
        ]
    )
    basic_encoder_no_pre.encode(docs, {})

    assert docs[0].embedding.shape == (1024,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


@pytest.mark.parametrize(
    "path, expected_counts",
    [['c', (('r', 0), ('c', 3), ('cc', 0))], ['cc', (('r', 0), ('c', 0), ('cc', 2))]],
)
def test_traversal_path(
    path: str,
    expected_counts: Tuple[str, int],
    nested_docs: DocumentArray,
    basic_encoder: AudioCLIPImageEncoder,
):
    basic_encoder.encode(nested_docs, parameters={'traversal_paths': [path]})
    for path_check, count in expected_counts:
        assert (
            len(nested_docs.traverse_flat([path_check]).get_attributes('embedding'))
            == count
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_size(basic_encoder: AudioCLIPImageEncoder, batch_size: int):
    blob = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(blob=blob) for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (1024,)


def test_embeddings_quality(basic_encoder: AudioCLIPImageEncoder):
    """
    This tests that the embeddings actually "make sense".
    We check this by making sure that the distance between the embeddings
    of two similar images is smaller than everything else.
    """

    data_dir = Path(__file__).parent.parent / 'imgs'
    dog = Document(id='dog', blob=np.array(Image.open(data_dir / 'dog.jpg')))
    cat = Document(id='cat', blob=np.array(Image.open(data_dir / 'cat.jpg')))
    airplane = Document(
        id='airplane', blob=np.array(Image.open(data_dir / 'airplane.jpg'))
    )
    helicopter = Document(
        id='helicopter', blob=np.array(Image.open(data_dir / 'helicopter.jpg'))
    )

    docs = DocumentArray([dog, cat, airplane, helicopter])
    basic_encoder.encode(docs, {})

    docs.match(docs)
    matches = ["cat", "dog", "helicopter", "airplane"]
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]
