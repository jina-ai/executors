__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import numpy as np
from PIL import Image
from jina import Document, DocumentArray, Executor

from image_tf_encoder import ImageTFEncoder

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


@pytest.fixture(scope='function')
def test_images(test_dir: str) -> Dict[str, np.ndarray]:
    def get_path(file_name_no_suffix: str) -> str:
        return os.path.join(test_dir, 'test_data', file_name_no_suffix + '.png')

    return {
        file_name: np.array(Image.open(get_path(file_name)), dtype=np.float32)[
            :, :, 0:3
        ]
        / 255
        for file_name in ['airplane', 'banana1', 'banana2', 'satellite', 'studio']
    }


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


def test_image_results(test_images: Dict[str, np.array]):
    embeddings = {}
    encoder = ImageTFEncoder()
    for name, image_arr in test_images.items():
        docs = DocumentArray([Document(blob=image_arr)])
        encoder.encode(docs, parameters={})
        embeddings[name] = docs[0].embedding
        assert docs[0].embedding.shape == (_EMBEDDING_DIM,)

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satellite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
    assert small_distance < dist('banana2', 'satellite')
    assert small_distance < dist('banana2', 'studio')
    assert small_distance < dist('airplane', 'studio')
    assert small_distance < dist('airplane', 'satellite')


@pytest.mark.gpu
def test_image_results_gpu(test_images: Dict[str, np.array]):
    num_doc = 2
    test_data = np.random.rand(num_doc, _INPUT_DIM, _INPUT_DIM, 3)
    doc = DocumentArray()
    for i in range(num_doc):
        doc.append(Document(blob=test_data[i]))

    encoder = ImageTFEncoder(device='/GPU:0')
    encoder.encode(doc, parameters={})
    assert len(doc) == num_doc
    for i in range(num_doc):
        assert doc[i].embedding.shape == (_EMBEDDING_DIM,)


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
    encoder: ImageTFEncoder,
):
    encoder.encode(nested_docs, parameters={"traversal_paths": traversal_paths})
    for path, count in counts:
        embeddings = nested_docs.traverse_flat([path]).get_attributes('embedding')
        assert len([em for em in embeddings if em is not None]) == count
