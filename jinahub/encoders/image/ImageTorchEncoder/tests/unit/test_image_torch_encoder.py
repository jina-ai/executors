__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path
from typing import Tuple, Dict

import pytest

import numpy as np
from jina import DocumentArray, Document, Executor

from ...torch_encoder import ImageTorchEncoder


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    ex.default_batch_size == 32


@pytest.mark.parametrize(
    ['content', 'out_shape'],
    [
        ([np.ones((10, 10, 3), dtype=np.uint8), (3, 224, 224)]),
        ([np.ones((360, 420, 3), dtype=np.uint8), (3, 224, 224)]),
        ([np.ones((300, 300, 3), dtype=np.uint8), (3, 224, 224)]),
    ],
)
def test_preprocessing_reshape_correct(content: np.ndarray, out_shape: Tuple):
    encoder = ImageTorchEncoder()

    reshaped_content = encoder._preprocess(content)

    assert (
        reshaped_content.shape == out_shape
    ), f'Expected shape {out_shape} but got {reshaped_content.shape}'


@pytest.mark.parametrize(
    'traversal_paths, docs',
    [
        (('r',), pytest.lazy_fixture('docs_with_blobs')),
        (('c',), pytest.lazy_fixture('docs_with_chunk_blobs')),
    ],
)
def test_encode_image_returns_correct_length(
    traversal_paths: Tuple[str], docs: DocumentArray
) -> None:
    encoder = ImageTorchEncoder(default_traversal_path=traversal_paths)

    encoder.encode(docs=docs, parameters={})

    for doc in docs.traverse_flat(traversal_paths):
        assert doc.embedding is not None
        assert doc.embedding.shape == (512,)


@pytest.mark.parametrize('model_name', ['resnet50', 'mobilenet_v3_large', 'googlenet'])
def test_encodes_semantic_meaning(test_images: Dict[str, np.array], model_name: str):
    encoder = ImageTorchEncoder(model_name=model_name)
    embeddings = {}

    for name, image_arr in test_images.items():
        docs = DocumentArray([Document(blob=image_arr)])
        encoder.encode(docs, parameters={})
        embeddings[name] = docs[0].embedding

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
    assert small_distance < dist('studio', 'satellite')


def test_no_preprocessing():
    encoder = ImageTorchEncoder(use_default_preprocessing=False)

    # without pre-processing the user needs to provide the right shape for the model directly
    arr_in = np.ones((3, 224, 224), dtype=np.float32)
    docs = DocumentArray([Document(blob=arr_in)])

    encoder.encode(docs=docs, parameters={})

    assert docs[0].embedding.shape == (512,)


def test_empty_doc_array():
    docs = DocumentArray()
    encoder = ImageTorchEncoder()

    encoder.encode(docs, parameters={})

    assert len(docs) == 0


def test_docs_array_with_no_text():
    docs = DocumentArray([Document(text='hello world')])
    encoder = ImageTorchEncoder()

    encoder.encode(docs, parameters={})

    assert docs[0].embedding is None
