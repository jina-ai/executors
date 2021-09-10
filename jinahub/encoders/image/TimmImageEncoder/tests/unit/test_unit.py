from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
from jina import Document, DocumentArray, Executor

from ...timm_encoder import TimmImageEncoder


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / "config.yml"))
    assert ex.default_batch_size == 32


@pytest.mark.parametrize(
    ["content", "out_shape"],
    [
        ([np.ones((10, 10, 3), dtype=np.uint8), (3, 224, 224)]),
        ([np.ones((360, 420, 3), dtype=np.uint8), (3, 224, 224)]),
        ([np.ones((300, 300, 3), dtype=np.uint8), (3, 224, 224)]),
    ],
)
def test_preprocessing_reshape_correct(content: np.ndarray, out_shape: Tuple):
    encoder = TimmImageEncoder()

    reshaped_content = encoder._preprocess(content)

    assert (
        reshaped_content.shape == out_shape
    ), f"Expected shape {out_shape} but got {reshaped_content.shape}"


@pytest.mark.parametrize(
    "traversal_paths, docs",
    [
        (("r",), pytest.lazy_fixture("docs_with_blobs")),
        (("c",), pytest.lazy_fixture("docs_with_chunk_blobs")),
    ],
)
def test_encode_image_returns_correct_length(
    traversal_paths: Tuple[str], docs: DocumentArray
) -> None:
    encoder = TimmImageEncoder(default_traversal_path=traversal_paths)

    encoder.encode(docs=docs, parameters={})

    for doc in docs.traverse_flat(traversal_paths):
        assert doc.embedding.shape == (512,)


def test_embeddings_quality(test_images: Dict[str, np.array]):
    encoder = TimmImageEncoder()
    doc_list = []

    for name, image_arr in test_images.items():
        print(name)
        doc_list.append(Document(id=name, blob=image_arr))

    docs = DocumentArray(doc_list)
    encoder.encode(docs, parameters={})

    docs.match(docs)
    matches = ["studio", "banana2", "banana1", "airplane"]

    for i, doc in enumerate(docs):
        print(doc.matches[1].id)

    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]


def test_no_preprocessing():
    encoder = TimmImageEncoder(use_default_preprocessing=False)

    # without pre-processing the user needs to provide the
    # right shape for the model directly
    arr_in = np.ones((3, 224, 224), dtype=np.float32)
    docs = DocumentArray([Document(blob=arr_in)])

    encoder.encode(docs=docs, parameters={})

    assert docs[0].embedding.shape == (512,)


def test_docs_array_with_text():
    docs = DocumentArray([Document(text="hello world")])
    encoder = TimmImageEncoder()

    encoder.encode(docs, parameters={})

    assert docs[0].embedding is None


@pytest.mark.parametrize(
    ["model_name", "out_shape"],
    [
        ("resnet50", (2048,)),
        ("mobilenetv3_large_100", (1280,)),
        ("efficientnet_b1", (1280,)),
    ],
)
def test_available_models(model_name: str, out_shape: Tuple):
    encoder = TimmImageEncoder(model_name=model_name)

    # without pre-processing the user needs to provide the
    # right shape for the model directly
    arr_in = np.ones((224, 224, 3), dtype=np.uint8)
    docs = DocumentArray([Document(blob=arr_in)])

    encoder.encode(docs=docs, parameters={})

    assert docs[0].embedding.shape == out_shape


def test_empty_doc_array():
    docs = DocumentArray()
    encoder = TimmImageEncoder()

    encoder.encode(docs, parameters={})

    assert len(docs) == 0


def test_none_docs():
    encoder = TimmImageEncoder()
    encoder.encode(docs=None, parameters={})
