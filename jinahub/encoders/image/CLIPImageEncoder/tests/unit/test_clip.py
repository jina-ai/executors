from pathlib import Path
from typing import Tuple

import clip
import numpy as np
import pytest
import torch
from jina import Document, DocumentArray, Executor
from PIL import Image

from ...clip_image import CLIPImageEncoder

_EMBEDDING_DIM = 512


@pytest.fixture(scope="module")
def encoder() -> CLIPImageEncoder:
    return CLIPImageEncoder()


@pytest.fixture(scope="module")
def encoder_no_pre() -> CLIPImageEncoder:
    return CLIPImageEncoder(use_default_preprocessing=False)


@pytest.fixture(scope="function")
def nested_docs() -> DocumentArray:
    blob = np.ones((224, 224, 3), dtype=np.uint8)
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
    assert ex.default_batch_size == 32
    assert len(ex.default_traversal_paths) == 1
    assert ex.default_traversal_paths[0] == "r"
    assert ex.device == "cpu"
    assert ex.is_updated is False


def test_no_documents(encoder: CLIPImageEncoder):
    docs = DocumentArray()
    encoder.encode(docs=docs, parameters={})
    assert len(docs) == 0  # SUCCESS


def test_none_docs(encoder: CLIPImageEncoder):
    encoder.encode(docs=None, parameters={})


def test_docs_no_blobs(encoder: CLIPImageEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_single_image(encoder: CLIPImageEncoder):
    docs = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)
    assert docs[0].embedding.dtype == np.float32


def test_single_image_no_preprocessing(encoder_no_pre: CLIPImageEncoder):
    docs = DocumentArray([Document(blob=np.ones((3, 224, 224), dtype=np.uint8))])
    encoder_no_pre.encode(docs, {})

    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)
    assert docs[0].embedding.dtype == np.float32


def test_encoding_cpu():
    encoder = CLIPImageEncoder(device="cpu")
    input_data = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


def test_cpu_no_preprocessing():
    encoder = CLIPImageEncoder(device="cpu", use_default_preprocessing=False)
    input_data = DocumentArray([Document(blob=np.ones((3, 224, 224), dtype=np.uint8))])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
def test_encoding_gpu():
    encoder = CLIPImageEncoder(device="cuda")
    input_data = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
def test_gpu_no_preprocessing():
    encoder = CLIPImageEncoder(device="cuda", use_default_preprocessing=False)
    input_data = DocumentArray(
        [Document(blob=np.ones((3, 224, 224), dtype=np.float32))]
    )

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


def test_clip_any_image_shape(encoder: CLIPImageEncoder):
    docs = DocumentArray([Document(blob=np.ones((224, 224, 3), dtype=np.uint8))])

    encoder.encode(docs=docs, parameters={})
    assert len(docs.get_attributes("embedding")) == 1

    docs = DocumentArray([Document(blob=np.ones((100, 100, 3), dtype=np.uint8))])
    encoder.encode(docs=docs, parameters={})
    assert len(docs.get_attributes("embedding")) == 1


def test_clip_batch(encoder: CLIPImageEncoder):
    """
    This tests that the encoder can handle inputs of various size
    which is not a factorial of ``default_batch_size``

    """
    docs = DocumentArray(
        [
            Document(blob=np.ones((100, 100, 3), dtype=np.uint8)),
            Document(blob=np.ones((100, 100, 3), dtype=np.uint8)),
        ]
    )
    encoder.encode(docs, parameters={})
    assert len(docs.get_attributes("embedding")) == 2
    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


def test_batch_no_preprocessing(encoder_no_pre: CLIPImageEncoder):
    docs = DocumentArray(
        [
            Document(blob=np.ones((3, 224, 224), dtype=np.float32)),
            Document(blob=np.ones((3, 224, 224), dtype=np.float32)),
        ]
    )
    encoder_no_pre.encode(docs, {})
    assert len(docs.get_attributes("embedding")) == 2
    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)
    assert docs[0].embedding.dtype == np.float32
    np.testing.assert_allclose(docs[0].embedding, docs[1].embedding)


@pytest.mark.parametrize(
    "path, expected_counts",
    [["c", (("r", 0), ("c", 3), ("cc", 0))], ["cc", (("r", 0), ("c", 0), ("cc", 2))]],
)
def test_traversal_path(
    path: str,
    expected_counts: Tuple[str, int],
    nested_docs: DocumentArray,
    encoder: CLIPImageEncoder,
):
    encoder.encode(nested_docs, parameters={"traversal_paths": [path]})
    for path_check, count in expected_counts:
        assert (
            len(nested_docs.traverse_flat([path_check]).get_attributes("embedding"))
            == count
        )


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_size(encoder: CLIPImageEncoder, batch_size: int):
    blob = np.ones((100, 100, 3), dtype=np.uint8)
    docs = DocumentArray([Document(blob=blob) for _ in range(32)])
    encoder.encode(docs, parameters={"batch_size": batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_size_no_preprocessing(encoder_no_pre: CLIPImageEncoder, batch_size: int):
    blob = np.ones((3, 224, 224), dtype=np.uint8)
    docs = DocumentArray([Document(blob=blob) for _ in range(32)])
    encoder_no_pre.encode(docs, parameters={"batch_size": batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


def test_embeddings_quality(encoder: CLIPImageEncoder):
    """
    This tests that the embeddings actually "make sense".
    We check this by making sure that the distance between the embeddings
    of two similar images is smaller than everything else.
    """

    data_dir = Path(__file__).parent.parent / "imgs"
    dog = Document(id="dog", blob=np.array(Image.open(data_dir / "dog.jpg")))
    cat = Document(id="cat", blob=np.array(Image.open(data_dir / "cat.jpg")))
    airplane = Document(
        id="airplane", blob=np.array(Image.open(data_dir / "airplane.jpg"))
    )
    helicopter = Document(
        id="helicopter", blob=np.array(Image.open(data_dir / "helicopter.jpg"))
    )

    docs = DocumentArray([dog, cat, airplane, helicopter])
    encoder.encode(docs, {})

    docs.match(docs)
    matches = ["cat", "dog", "helicopter", "airplane"]
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]


def test_openai_embed_match():
    data_dir = Path(__file__).parent.parent / "imgs"
    dog = Document(id="dog", blob=np.array(Image.open(data_dir / "dog.jpg")))
    airplane = Document(
        id="airplane", blob=np.array(Image.open(data_dir / "airplane.jpg"))
    )
    helicopter = Document(
        id="helicopter", blob=np.array(Image.open(data_dir / "helicopter.jpg"))
    )

    docs = DocumentArray([dog, airplane, helicopter])

    clip_text_encoder = CLIPImageEncoder("openai/clip-vit-base-patch32", device="cpu")
    clip_text_encoder.encode(docs, {})

    actual_embedding = np.stack(docs.get_attributes("embedding"))

    # assert same results with OpenAI's implementation
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    blobs = docs.get_attributes("blob")

    with torch.no_grad():
        images = [Image.fromarray(blob) for blob in blobs]
        tensors = [preprocess(img) for img in images]
        tensor = torch.stack(tensors)
        expected_embedding = model.encode_image(tensor).numpy()

    np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)
