from pathlib import Path
from typing import List

import clip
import numpy as np
import pytest
import torch
from clip_text import CLIPTextEncoder
from jina import Document, DocumentArray, Executor


@pytest.fixture(scope="module")
def encoder() -> CLIPTextEncoder:
    return CLIPTextEncoder()


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.pretrained_model_name_or_path == 'openai/clip-vit-base-patch32'


def test_no_documents(encoder: CLIPTextEncoder):
    docs = DocumentArray()
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 0


def test_none_docs(encoder: CLIPTextEncoder):
    encoder.encode(docs=None, parameters={})


def test_docs_no_texts(encoder: CLIPTextEncoder):
    docs = DocumentArray([Document()])
    encoder.encode(docs=DocumentArray(), parameters={})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_compute_tokens(encoder: CLIPTextEncoder):
    tokens = encoder._generate_input_tokens(
        ["hello this is a test", "and another test"]
    )

    assert tokens["input_ids"].shape == (2, 7)
    assert tokens["attention_mask"].shape == (2, 7)


def test_encoding_cpu():
    encoder = CLIPTextEncoder(device="cpu")
    input_data = DocumentArray([Document(text="hello world")])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (512,)


@pytest.mark.gpu
def test_encoding_gpu():
    encoder = CLIPTextEncoder(device="cuda")
    input_data = DocumentArray([Document(text="hello world")])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (512,)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batch_size(encoder: CLIPTextEncoder, batch_size: int):
    text = "Jina is Lit"
    docs = DocumentArray([Document(text=text) for _ in range(32)])
    encoder.encode(docs, parameters={"batch_size": batch_size})

    for doc in docs:
        assert doc.embedding.shape == (512,)


def test_encodes_semantic_meaning():
    """
    Check if the distance between embeddings of similar sentences are smaller
    than dissimilar pair of sentences.
    """

    docs = DocumentArray(
        [
            Document(id="A", text="a furry animal that with a long tail"),
            Document(id="B", text="a domesticated mammal with four legs"),
            Document(id="C", text="a type of aircraft that uses rotating wings"),
            Document(id="D", text="flying vehicle that has fixed wings and engines"),
        ]
    )

    clip_text_encoder = CLIPTextEncoder()
    clip_text_encoder.encode(DocumentArray(docs), {})

    # assert semantic meaning is captured in the encoding
    docs.match(docs)
    matches = ["B", "A", "D", "C"]
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]


def test_openai_embed_match():
    docs = []
    sentences = [
        "Jina AI is lit",
        "Jina AI is great",
        "Jina AI is a cloud-native neural search company",
        "Jina AI is a github repo",
        "Jina AI is an open source neural search project",
    ]
    for sentence in sentences:
        docs.append(Document(text=sentence))

    clip_text_encoder = CLIPTextEncoder("openai/clip-vit-base-patch32")
    clip_text_encoder.encode(DocumentArray(docs), {})

    txt_to_ndarray = {}
    for d in docs:
        txt_to_ndarray[d.text] = d.embedding

    # assert same results with OpenAI's implementation
    model, _ = clip.load("ViT-B/32", device="cpu")
    assert len(txt_to_ndarray) == 5
    for text, actual_embedding in txt_to_ndarray.items():
        with torch.no_grad():
            tokens = clip.tokenize(text)
            expected_embedding = model.encode_text(tokens).detach().numpy().flatten()

        np.testing.assert_almost_equal(actual_embedding, expected_embedding, 5)


@pytest.mark.parametrize(
    'traversal_paths, counts',
    [
        (['r'], [['r', 1], ['c', 0], ['cc', 0]]),
        (['c'], [['r', 0], ['c', 3], ['cc', 0]]),
        (['cc'], [['r', 0], ['c', 0], ['cc', 2]]),
        (['cc', 'r'], [['r', 1], ['c', 0], ['cc', 2]]),
    ],
)
def test_traversal_path(
    traversal_paths: List[str], counts: List, encoder: CLIPTextEncoder
):
    text = 'blah'
    docs = DocumentArray([Document(id='root1', text=text)])
    docs[0].chunks = [
        Document(id='chunk11', text=text),
        Document(id='chunk12', text=text),
        Document(id='chunk13', text=text),
    ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', text=text),
        Document(id='chunk112', text=text),
    ]

    encoder.encode(docs=docs, parameters={'traversal_paths': traversal_paths})
    for path, count in counts:
        embeddings = docs.traverse_flat([path]).get_attributes('embedding')
        assert len(list(filter(lambda x: x is not None, embeddings))) == count
