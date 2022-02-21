from pathlib import Path
from typing import List

import pytest
from jina import Document, DocumentArray, Executor
from sentence_encoder import TransformerSentenceEncoder

_EMBEDDING_DIM = 384


@pytest.fixture(scope='session')
def basic_encoder() -> TransformerSentenceEncoder:
    return TransformerSentenceEncoder()


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.__class__.__name__ == 'TransformerSentenceEncoder'


def test_encoding_cpu():
    enc = TransformerSentenceEncoder(device='cpu')
    input_data = DocumentArray([Document(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.gpu
def test_encoding_gpu():
    enc = TransformerSentenceEncoder(device='cuda')
    input_data = DocumentArray([Document(text='hello world')])

    enc.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    'model_name, emb_dim',
    [
        ('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 384),
        ('sentence-transformers/msmarco-distilbert-base-tas-b', 768),
        ('distilbert-base-uncased', 768),
    ],
)
def test_models(model_name: str, emb_dim: int):
    encoder = TransformerSentenceEncoder(model_name)
    input_data = DocumentArray([Document(text='hello world')])

    encoder.encode(docs=input_data, parameters={})

    assert input_data[0].embedding.shape == (emb_dim,)


@pytest.mark.parametrize(
    'traversal_paths, counts',
    [
        ('@r', [['@r', 1], ['@c', 0], ['@cc', 0]]),
        ('@c', [['@r', 0], ['@c', 3], ['@cc', 0]]),
        ('@cc', [['@r', 0], ['@c', 0], ['@cc', 2]]),
        ('@r,cc', [['@r', 1], ['@c', 0], ['@cc', 2]]),
    ],
)
def test_traversal_path(
    traversal_paths: str, counts: List, basic_encoder: TransformerSentenceEncoder
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

    basic_encoder.encode(docs=docs, parameters={'traversal_paths': traversal_paths})
    for path, count in counts:
        embeddings = DocumentArray(docs[path]).embeddings
        if count == 0:
            assert embeddings is None
        else:
            len(embeddings) == count


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(basic_encoder: TransformerSentenceEncoder, batch_size: int):
    docs = DocumentArray([Document(text='hello there') for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


def test_quality_embeddings(basic_encoder: TransformerSentenceEncoder):
    docs = DocumentArray(
        [
            Document(id='A', text='a furry animal that with a long tail'),
            Document(id='B', text='a domesticated mammal with four legs'),
            Document(id='C', text='a type of aircraft that uses rotating wings'),
            Document(id='D', text='flying vehicle that has fixed wings and engines'),
        ]
    )

    basic_encoder.encode(DocumentArray(docs), {})

    # assert semantic meaning is captured in the encoding
    docs.match(docs)
    matches = ['B', 'A', 'D', 'C']
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]
