from typing import List

import pytest
import torch
from jina import Document, DocumentArray
from jina.executors import BaseExecutor

from ...dpr_text import DPRTextEncoder


@pytest.fixture(scope='session')
def basic_encoder() -> DPRTextEncoder:
    return DPRTextEncoder()


def test_config():
    encoder = BaseExecutor.load_config('../../config.yml')
    assert encoder.default_batch_size == 32
    assert encoder.default_traversal_paths == ['r']


def test_no_document(basic_encoder: DPRTextEncoder):
    basic_encoder.encode(None, {})


def test_empty_documents(basic_encoder: DPRTextEncoder):
    docs = DocumentArray([])
    basic_encoder.encode(docs, {})
    assert len(docs) == 0


def test_no_text_documents(basic_encoder: DPRTextEncoder):
    docs = DocumentArray([Document()])
    basic_encoder.encode(docs, {})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_wrong_encoder_type():
    with pytest.raises(ValueError, match='The ``encoder_type`` parameter'):
        encoder = DPRTextEncoder(encoder_type='worng_type')


def test_encoding_cpu():
    docs = DocumentArray([Document(text='hello there')])
    encoder = DPRTextEncoder(device='cpu')
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (768,)


def test_encoding_question_type():
    docs = DocumentArray([Document(text='hello there')])
    encoder = DPRTextEncoder(
        pretrained_model_name_or_path='facebook/dpr-question_encoder-single-nq-base',
        encoder_type='question',
        device='cpu',
    )
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (768,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU is needed for this test')
def test_encoding_gpu():
    docs = DocumentArray([Document(text='hello there')])
    encoder = DPRTextEncoder(device='cuda')
    encoder.encode(docs, {})

    assert docs[0].embedding.shape == (768,)


@pytest.mark.parametrize(
    'traversal_path, counts',
    [
        ('r', [['r', 1], ['c', 0], ['cc', 0]]),
        ('c', [['r', 0], ['c', 3], ['cc', 0]]),
        ('cc', [['r', 0], ['c', 0], ['cc', 2]]),
    ],
)
def test_traversal_path(
    traversal_path: str, counts: List, basic_encoder: DPRTextEncoder
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

    basic_encoder.encode(
        docs=docs, parameters={'traversal_paths': [traversal_path]}, return_results=True
    )
    for path, count in counts:
        assert len(docs.traverse_flat([path]).get_attributes('embedding')) == count


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(basic_encoder: DPRTextEncoder, batch_size: int):
    docs = DocumentArray([Document(text='hello there') for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (768,)


def test_quality_embeddings(basic_encoder: DPRTextEncoder):
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
