from pathlib import Path
from typing import List

import numpy as np
import pytest
import scipy
from jina import Document, DocumentArray, Executor
from tfidf_text_executor import TFIDFTextEncoder

_EMBEDDING_DIM = 130107


@pytest.fixture(scope='session')
def basic_encoder() -> TFIDFTextEncoder:
    return TFIDFTextEncoder()


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.__class__.__name__ == 'TFIDFTextEncoder'


def test_error_no_file():
    pass


def test_no_document(basic_encoder: TFIDFTextEncoder):
    basic_encoder.encode(None, {})


def test_empty_documents(basic_encoder: TFIDFTextEncoder):
    docs = DocumentArray([])
    basic_encoder.encode(docs, {})
    assert len(docs) == 0


def test_no_text_documents(basic_encoder: TFIDFTextEncoder):
    docs = DocumentArray([Document()])
    basic_encoder.encode(docs, {})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_tfidf_text_encoder(basic_encoder: TFIDFTextEncoder):
    doc = Document(text='Han likes eating pizza')
    docarray = DocumentArray([doc])
    basic_encoder.encode(docarray, parameters={})
    embedding = doc.embedding

    assert embedding.shape == (1, _EMBEDDING_DIM)
    assert embedding.size == 4


def test_tfidf_text_encoder_batch(basic_encoder: TFIDFTextEncoder):
    # Input
    text_batch = ['Han likes eating pizza', 'Han likes pizza', 'Jina rocks']

    # Encoder embedding
    docarray = DocumentArray([Document(text=text) for text in text_batch])
    basic_encoder.encode(docarray, parameters={})
    embeddeding_batch = scipy.sparse.vstack(docarray.get_attributes('embedding'))

    assert embeddeding_batch.shape == (3, _EMBEDDING_DIM)
    assert embeddeding_batch.size == 8

    embs = np.asarray(embeddeding_batch.todense())

    # They overlap in Han
    assert (embs[0] * embs[1]).sum() > 0.1

    # They do not overlap
    assert (embs[0] * embs[2]).sum() == 0


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
    traversal_paths: List[str], counts: List, basic_encoder: TFIDFTextEncoder
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
        embeddings = docs.traverse_flat([path]).get_attributes('embedding')
        assert len(list(filter(lambda x: x is not None, embeddings))) == count


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(basic_encoder: TFIDFTextEncoder, batch_size: int):
    docs = DocumentArray([Document(text='hello there') for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (1, _EMBEDDING_DIM)
