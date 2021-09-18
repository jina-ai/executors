__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path
from typing import List

import pytest
from jina import Document, DocumentArray, Executor
from laser_encoder import LaserEncoder

_EMBEDDING_DIM = 1024


@pytest.fixture(scope='session')
def basic_encoder() -> LaserEncoder:
    return LaserEncoder()


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.default_language == 'en'


def test_no_document(basic_encoder: LaserEncoder):
    basic_encoder.encode(None, {})


def test_empty_documents(basic_encoder: LaserEncoder):
    docs = DocumentArray([])
    basic_encoder.encode(docs, {})
    assert len(docs) == 0


@pytest.mark.gpu
def test_encoding_gpu():
    encoder = LaserEncoder(device='cuda')
    docs = DocumentArray((Document(text='random text')))
    encoder.encode(docs, {})

    assert len(docs.get_attributes('embedding')) == 1
    assert docs[0].embedding.shape == (1024,)


def test_no_text_documents(basic_encoder: LaserEncoder):
    docs = DocumentArray([Document()])
    basic_encoder.encode(docs, {})
    assert len(docs) == 1
    assert docs[0].embedding is None


def test_encoding_cpu(basic_encoder: LaserEncoder):
    docs = DocumentArray([Document(text='hello there')])
    basic_encoder.encode(docs, {})

    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)


@pytest.mark.parametrize(
    'language, sentence',
    [
        ('en', 'Today is a nice day'),
        ('es', 'hoy es un buen día'),
        ('ru', 'сегодня хороший день'),
    ],
)
def test_languages(language: str, sentence: str, basic_encoder: LaserEncoder):
    docs = DocumentArray([Document(text=sentence)])
    basic_encoder.encode(docs, {'language': language})

    assert docs[0].embedding.shape == (_EMBEDDING_DIM,)


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
    traversal_paths: List[str], counts: List, basic_encoder: LaserEncoder
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


def test_no_documents():
    encoder = LaserEncoder()
    docs = []
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})
    assert not docs


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8])
def test_batch_size(basic_encoder: LaserEncoder, batch_size: int):
    docs = DocumentArray([Document(text='hello there') for _ in range(32)])
    basic_encoder.encode(docs, parameters={'batch_size': batch_size})

    for doc in docs:
        assert doc.embedding.shape == (_EMBEDDING_DIM,)


def test_quality_embeddings(basic_encoder: LaserEncoder):
    docs = DocumentArray(
        [
            # Different than usual example - because embeddings suck (manually verified
            # using the laser embedings module)
            Document(id='A', text='car'),
            Document(id='B', text='truck'),
            Document(id='C', text='radio'),
            Document(id='D', text='TV'),
        ]
    )

    basic_encoder.encode(DocumentArray(docs), {})

    # assert semantic meaning is captured in the encoding
    docs.match(docs)
    matches = ['B', 'A', 'D', 'C']
    for i, doc in enumerate(docs):
        assert doc.matches[1].id == matches[i]
