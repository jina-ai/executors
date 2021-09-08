__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path

import pytest
from jina import Document, DocumentArray, Executor

from ...laser_encoder import LaserEncoder


@pytest.fixture()
def docs_generator():
    return DocumentArray((Document(text='random text') for _ in range(30)))


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.language == 'en'


def test_flair_batch(docs_generator):
    encoder = LaserEncoder()
    docs = docs_generator
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})

    assert len(docs.get_attributes('embedding')) == 30
    assert docs[0].embedding.shape == (1024,)


@pytest.mark.gpu
def test_gpu_encoding(docs_generator):
    encoder = LaserEncoder(device='cuda')
    docs = DocumentArray((Document(text='random text')))
    encoder.encode(docs, {})

    assert len(docs.get_attributes('embedding')) == 1
    assert docs[0].embedding.shape == (1024,)


def test_traversal_path():
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

    encoder = LaserEncoder()
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['c']})

    for path, count in [[['r'], 0], [['c'], 3], [['cc'], 0]]:
        embeddings = docs.traverse_flat(path).get_attributes('embedding')
        assert len([em for em in embeddings if em is not None]) == count
        if count > 0:
            assert docs.traverse_flat(path).get_attributes('embedding')[0].shape == (
                1024,
            )

    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['cc']})
    for path, count in [[['r'], 0], [['c'], 3], [['cc'], 2]]:
        embeddings = docs.traverse_flat(path).get_attributes('embedding')
        assert len([em for em in embeddings if em is not None]) == count
        if count > 0:
            assert docs.traverse_flat(path).get_attributes('embedding')[0].shape == (
                1024,
            )


def test_no_documents():
    encoder = LaserEncoder()
    docs = []
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})
    assert not docs
