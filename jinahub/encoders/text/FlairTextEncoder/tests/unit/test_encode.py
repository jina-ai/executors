import numpy as np
import pytest
from jina import DocumentArray, Document
from jinahub.encoder.flair_text import FlairTextEncoder


@pytest.fixture()
def docs_generator():
    return DocumentArray((Document(text='random text') for _ in range(30)))


def test_flair_batch(docs_generator):
    encoder = FlairTextEncoder(pooling_strategy='mean')
    docs = docs_generator
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})

    assert len(docs.get_attributes('embedding')) == 30
    assert docs[0].embedding.shape == (100,)


def test_traversal_path():
    text = 'blah'
    docs = DocumentArray([Document(id='root1', text=text)])
    docs[0].chunks = [Document(id='chunk11', text=text),
                      Document(id='chunk12', text=text),
                      Document(id='chunk13', text=text)
                      ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', text=text),
        Document(id='chunk112', text=text),
    ]

    encoder = FlairTextEncoder()
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['c']})

    for path, count in [[['r'], 0], [['c'], 3], [['cc'], 0]]:
        assert len(docs.traverse_flat(path).get_attributes('embedding')) == count
        if count > 0:
            assert docs.traverse_flat(path).get_attributes('embedding')[0].shape == (100,)

    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['cc']})
    for path, count in [[['r'], 0], [['c'], 3], [['cc'], 2]]:
        assert len(docs.traverse_flat(path).get_attributes('embedding')) == count
        if count > 0:
            assert docs.traverse_flat(path).get_attributes('embedding')[0].shape == (100,)


def test_no_documents():
    encoder = FlairTextEncoder()
    docs = []
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})
    assert not docs


def test_flair_word_encode():
    docs = []
    words = ['apple', 'banana1', 'banana2', 'studio', 'satelite', 'airplane']
    for word in words:
        docs.append(Document(text=word))

    text_encoder = FlairTextEncoder()
    text_encoder.encode(DocumentArray(docs), {})

    txt_to_ndarray = {}
    for d in docs:
        txt_to_ndarray[d.text] = d.embedding

    def dist(a, b):
        nonlocal txt_to_ndarray
        a_embedding = txt_to_ndarray[a]
        b_embedding = txt_to_ndarray[b]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satelite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
