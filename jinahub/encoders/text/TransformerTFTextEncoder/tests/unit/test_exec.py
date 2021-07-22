__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import numpy as np
import pytest
from jina import Document, DocumentArray
from jinahub.encoder.transformer_tf_text_encode import TransformerTFTextEncoder

target_dim = 768


@pytest.fixture()
def docs_generator():
    return DocumentArray((Document(text='random text') for _ in range(30)))


def test_tf_batch(docs_generator):
    encoder = TransformerTFTextEncoder()
    docs = docs_generator
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})

    assert len(docs.get_attributes('embedding')) == 30
    assert docs[0].embedding.shape == (target_dim,)


def test_encodes_semantic_meaning():
    sentences = dict()
    sentences['A'] = 'Hello, my name is Michael.'
    sentences['B'] = 'Today we are going to Disney World.'
    sentences['C'] = 'There are animals on the road'
    sentences['D'] = 'A dog is running down the road'

    encoder = TransformerTFTextEncoder()

    embeddings = {}
    for id_, sentence in sentences.items():
        docs = DocumentArray([Document(text=sentence)])
        encoder.encode(docs, parameters={})
        embeddings[id_] = docs[0].embedding

    def dist(a, b):
        a_embedding = embeddings[a]
        b_embedding = embeddings[b]
        return np.linalg.norm(a_embedding - b_embedding)

    small_distance = dist('C', 'D')
    assert small_distance < dist('C', 'B')
    assert small_distance < dist('C', 'A')
    assert small_distance < dist('B', 'A')


@pytest.mark.parametrize(
    ['docs', 'docs_per_path', 'traversal_path'],
    [
        (pytest.lazy_fixture('docs_with_text'), [[['r'], 10], [['c'], 0], [['cc'], 0]], ['r']),
        (
            pytest.lazy_fixture("docs_with_chunk_text"),
            [[['r'], 0], [['c'], 10], [['cc'], 0]],
            ['c'],
        ),
        (
            pytest.lazy_fixture("docs_with_chunk_chunk_text"),
            [[['r'], 0], [['c'], 0], [['cc'], 10]],
            ['cc'],
        ),
    ],
)
def test_traversal_path(docs: DocumentArray, docs_per_path, traversal_path):
    encoder = TransformerTFTextEncoder()
    encoder.encode(docs, parameters={'traversal_paths': traversal_path})

    for path, count in docs_per_path:
        assert len(docs.traverse_flat(path).get_attributes("embedding")) == count
