from pathlib import Path

import numpy as np
import scipy
from jina import Document, DocumentArray, Executor

from ...tfidf_text_executor import TFIDFTextEncoder


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.path_vectorizer.endswith('tfidf_vectorizer.pickle')


def test_tfidf_text_encoder():
    text = 'Han likes eating pizza'

    encoder = TFIDFTextEncoder()
    doc = Document(text=text)
    docarray = DocumentArray([doc])
    encoder.encode(docarray, parameters={})
    embedding = doc.embedding

    expected = scipy.sparse.load_npz(Path(__file__).parent / 'expected.npz')
    np.testing.assert_almost_equal(embedding.todense(), expected.todense(), decimal=4)
    assert expected.shape[0] == 1


def test_tfidf_text_encoder_batch():
    # Input
    text_batch = ['Han likes eating pizza', 'Han likes pizza', 'Jina rocks']

    # Encoder embedding
    encoder = TFIDFTextEncoder()
    doc0 = Document(text=text_batch[0])
    doc1 = Document(text=text_batch[1])
    doc2 = Document(text=text_batch[2])
    docarray = DocumentArray([doc0, doc1, doc2])
    encoder.encode(docarray, parameters={})
    embeddeding_batch = scipy.sparse.vstack(docarray.get_attributes('embedding'))

    # Compare with ouptut
    expected_batch = scipy.sparse.load_npz(Path(__file__).parent / 'expected_batch.npz')
    np.testing.assert_almost_equal(
        embeddeding_batch.todense(), expected_batch.todense(), decimal=2
    )
    assert expected_batch.shape[0] == len(text_batch)
