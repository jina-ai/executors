__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from pathlib import Path

import pytest
import spacy

from jina import Document, DocumentArray, Executor

from ...spacy_text_encoder import SpacyTextEncoder


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.lang == 'en_core_web_sm'


def test_spacy_text_encoder():
    # Input
    docs = DocumentArray(
        [
            Document(text='Han likes eating pizza'),
            Document(text='Han likes pizza'),
            Document(text='Jina rocks'),
        ]
    )

    # Encoder embedding
    encoder = SpacyTextEncoder()
    encoder.encode(docs, parameters={})
    # Compare with ouptut
    assert len(docs) == 3
    for doc in docs:
        assert doc.embedding.shape == (96,)


@pytest.mark.gpu
def test_spacy_text_encoder_gpu():
    # Input
    docs = DocumentArray(
        [
            Document(text='Jina rocks'),
        ]
    )
    # Encoder embedding
    encoder = SpacyTextEncoder(device='cuda')
    encoder.encode(docs, parameters={})
    assert len(docs) == 1
    assert docs[0].embedding.shape == (96,)


def test_spacy_text_encoder_traversal_paths():
    # Input
    docs = DocumentArray(
        [
            Document(
                chunks=[
                    Document(text='Han likes eating pizza'),
                    Document(text='Han likes pizza'),
                ]
            ),
            Document(chunks=[Document(text='Jina rocks')]),
        ]
    )

    # Encoder embedding
    encoder = SpacyTextEncoder()
    encoder.encode(docs, parameters={'traversal_paths': ['c']})
    # Compare with ouptut
    assert len(docs) == 2
    assert len(docs[0].chunks) == 2
    for chunk in docs[0].chunks:
        assert chunk.embedding.shape == (96,)
    assert len(docs[1].chunks) == 1
    for chunk in docs[1].chunks:
        assert chunk.embedding.shape == (96,)


def test_unsupported_lang(tmpdir):
    dummy1 = spacy.blank('xx')
    dummy1_dir_path = os.path.join(tmpdir, 'xx1')
    dummy1.to_disk(dummy1_dir_path)
    dummy2 = spacy.blank('xx')
    dummy2_dir_path = os.path.join(tmpdir, 'xx2')
    dummy2.to_disk(dummy2_dir_path)
    # No available language
    with pytest.raises(IOError):
        SpacyTextEncoder('abcd')

    # Language does not have DependencyParser should thrown an error
    # when try to use default encoder
    with pytest.raises(ValueError):
        SpacyTextEncoder(dummy1_dir_path, use_default_encoder=True)

    # And should be fine when 'parser' pipeline is added
    dummy1.add_pipe('parser')
    dummy1.to_disk(dummy1_dir_path)
    SpacyTextEncoder(dummy1_dir_path, use_default_encoder=True)

    # Language does not have SentenceRecognizer should thrown an error
    # when try to use non default encoder
    with pytest.raises(ValueError):
        SpacyTextEncoder(dummy2_dir_path, use_default_encoder=False)

    # And should be fine when 'senter' pipeline is added
    dummy2.add_pipe('tok2vec')
    dummy2.to_disk(dummy2_dir_path)
    SpacyTextEncoder(dummy2_dir_path, use_default_encoder=False)
