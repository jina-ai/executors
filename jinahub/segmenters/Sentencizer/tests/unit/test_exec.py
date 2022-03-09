from pathlib import Path

import pytest
from jina import Document, DocumentArray, Executor
from sentencizer import Sentencizer


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.min_sent_len == 1


@pytest.mark.parametrize('traversal_paths', ['r', 'c'])
def test_executor(traversal_paths):
    ex = Sentencizer(traversal_paths=traversal_paths)
    doc = Document(text='Hello. World! Go? Back')
    if 'c' in traversal_paths:
        da = DocumentArray([Document(chunks=[doc])])
    else:
        da = DocumentArray([doc])
    ex.segment(da, {})
    flattened_docs = da.traverse_flat(traversal_paths)
    assert len(flattened_docs) == 1
    assert len(flattened_docs[0].chunks) == 4
    assert flattened_docs[0].chunks[0].text == 'Hello.'
    assert flattened_docs[0].chunks[1].text == 'World!'
    assert flattened_docs[0].chunks[2].text == 'Go?'
    assert flattened_docs[0].chunks[3].text == 'Back'


def test_executor_with_punct_chars():
    ex = Sentencizer(punct_chars=['.'])
    da = DocumentArray([Document(text='Hello. World! Go? Back')])
    ex.segment(da, {})
    assert len(da) == 1
    assert len(da[0].chunks) == 2
    assert da[0].chunks[0].text == 'Hello.'
    assert da[0].chunks[1].text == 'World! Go? Back'


def test_executor_with_max_sent_length():
    ex = Sentencizer(punct_chars=['.'], max_sent_len=3)
    da = DocumentArray([Document(text='Hello. World')])
    ex.segment(da, {})
    assert len(da) == 1
    assert len(da[0].chunks) == 2
    assert da[0].chunks[0].text == 'Hel'
    assert da[0].chunks[1].text == 'Wor'


def test_executor_empty_input():
    ex = Sentencizer()
    da = DocumentArray()
    ex.segment(da, {})
    assert len(da) == 0


def test_executor_none_input():
    ex = Sentencizer()
    ex.segment(None, {})
