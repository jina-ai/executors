__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from pathlib import Path

from jina import Document, DocumentArray, Executor
from sentencizer import Sentencizer


def test_config():
    ex = Executor.load_config(str(Path(__file__).parents[2] / 'config.yml'))
    assert ex.min_sent_len == 1


def test_executor():
    ex = Sentencizer()
    input = DocumentArray([Document(text='Hello. World.')])
    ex.segment(input, {})
    assert input[0].chunks[0].text == 'Hello.'
    assert input[0].chunks[1].text == 'World.'
