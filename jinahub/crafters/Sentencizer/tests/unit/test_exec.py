__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from jina import Document, DocumentArray
from jinahub.segmenter.sentencizer import Sentencizer


def test_executor():
    ex = Sentencizer.load_config('../../config.yml')
    input = DocumentArray([Document(text='Hello. World.')])
    ex.segment(input, {})
    assert input[0].chunks[0].text == 'Hello.'
    assert input[0].chunks[1].text == 'World.'
