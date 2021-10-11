import os

import pytest
from jina import Document, DocumentArray
from jina.excepts import PretrainedModelFileDoesNotExist
from spell_checker import SpellChecker

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_spell_checker_no_correction(model_path, correct_text):
    spell_checker = SpellChecker(model_path=model_path)
    processed_docs = DocumentArray([Document(content=t) for t in correct_text])
    spell_checker.spell_check(processed_docs)

    assert len(processed_docs) == len(correct_text)
    for crafted_doc, expected in zip(processed_docs, correct_text):
        assert crafted_doc.content == expected


def test_spell_checker_correct(model_path, incorrect_text, correct_text):
    spell_checker = SpellChecker(model_path=model_path)
    processed_docs = DocumentArray([Document(content=t) for t in incorrect_text])
    spell_checker.spell_check(processed_docs)

    assert len(processed_docs) == len(incorrect_text)
    for crafted_doc, expected in zip(processed_docs, correct_text):
        assert crafted_doc.content == expected
