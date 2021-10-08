import os

import pytest
from jina import Document, DocumentArray
from jina.excepts import PretrainedModelFileDoesNotExist
from pyngramspell import PyNgramSpell
from spell_checker import SpellChecker

cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_missing_model_path():
    wrong_path_vectorizer = os.path.join(cur_dir, '/model/non_existant.pickle')
    with pytest.raises(PretrainedModelFileDoesNotExist):
        _ = SpellChecker(wrong_path_vectorizer)


@pytest.fixture
def input_training_data():
    correct_sentences = [
        ['they', 'can', 'go', 'quite', 'fast'],
        ['there', 'were', 'the', 'new', 'Japanese', 'Honda'],
    ]
    yield [' '.join(x) for x in correct_sentences]


@pytest.fixture
def model_path(tmpdir, input_training_data):
    model_path = os.path.join(str(tmpdir), 'tmp_model.pickle')
    speller = PyNgramSpell(min_freq=0)
    speller.fit(input_training_data)
    speller.save(model_path)
    yield model_path


@pytest.fixture()
def correct_text():
    return [
        'they can go quite fast',
        'they can go',
        'there japanese honda',
        'new fast honda',
        'the new fast japanese',
    ]


@pytest.fixture()
def incorrect_text():
    return [
        'they can go quit fast',
        'they cn go',
        'there japanes hnda',
        'new fast honda',
        'the new fst japanse',
    ]


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
