import os

import pytest
from pyngramspell import PyNgramSpell


@pytest.fixture
def input_training_data():
    training_sentences = [
        'they can go quite fast',
        'there were the new Japanese Honda',
    ]
    return training_sentences


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
