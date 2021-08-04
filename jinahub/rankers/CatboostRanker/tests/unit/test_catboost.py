import os

import pytest


def test_init(ranker):
    assert not ranker.model.is_fitted()


def test_train(ranker, documents_to_train_stub_model):
    ranker.train(docs=documents_to_train_stub_model)
    assert ranker.model.is_fitted()


def test_train_with_weights(ranker_with_weight, documents_to_train_stub_model):
    """Weight field specify the importance of the features."""
    ranker_with_weight.train(docs=documents_to_train_stub_model)
    assert ranker_with_weight.model.is_fitted()


def test_dump_load(ranker, documents_to_train_stub_model, tmpdir):
    model_path = str(tmpdir) + '/model.cbm'
    ranker.train(docs=documents_to_train_stub_model)
    assert ranker.model.is_fitted()
    ranker.dump(parameters={'model_path': model_path})
    assert os.path.exists(model_path)
    ranker.load({'model_path': model_path})
    assert ranker.model.is_fitted()


def test_predict(
    ranker, documents_to_train_stub_model, documents_to_train_price_sensitive_model
):
    pass
