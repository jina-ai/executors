import pytest


def test_init(ranker):
    assert not ranker.model.is_fitted()


def test_train(ranker, documents_to_train_stub_model):
    ranker.train(docs=documents_to_train_stub_model)
