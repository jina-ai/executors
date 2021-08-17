import os
from pathlib import Path

from jina import Executor


def test_config():
    ex = Executor.load_config(
        str(Path(__file__).parents[2] / 'config.yml'),
        uses_with={
            'query_features': ['query'],
            'match_features': ['match'],
            'relevance_label': 'rel',
        },
    )
    assert ex.q_features == ['query']


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
    print(model_path)
    assert os.path.exists(model_path)
    ranker.load({'model_path': model_path})
    assert ranker.model.is_fitted()


def test_rank(
    ranker, documents_to_train_stub_model, documents_without_label_random_price
):
    ranker.train(docs=documents_to_train_stub_model)
    assert ranker.model.is_fitted()
    matches_before_rank = documents_without_label_random_price.traverse_flat(['m'])
    for match in matches_before_rank:
        assert not match.scores.get('relevance').value
    ranker.rank(documents_without_label_random_price)
    matches_after_rank = documents_without_label_random_price.traverse_flat(['m'])
    for match in matches_after_rank:
        assert isinstance(match.scores.get('relevance').value, float)


def test_rank_price_sensitive_model(
    ranker,
    documents_to_train_price_sensitive_model,
    documents_without_label_random_brand,
):
    """train the model using price sensitive data, assure higher price get lower relevance score."""
    ranker.train(docs=documents_to_train_price_sensitive_model)
    assert ranker.model.is_fitted()
    ranker.rank(documents_without_label_random_brand)
    for doc in documents_without_label_random_brand:
        predicted_relevances = []
        predicted_ids = []
        expected_ids = ['3', '4', '2', '1']  # Price smaller to large.
        for match in doc.matches:
            predicted_relevances.append(match.scores.get('relevance').value)
            predicted_ids.append(match.id)
        assert (
            predicted_relevances[0]
            >= predicted_relevances[1]
            >= predicted_relevances[2]
        )
        assert predicted_ids == expected_ids
