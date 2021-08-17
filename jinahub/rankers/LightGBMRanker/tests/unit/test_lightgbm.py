import os
from pathlib import Path

from jina import Executor


def test_config():
    ranker = Executor.load_config(
        str(Path(__file__).parents[2] / 'config.yml'),
        override_with={
            'query_features': ['query'],
            'match_features': ['match'],
            'relevance_label': 'relevance',
        },
    )
    assert ranker.query_features == ['query']
    assert ranker.match_features == ['match']


def test_train(ranker, documents_to_train_price_sensitive_model):
    ranker.train(docs=documents_to_train_price_sensitive_model)
    assert ranker.booster


def test_train_with_categorical_features(
    ranker_with_categorical_features, documents_to_train_price_sensitive_model
):
    """Weight field specify the importance of the features."""
    ranker_with_categorical_features.train(
        docs=documents_to_train_price_sensitive_model
    )
    assert ranker_with_categorical_features.booster


def test_dump_load(ranker, documents_to_train_price_sensitive_model, tmpdir):
    model_path = str(tmpdir) + 'model.txt'
    ranker.train(docs=documents_to_train_price_sensitive_model)
    assert ranker.booster
    ranker.dump(parameters={'model_path': model_path})
    assert os.path.exists(model_path)
    ranker.load({'model_path': model_path})
    assert ranker.booster


def test_rank_price_sensitive_model(
    ranker,
    documents_to_train_price_sensitive_model,
    documents_random_brand,
):
    """train the model using price sensitive data, assure higher price get lower relevance score."""
    ranker.train(docs=documents_to_train_price_sensitive_model)
    assert ranker.booster
    ranker.rank(documents_random_brand)
    for doc in documents_random_brand:
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
