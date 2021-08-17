__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest
from jina import Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def model_path(tmpdir):
    """
    Pretrained model to be stored in tmpdir.
    It will be trained on fake data forcing that the first feature correlates positively with relevance, while the rest are
    are random
    :param tmpdir:
    :return:
    """
    model_path = os.path.join(tmpdir, 'model.txt')
    return model_path


def test_train_offline(
    documents_to_train_price_sensitive_model, documents_random_brand, model_path
):
    # Step 1, without a model (train), we manually add score to the document
    # The ranking score is not affected by the model

    for idx, document in enumerate(documents_to_train_price_sensitive_model):
        document.scores['relevance'] = idx  # manually assign score to document
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        resp = f.post(
            on='/search',
            inputs=documents_to_train_price_sensitive_model,
            return_results=True,
        )
        scores = []
        for doc in resp[0].docs:
            scores.append(doc.scores['relevance'].value)
        assert scores[0] < scores[1] < scores[2]

    # Step 2, train a model (train), will leave an model inside workdir
    assert not os.path.exists(model_path)
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        f.post(on='/train', inputs=documents_to_train_price_sensitive_model)
        f.post(on='/dump', parameters={'model_path': model_path})
    assert os.path.exists(model_path)  # after train, dump a model in model path

    # Step 3, call the search endpoint again, assure the relevance
    with Flow.load_config(os.path.join(cur_dir, 'flow.yml')) as f:
        rv = f.post(on='/search', inputs=documents_random_brand, return_results=True)
        scores = []
        for doc in rv[0].docs:
            for match in doc.matches:
                scores.append(match.scores['relevance'].value)
        assert len(scores) == 4
        assert scores[0] >= scores[1] >= scores[2]
