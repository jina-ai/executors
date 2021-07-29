__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

from typing import Dict, Callable

import pytest

from jina import DocumentArray

from ...transform_encoder  import TransformerTorchEncoder


MODELS_TO_TEST = [
    'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
    'bert-base-uncased',
    'distilroberta-base',
    'distilbert-base-cased-distilled-squad',
]


@pytest.mark.parametrize(
    'model_name', MODELS_TO_TEST
)
def test_load_torch_models(model_name: str, data_generator: Callable):
    encoder = TransformerTorchEncoder(pretrained_model_name_or_path=model_name)

    docs = DocumentArray([doc for doc in data_generator()])
    encoder.encode(
        docs=docs,
        parameters={}
    )

    for doc in docs:
        assert doc.embedding is not None
