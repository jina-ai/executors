__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict

import pytest

import numpy as np
from jina import DocumentArray, Document

try:
    from torch_encoder import ImageTorchEncoder
except:
    from jinahub.image.encoder.torch_encoder import ImageTorchEncoder


MODELS_TO_TEST = [
    'mobilenet_v2',
    'squeezenet1_0',
    'alexnet',
    'vgg11',
    'densenet121',
    'mnasnet0_5',
]


@pytest.mark.parametrize(
    'model_name', MODELS_TO_TEST
)
def test_load_torch_models(model_name: str, test_images: Dict[str, np.array]):
    encoder = ImageTorchEncoder(model_name=model_name)

    docs = DocumentArray([Document(blob=img_arr) for img_arr in test_images.values()])
    encoder.encode(
        docs=docs,
        parameters={}
    )

    for doc in docs:
        assert doc.embedding is not None
