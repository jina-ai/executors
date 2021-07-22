__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import pytest
import numpy as np
import torch

from jinahub.image.encoder.models import EmbeddingModelWrapper, _ModelCatalogue


@pytest.mark.parametrize(
    ['model_name', 'is_supported'],
    [
        ('ResNet', False),
        ('resnet18', True),
        ('resnet50', True),
        ('alexnet', True),
        ('xResNet', False),
        ('Alexnet', False)
    ]
)
def test_is_model_supported(model_name: str, is_supported: bool):
    assert _ModelCatalogue.is_model_supported(model_name) == is_supported


@pytest.mark.parametrize(
    ['model_name', 'layer'],
    [
        ('alexnet', 'features'),
        ('vgg11', 'features'),
        ('squeezenet1_0', 'features'),
        ('densenet121', 'features'),
        ('mnasnet0_5', 'layers'),
        ('mobilenet_v2', 'features'),
    ]
)
def test_is_correct_layer(model_name: str, layer: str):
    assert _ModelCatalogue.get_layer_name(model_name) == layer


@pytest.mark.parametrize(
    ['model_name', 'dim'],
    [
        ('mobilenet_v2', 1280),
        ('resnet18', 512)
    ]
)
def test_get_features(model_name: str, dim: int):
    model_wrapper = EmbeddingModelWrapper(model_name)

    embeddings = model_wrapper.compute_embeddings(
        np.ones((10, 3, 224, 224), dtype=np.float32)
    )

    assert embeddings.shape == (10, dim)


@pytest.mark.parametrize(
    'feature_map',
    [
        np.ones((1, 10, 10, 3)),
        np.random.rand(1, 224, 224, 3),
        np.zeros((1, 100, 100, 3))
    ]
)
def test_get_pooling(
    feature_map: np.ndarray,
):
    wrapper = EmbeddingModelWrapper('mobilenet_v2')

    feature_map_after_pooling = wrapper._pooling_function(torch.from_numpy(feature_map))

    np.testing.assert_array_almost_equal(feature_map_after_pooling, np.mean(feature_map, axis=(2, 3)))


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='requires GPU and CUDA')
def test_get_features_gpu():
    wrapper = EmbeddingModelWrapper('mobilenet_v2')
    arr_in = np.ones((2, 3, 10, 10), dtype=np.float32)

    encodings = wrapper.get_features(torch.from_numpy(arr_in).to(wrapper.device)).detach().cpu().numpy()

    assert encodings.shape == (2, 1280, 1, 1)


def test_get_features_cpu():
    wrapper = EmbeddingModelWrapper('mobilenet_v2', device='cpu')
    arr_in = np.ones((2, 3, 224, 224), dtype=np.float32)

    encodings = wrapper.get_features(torch.from_numpy(arr_in)).detach().numpy()

    assert encodings.shape[1] == 1280
