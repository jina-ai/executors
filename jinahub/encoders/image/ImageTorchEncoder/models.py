""" Helper module to manage torch vision models """
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.alexnet import __all__ as all_alexnet_models
from torchvision.models.densenet import __all__ as all_densenet_models
from torchvision.models.googlenet import __all__ as all_googlenet_models
from torchvision.models.mnasnet import __all__ as all_mnasnet_models
from torchvision.models.mobilenet import __all__ as all_mobilenet_models
from torchvision.models.resnet import __all__ as all_resnet_models
from torchvision.models.squeezenet import __all__ as all_squeezenet_models
from torchvision.models.vgg import __all__ as all_vgg_models


class EmbeddingModelWrapper:
    """
    The ``EmbeddingModelWrapper`` acts as an unified interface to the `torchvision` models.
    It hides the model specific logic for computing embeddings from the user.

    :param model_name: Name of the `torchvision` model. Classnames are not allowed, i.e.
                       use `resnet_18` instead of `ResNet`.
    :param device: Which device the model runs on. Can be 'cpu' or 'cuda'
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._layer_name = _ModelCatalogue.get_layer_name(model_name)
        self._model = getattr(models, model_name)(pretrained=True)

        self.device = device

        self._pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self._pooling_layer.to(torch.device(self.device))

    def _pooling_function(self, tensor_in: 'torch.Tensor') -> 'torch.Tensor':
        return torch.flatten(self._pooling_layer(tensor_in), 1)

    def get_features(self, content: 'torch.Tensor') -> 'torch.Tensor':
        feature_map = None

        def get_activations(model, model_input, output):
            nonlocal feature_map
            feature_map = output.detach()

        layer = getattr(self._model, self._layer_name)
        handle = layer.register_forward_hook(get_activations)
        self._model(content)
        handle.remove()
        return feature_map

    def compute_embeddings(self, images: 'np.ndarray') -> 'np.ndarray':
        tensor = torch.from_numpy(images).to(self.device)
        features = self.get_features(tensor)
        features = self._pooling_function(features)
        features = features.detach().numpy()
        return features


class _ModelCatalogue:
    # maps the tuple of available model names to the layer from which we want to
    # extract the embedding. Removes the first entry because it the model class
    # not the factory method.
    all_supported_models_to_layer_mapping = {
        tuple(all_resnet_models[1:]): 'layer4',
        tuple(all_alexnet_models[1:]): 'features',
        tuple(all_vgg_models[1:]): 'features',
        tuple(all_squeezenet_models[1:]): 'features',
        tuple(all_densenet_models[1:]): 'features',
        tuple(all_mnasnet_models[1:]): 'layers',
        tuple(all_mobilenet_models[1:]): 'features',
        tuple(all_googlenet_models[1:]): 'inception5b',
    }

    @classmethod
    def is_model_supported(cls, model_name: str):
        return any([model_name in m for m in cls.all_supported_models_to_layer_mapping])

    @classmethod
    def get_layer_name(cls, model_name: str) -> str:
        """
        Checks if model is supported and returns the lookup on the layer name.

        :param model_name: Name of the layer
        """
        if not cls.is_model_supported(model_name):
            raise ValueError(
                f'Model with name {model_name} is not supported. '
                f'Supported models are: {cls.all_supported_models_to_layer_mapping.keys()}'
            )

        for (
            model_names,
            layer_name,
        ) in cls.all_supported_models_to_layer_mapping.items():
            if model_name in model_names:
                return layer_name
