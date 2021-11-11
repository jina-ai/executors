__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import importlib
import os
import types
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from jina import DocumentArray, Executor, requests
from jina.excepts import PretrainedModelFileDoesNotExist
from jina.logging.logger import JinaLogger

cur_dir = os.path.dirname(os.path.abspath(__file__))


class CustomImageTorchEncoder(Executor):
    """
    :class:`CustomImageTorchEncoder` encodes ``Document`` content from a ndarray,
    potentially B x (Channel x Height x Width) into a ndarray of `B x D`.
    Internally, :class:`CustomImageTorchEncoder` wraps any custom torch
    model not part of models from `torchvision.models`.
    https://pytorch.org/docs/stable/torchvision/models.html
    """

    def __init__(
        self,
        model_state_dict_path: Optional[str] = None,
        model_definition_file: Optional[str] = None,
        model_class_name: Optional[str] = None,
        layer_name: Optional[str] = 'features',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs,
    ):
        """
        :param model_state_dict_path: The path where the model state dict is stored.
        :param model_definition_file: The python file path where the model class is defined
        :param model_class_name: The model class name to instantiate with the `state_dict`
            in `model_state_dict_path`
        :param layer_name: The layer name from which to extract the feature maps.
            These feature maps will then be fed into an `AdaptiveAvgPool2d` layer
        to extract the embeddings
        :param traversal_paths: fallback traversal path in case there is no
            traversal path sent in the request
        :param batch_size: fallback batch size in case there is not batch size
            sent in the request
        :param device: The device where to load the model.
        """
        super().__init__(*args, **kwargs)
        self.layer_name = layer_name
        self.logger = JinaLogger(self.__class__.__name__)
        self.device = device
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.model_state_dict_path = model_state_dict_path
        self.model_definition_file = model_definition_file
        self.model_class_name = model_class_name

        if self.model_state_dict_path and (
            not self.model_definition_file or not self.model_class_name
        ):
            raise ValueError(
                'model_state_dict_path option requires to have model_definition_file '
                'and model_class_name parameters'
            )

        if self.model_state_dict_path and os.path.exists(self.model_state_dict_path):
            loader = importlib.machinery.SourceFileLoader(
                '__imported_module__', self.model_definition_file
            )
            mod = types.ModuleType(loader.name)
            loader.exec_module(mod)
            self.model = getattr(mod, self.model_class_name)(*args, **kwargs)
            self.model.load_state_dict(torch.load(self.model_state_dict_path))
            self.model.eval()
            self.model.to(self.device)
        else:
            raise PretrainedModelFileDoesNotExist(
                f'model state dict {self.model_state_dict_path} does not exist'
            )
        self.layer = getattr(self.model, self.layer_name)
        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def _get_features(self, content):
        feature_map = None

        def get_activation(model, model_input, output):
            nonlocal feature_map
            feature_map = output.detach()

        handle = self.layer.register_forward_hook(get_activation)
        self.model(content)
        handle.remove()
        return feature_map

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """
        Encode all docs with images and store the encodings in the embedding attribute
        of the docs.
        :param docs: documents sent to the encoder. The docs must have `blob` with a
            shape and content as expected by the pretrained loaded model
        :param parameters: dictionary to define the `traversal_paths` and the
            `batch_size`. For example, `parameters={'traversal_paths': ['r'],
            'batch_size': 10}` will override the `self.default_traversal_paths` and
            `self.default_batch_size`.
        """
        if docs:
            document_batches_generator = docs.traverse_flat(
                traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
                filter_fn=lambda doc: doc.blob is not None
            ).batch(
                batch_size=parameters.get('batch_size', self.batch_size),
            )
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        with torch.inference_mode():
            for document_batch in document_batches_generator:
                blob_batch = np.array([d.blob for d in document_batch])
                _input = torch.from_numpy(blob_batch.astype('float32'))
                _input = _input.to(self.device)
                _feature = self.adaptiveAvgPool2d(
                    self._get_features(content=_input).detach()
                )
                _feature = _feature.cpu().numpy()
                for doc, embedding in zip(document_batch, _feature):
                    doc.embedding = embedding.squeeze()
