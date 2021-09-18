__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator
from jina_commons.encoders.image.preprocessing import (
    crop_image,
    load_image,
    move_channel_axis,
    resize_short,
)
from tensorflow.python.keras.models import load_model


class BigTransferEncoder(Executor):
    """
    :class:`BigTransferEncoder` is Big Transfer (BiT) presented by
    Google (https://github.com/google-research/big_transfer).
    Uses pretrained BiT to encode data from a ndarray with shape
    B x (Height x Width x Channels) into a ndarray of `B x D`.
    Internally, :class:`BigTransferEncoder` wraps the models from
    https://storage.googleapis.com/bit_models/.
    """

    def __init__(
        self,
        model_path: Optional[str] = 'pretrained',
        model_name: Optional[str] = 'Imagenet21k/R50x1',
        device: str = '/CPU:0',
        target_dim: Optional[Tuple[int, int, int]] = None,
        traversal_paths: Sequence[str] = ('r',),
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        :param model_path: the path of the model in the `SavedModel` format.
        :param model_name: includes ``Imagenet1k/R50x1``, ``Imagenet1k/R101x1``,
            ``Imagenet1k/R50x3``, ``Imagenet1k/R101x3``, ``Imagenet1k/R152x4``,
            ``Imagenet21k/R50x1``, ``Imagenet21k/R101x1``, ``Imagenet21k/R50x3``,
            ``Imagenet21k/R101x3``, ``Imagenet21k/R152x4``

        This encoder checks if the specified model_path exists.
        If it exists, the model in this folder is used.
        If it does not exist, the model specified in the `model_name` will be
        downloaded into this path and the downloaded model is used.

        In the end, the `model_path` should be a directory path,
        which has the following structure:

        .. highlight:: bash
         .. code-block:: bash
            .
            ├── saved_model.pb
            └── variables
                ├── variables.data-00000-of-00001
                └── variables.index
        :param device: Device ('/CPU:0', '/GPU:0', '/GPU:X')
        :param target_dim: preprocess the data image into shape of `target_dim`,
            (e.g. (256, 256, 3) ), if set to None then preoprocessing will not be conducted
        :param traversal_paths: Traversal path through the docs
        :param batch_size: Batch size to be used in the encoder model
        """
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.target_dim = target_dim
        self.logger = JinaLogger(self.__class__.__name__)
        self.default_batch_size = batch_size
        self.default_traversal_paths = traversal_paths

        if not os.path.exists(self.model_path):
            self.download_model()

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if 'GPU' in device:
            gpu_index = 0 if 'GPU:' not in device else int(device.split(':')[-1])
            if len(gpus) < gpu_index + 1:
                raise RuntimeError(f'Device {device} not found on your system!')
        self.device = tf.device(device)

        self.logger.info(f'BiT model path: {self.model_path}')
        with self.device:
            _model = load_model(self.model_path)
            self.model = _model.signatures['serving_default']
            self._get_input = tf.convert_to_tensor

    def download_model(self):
        dataset = ['Imagenet1k', 'Imagenet21k']
        _available_models = ['R50x1', 'R101x1', 'R50x3', 'R101x3', 'R152x4']
        available_models = [f'{d}/{m}' for m in _available_models for d in dataset]
        if self.model_name not in available_models:
            raise AttributeError(
                f'{self.model_name} model does not exists. '
                f'Choose one from {available_models}!'
            )

        self.logger.info(f'Starting download of {self.model_name} BiT model')
        import requests

        os.makedirs(self.model_path)
        os.makedirs((os.path.join(self.model_path, 'variables')))
        response = requests.get(
            f'https://storage.googleapis.com/bit_models/'
            f'{self.model_name}/feature_vectors/saved_model.pb'
        )
        with open(os.path.join(self.model_path, 'saved_model.pb'), 'wb') as file:
            file.write(response.content)
        response = requests.get(
            f'https://storage.googleapis.com/bit_models/'
            f'{self.model_name}/feature_vectors/variables/'
            f'variables.data-00000-of-00001'
        )
        with open(
            os.path.join(self.model_path, 'variables/variables.data-00000-of-00001'),
            'wb',
        ) as file:
            file.write(response.content)
        response = requests.get(
            f'https://storage.googleapis.com/bit_models/'
            f'{self.model_name}/feature_vectors/variables/'
            f'variables.index'
        )
        with open(
            os.path.join(self.model_path, 'variables/variables.index'), 'wb'
        ) as file:
            file.write(response.content)
        self.logger.info(f'Completed download of {self.model_name} BiT model')

    @requests
    def encode(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Encode data into a ndarray of `B x D`.
        Where `B` is the batch size and `D` is the Dimension.

        :param docs: DocumentArray containing image data as an array
        :param parameters: parameters dictionary
        """
        docs_batch_generator = get_docs_batch_generator(
            docs,
            traversal_path=parameters.get(
                'traversal_paths', self.default_traversal_paths
            ),
            batch_size=parameters.get('batch_size', self.default_batch_size),
            needs_attr='blob',
        )
        for batch in docs_batch_generator:
            if self.target_dim:
                data = np.zeros((batch.__len__(),) + self.target_dim)
            else:
                data = np.zeros((batch.__len__(),) + batch[0].blob.shape)
            for index, doc in enumerate(batch):
                if self.target_dim:
                    data[index] = self._preprocess(data[index])
                else:
                    data[index] = doc.blob
            with self.device:
                _output = self.model(self._get_input(data.astype(np.float32)))
            output = _output['output_1'].numpy()
            for index, doc in enumerate(batch):
                doc.embedding = output[index]

    def _preprocess(self, blob: 'np.ndarray'):
        img = load_image(blob)
        img_mean = (0, 0, 0)
        img_std = (1, 1, 1)
        img = resize_short(img)
        img, _, _ = crop_image(img, how='center', target_size=self.target_dim[0])
        img = np.array(img).astype('float32') / 255
        img -= img_mean
        img /= img_std
        img = move_channel_axis(img, channel_axis_to_move=-1, target_channel_axis=-1)
        return img
