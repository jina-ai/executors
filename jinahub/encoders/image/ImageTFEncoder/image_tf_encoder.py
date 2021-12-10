__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable, Optional

import numpy as np
import tensorflow as tf
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class ImageTFEncoder(Executor):
    """
    :class:`ImageTFEncoder` encodes ``Document`` content from a ndarray,
    potentially B x (Height x Width x Channel) into a ndarray of `B x D`.

    Where `B` is the batch size and `D` is the Dimension.

    The :class:`ImageTFEncoder` wraps the models from
    `tensorflow.keras.applications`.
    <https://www.tensorflow.org/api_docs/python/tf/keras/applications>`_
    """

    def __init__(
        self,
        model_name: str = 'MobileNetV2',
        img_shape: int = 336,
        pool_strategy: str = 'max',
        traversal_paths: Iterable[str] = ('r',),
        batch_size: int = 32,
        device: str = '/CPU:0',
        *args,
        **kwargs,
    ):
        """
        :param model_name: the name of the model. Supported models include
            ``DenseNet121``, ``DenseNet169``, ``DenseNet201``,
            ``InceptionResNetV2``, ``InceptionV3``, ``MobileNet``,
            ``MobileNetV2``, ``NASNetLarge``, ``NASNetMobile``,
            ``ResNet101``, ``ResNet152``, ``ResNet50``, ``ResNet101V2``,
            ``ResNet152V2``, ``ResNet50V2``, ``VGG16``, ``VGG19``,
            ``Xception`` and etc. A full list can be find at
            <https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions>`_
        :param img_shape: The shape of the image to be encoded.
        :param pool_strategy: the pooling strategy. Options are:
            - `None`: Means that the output of the model will be the 4D tensor
                output of the last convolutional block.
            - `avg`: Means that global average pooling will be applied to the
                output of the last convolutional block, and thus the output of
                the model will be a 2D tensor.
            - `max`: Means that global max pooling will be applied.
        :param traversal_paths: traversal path of the Documents, (e.g. 'r', 'c')
        :param batch_size: size of each batch
        :param device: Device ('/CPU:0', '/GPU:0', '/GPU:X')
        """
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.pool_strategy = pool_strategy
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.logger = JinaLogger(self.__class__.__name__)

        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        if 'GPU' in device:
            gpu_index = 0 if 'GPU:' not in device else int(device.split(':')[-1])
            if len(gpus) < gpu_index + 1:
                raise RuntimeError(f'Device {device} not found on your system!')
        self.device = tf.device(device)

        with self.device:
            model = getattr(tf.keras.applications, self.model_name)(
                input_shape=(self.img_shape, self.img_shape, 3),
                include_top=False,
                pooling=self.pool_strategy,
                weights='imagenet',
            )
            model.trainable = False
            self.model = model

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """
        Encode document 'blob' into one-dimentional vector and store it into `embedding`.

        :param docs: `DocumentArray` containing `Document`s with the `blob` of image data
            in the size of `Height x Width x 3`. The `dtype` of the `blob` must be `float32`.
            The `blob` must be preprocessed to `[0, 1]`. Check out the preprocessed module of
            different models at https://www.tensorflow.org/api_docs/python/tf/keras/applications#functions
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`.
            For example, `parameters={'traversal_paths': ['r'], 'batch_size': 10}` will
            override the `self.traversal_paths` and `self.batch_size`.
        """
        if docs:
            document_batches_generator = docs.traverse_flat(
                traversal_paths=parameters.get('traversal_paths', self.traversal_paths),
                filter_fn=lambda doc: doc.blob is not None,
            ).batch(
                batch_size=parameters.get('batch_size', self.batch_size),
            )
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            blob_batch = np.stack([d.blob for d in document_batch])
            with self.device:
                embedding_batch = self.model(blob_batch)
            for document, embedding in zip(document_batch, embedding_batch):
                document.embedding = np.array(embedding)
