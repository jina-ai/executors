__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from jina_commons.batching import get_docs_batch_generator

from .models import EmbeddingModelWrapper


class ImageTorchEncoder(Executor):
    """
    :class:`ImageTorchEncoder` encodes ``Document`` blobs of type `ndarray` (`float32`) and shape
    `B x H x W x C` into `ndarray` of `B x D`.
    Where `B` is the batch size and `D` is the Dimension of the embedding.
    If `use_default_preprocessing=False`, the expected input shape is `B x C x H x W` with `float32` dtype.

    Internally, :class:`ImageTorchEncoder` wraps the models from
    `torchvision.models`.
    https://pytorch.org/vision/stable/models.html

    :param model_name: the name of the model. Some of the models:
        ``alexnet``, `squeezenet1_0``,  ``vgg16``,
        ``densenet161``, ``inception_v3``, ``googlenet``,
        ``shufflenet_v2_x1_0``, ``mobilenet_v2``,
        ``mnasnet1_0``, ``resnet18``. See full list above.
    :param device: Which device the model runs on. Can be 'cpu' or 'cuda'
    :param default_traversal_paths: Used in the encode method an defines traversal on the received `DocumentArray`
    :param default_batch_size: Defines the batch size for inference on the loaded PyTorch model.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        device: str = 'cpu',
        default_traversal_path: Tuple = ('r',),
        default_batch_size: Optional[int] = 32,
        use_default_preprocessing: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        if (device not in ['cpu', 'cuda']) and (not device.startswith('cuda:')):
            self.logger.error(
                f'Torch device {device} not supported. Must be cpu or cuda!'
            )
            raise RuntimeError(
                f'Torch device {device} not supported. Must be cpu or cuda!'
            )

        if device.startswith('cuda') and not torch.cuda.is_available():
            self.logger.warning(
                'You tried to use GPU but torch did not detect your '
                'GPU correctly. Defaulting to CPU. Check your CUDA installation!'
            )
            device = 'cpu'
        self.device = device
        self.default_batch_size = default_batch_size
        self.use_default_preprocessing = use_default_preprocessing

        self.default_traversal_path = default_traversal_path

        # axis 0 is the batch
        self._default_channel_axis = 1

        self.model_wrapper = EmbeddingModelWrapper(model_name, device=self.device)

        self._preprocess = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode image data into a ndarray of `D` as dimension, and fill the embedding of each Document.

        :param docs: DocumentArray containing images
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
               `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """
        if docs:
            docs_batch_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get(
                    'traversal_paths', self.default_traversal_path
                ),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='blob',
            )
            self._compute_embeddings(docs_batch_generator)

    def _compute_embeddings(self, docs_batch_generator: Iterable) -> None:
        with torch.no_grad():
            for document_batch in docs_batch_generator:
                blob_batch = [d.blob for d in document_batch]
                if self.use_default_preprocessing:
                    images = np.stack(self._preprocess_image(blob_batch))
                else:
                    images = np.stack(blob_batch)
                features = self.model_wrapper.compute_embeddings(images)

                for doc, embed in zip(document_batch, features):
                    doc.embedding = embed

    def _preprocess_image(self, images: List[np.array]) -> List[np.ndarray]:
        return [self._preprocess(img) for img in images]
