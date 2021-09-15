from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from jina import DocumentArray, Executor, requests
from jina_commons.batching import get_docs_batch_generator
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TimmImageEncoder(Executor):
    """
    :class:`TimmImageEncoder` encodes ``Document`` blobs of type `ndarray`
    (`float32`) and shape `H x W x 3` into `ndarray` of `D`.
    Where `D` is the Dimension of the embedding.
    Input image in Document should be in RGB format.

    If `use_default_preprocessing=False`, the expected input shape is
    `3 x H x W` with `float32` dtype.

    Internally, :class:`TimmImageEncoder` wraps the pre-trained models from
    `Timm library`.
    https://rwightman.github.io/pytorch-image-models/

    :param model_name: the name of the model. Models listed on:
        https://rwightman.github.io/pytorch-image-models/models/
    :param device: Which device the model runs on. For example 'cpu' or 'cuda'.
    :param default_traversal_paths: Defines traversal path through the docs.
        Default input is the tuple ('r',) and can be overwritten.
    :param default_batch_size: Defines the batch size for inference on the loaded Timm model.
        Default batch size is 32 and can be updated by passing an int value.
    :param use_default_preprocessing: Should the input be preprocessed with default configuration.
    :param args:  Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        device: str = 'cpu',
        default_traversal_path: Tuple[str] = ('r',),
        default_batch_size: Optional[int] = 32,
        use_default_preprocessing: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.device = device
        self.default_batch_size = default_batch_size
        self.use_default_preprocessing = use_default_preprocessing

        self.default_traversal_path = default_traversal_path

        self._model = create_model(model_name, pretrained=True, num_classes=0)
        self._model = self._model.to(device)
        self._model.eval()

        config = resolve_data_config({}, model=self._model)
        self._preprocess = create_transform(**config)
        self._preprocess.transforms.insert(0, T.ToPILImage())

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
                    images = np.stack([self._preprocess(img) for img in blob_batch])
                else:
                    images = np.stack(blob_batch)

                tensor = torch.from_numpy(images).to(self.device)
                features = self._model(tensor)
                features = features.cpu().numpy()

                for doc, embed in zip(document_batch, features):
                    doc.embedding = embed
