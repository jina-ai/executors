from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from jina import DocumentArray, Executor, requests
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class TimmImageEncoder(Executor):
    """
    TimmImageEncoder encodes Document blobs of type `ndarray` (`float32`) and shape
    `H x W x 3` into `d`-dimensional embedding. The input image in Document should be
    in RGB format.

    If `use_default_preprocessing=False`, the expected input shape is
    `3 x H x W` with `float32` dtype.

    Internally, :class:`TimmImageEncoder` wraps the pre-trained models from
    [Timm library](https://rwightman.github.io/pytorch-image-models/).
    """

    def __init__(
        self,
        model_name: str = 'resnet18',
        device: str = 'cpu',
        traversal_path: Tuple[str] = ('r',),
        batch_size: Optional[int] = 32,
        use_default_preprocessing: bool = True,
        *args,
        **kwargs
    ):
        """
        :param model_name: the name of the model. Models listed on:
            https://rwightman.github.io/pytorch-image-models/models/
        :param device: Which device the model runs on. For example 'cpu' or 'cuda'.
        :param traversal_paths: Defines traversal path through the docs. It can be
            overridden via request params.
        :param batch_size: Defines the batch size for inference on the loaded Timm model.
            It can be overridden via request params.
        :param use_default_preprocessing: If the input should be preprocessed with
            default configuration. If `False`, inputs are expected to be pre-processed.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.device = device
        self.batch_size = batch_size
        self.use_default_preprocessing = use_default_preprocessing

        self.traversal_path = traversal_path

        self._model = create_model(model_name, pretrained=True, num_classes=0)
        self._model = self._model.to(device)
        self._model.eval()

        config = resolve_data_config({}, model=self._model)
        self._preprocess = create_transform(**config)
        self._preprocess.transforms.insert(0, T.ToPILImage())

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode image data from the `blob` attribute of Documents into a ndarray of
        `D` as dimension, and fill the embedding of each Document.

        :param docs: DocumentArray containing images
        :param parameters: dictionary with additional request parameters. Possible
            values are `traversal_paths` and the `batch_size`. For example,
               `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional keyword arguments.
        """
        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_path)
        batch_size = parameters.get('batch_size', self.batch_size)
        docs_batch_generator = docs.batch(
            traversal_path=traversal_paths,
            batch_size=batch_size,
            requires_attr='blob',
        )

        for document_batch in docs_batch_generator:
            blob_batch = [d.blob for d in document_batch]
            if self.use_default_preprocessing:
                images = np.stack([self._preprocess(img) for img in blob_batch])
            else:
                images = np.stack(blob_batch)

            with torch.inference_mode():
                tensor = torch.from_numpy(images).to(self.device)
                features = self._model(tensor)
                features = features.cpu().numpy()

            for doc, embed in zip(document_batch, features):
                doc.embedding = embed
