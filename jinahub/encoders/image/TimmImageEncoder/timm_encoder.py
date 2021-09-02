from typing import Dict, Iterable, List, Optional, Tuple

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
    (`float32`) and shape `H x W x C` into `ndarray` of `D`.
    Where `D` is the Dimension of the embedding.

    If `use_default_preprocessing=False`, the expected input shape is
    `C x H x W` with `float32` dtype.

    Internally, :class:`TimmImageEncoder` wraps the pre-trained models from
    `Timm library`.
    https://rwightman.github.io/pytorch-image-models/

    :param model_name: the name of the model. Models listed on:
        https://rwightman.github.io/pytorch-image-models/models/
    :param device: Which device the model runs on. Can be 'cpu' or 'cuda'.
    :param default_traversal_paths: Used in the encode method an defines traversal on the received `DocumentArray`.
    :param default_batch_size: Defines the batch size for inference on the loaded PyTorch model.
    :param use_default_preprocessing: Should the input be preprocessed with default configuration.
    :param args:  Additional positional arguments.
    :param kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        device: str = "cpu",
        default_traversal_path: Tuple = ("r",),
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

        # axis 0 is the batch
        self._default_channel_axis = 1

        self._model = create_model(model_name, pretrained=True, num_classes=0)
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
                    "traversal_paths", self.default_traversal_path
                ),
                batch_size=parameters.get("batch_size", self.default_batch_size),
                needs_attr="blob",
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

                tensor = torch.from_numpy(images).to(self.device)
                features = self._model(tensor)
                features = features.detach().cpu().numpy()

                for doc, embed in zip(document_batch, features):
                    doc.embedding = embed

    def _preprocess_image(self, images: List[np.array]) -> List[np.ndarray]:
        return [self._preprocess(img) for img in images]
