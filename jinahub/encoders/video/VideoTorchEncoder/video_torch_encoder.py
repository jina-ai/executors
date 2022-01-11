__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models.video as models
from docarray import DocumentArray
from jina import Executor, requests
from torchvision import transforms

# https://github.com/pytorch/vision/blob/d391a0e992a35d7fb01e11110e2ccf8e445ad8a0/references/video_classification/transforms.py#L13


class ConvertFHWCtoFCHW(nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(0, 3, 1, 2)


class ConvertFCHWtoCFHW(nn.Module):
    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return vid.permute(1, 0, 2, 3)


class VideoTorchEncoder(Executor):
    """
    Encode `Document` content, using the models from `torchvision.models`.
    :class:`VideoTorchEncoder` encodes content from Documents containing video blobs

    Internally, :class:`VideoTorchEncoder` wraps the models from
    `torchvision.models`: https://pytorch.org/docs/stable/torchvision/models.html
    """

    def __init__(
        self,
        model_name: str = 'r3d_18',
        use_preprocessing: bool = True,
        download_progress=True,
        traversal_paths: str = 'r',
        batch_size: int = 32,
        device: str = 'cpu',
        *args,
        **kwargs
    ):
        """
        :param model_name: the name of the model.
            Supported models include ``r3d_18``, ``mc3_18``, ``r2plus1d_18``
            Default is ``r3d_18``.
        :param use_preprocessing: if True, the same preprocessing is used which got used during training
            - prevents training-serving gap. When setting `use_preprocessing=True`,
              the input `blob` must have the size of `NumFrames x Height x Width x Channel`.
        :param traversal_paths: a comma-separated string that represents the traversal path, default `r`.
        :param batch_size: fallback batch size in case there is no batch size sent in the request
            Defaults to ('r', ), i.e. root level traversal.
        :param device: device to use for encoding ['cuda', 'cpu] - if not set, the device is detected automatically
        """
        super().__init__(*args, **kwargs)
        if not device or device not in ('cpu', 'cuda'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = batch_size
        self.traversal_paths = traversal_paths
        self.model = getattr(models, model_name)(
            pretrained=True, progress=download_progress
        )
        self.model.eval().to(self.device)
        self.use_preprocessing = use_preprocessing
        if self.use_preprocessing:
            # https://github.com/pytorch/vision/blob/master/references/video_classification/train.py
            # Eval preset transformation
            mean = (0.43216, 0.394666, 0.37645)
            std = (0.22803, 0.22145, 0.216989)
            resize_size = (128, 171)
            crop_size = (112, 112)
            self.transforms = transforms.Compose(
                [
                    ConvertFHWCtoFCHW(),
                    transforms.ConvertImageDtype(torch.float32),
                    transforms.Resize(resize_size),
                    transforms.Normalize(mean=mean, std=std),
                    transforms.CenterCrop(crop_size),
                    ConvertFCHWtoCFHW(),
                ]
            )

    def _get_embeddings(self, x) -> torch.Tensor:
        embeddings = None

        def get_activation(model, model_input, output):
            nonlocal embeddings
            embeddings = output

        handle = self.model.avgpool.register_forward_hook(get_activation)
        self.model(x)
        handle.remove()
        return embeddings.flatten(1)

    @requests
    def encode(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """
        Encode all docs with images and store the encodings in the embedding attribute of the docs.

        :param docs: documents sent to the encoder. The docs must have `blob` of the shape
            `Channels x NumFrames x Height x Width` with `Height` and `Width` equals to 112,
            if no preprocessing is requested. When setting `use_preprocessing=True`,
            the input `blob` must have the size of `NumFrames x Height x Width x Channel`.
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`.
            For example, `parameters={'traversal_paths': 'r', 'batch_size': 10}` will override
            the `self.traversal_paths` and `self.batch_size`.
        """
        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        batch_size = parameters.get('batch_size', self.batch_size)

        for batch in docs.traverse_flat(traversal_paths).batch(batch_size):
            try:
                self._create_embeddings(batch)
            except RuntimeError:
                error_msg = 'Input dim not match with expected dimensionality.'
                error_msg += 'if `use_preprocessing=True` expected input is (NUM_FRAMES, H, W C),'
                error_msg += 'if `use_preprocessing=False`, expected input is (C, NUM_FRAMES, H, W).'
                raise RuntimeError(error_msg)

    def _create_embeddings(self, batch_of_documents: DocumentArray):
        with torch.inference_mode():
            if self.use_preprocessing:
                tensors = [
                    self.transforms(torch.Tensor(d.blob).to(dtype=torch.uint8))
                    for d in batch_of_documents
                ]
                tensor = torch.stack(tensors).to(self.device)
            else:
                tensor = torch.stack(
                    [torch.Tensor(d.blob) for d in batch_of_documents]
                ).to(self.device)
            embedding_batch = self._get_embeddings(tensor)
            numpy_embedding_batch = embedding_batch.cpu().numpy()
            for document, numpy_embedding in zip(
                batch_of_documents, numpy_embedding_batch
            ):
                document.embedding = numpy_embedding
