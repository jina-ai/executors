from typing import Optional, Tuple, Dict, Iterable

import numpy as np
from jina import Executor, DocumentArray, requests
import torch
from jina_commons.batching import get_docs_batch_generator

from pointnet import model

class PointNetEncoder(Executor):
    """PointNetEncoder embeds point cloud 3 models into vectors"""
    """
    :class:`PointNetEncoder` encodes 3D point cloud blobs of type `ndarray` (`float32`) and shape
    `N x 6` into `ndarray` of shape `(1024,)`.
    Where `N` is the number of points in the point set.
    
    

    :param model_path: the path of the pointnet semantic segmenter model.
    :param device: Which device the model runs on. Can be 'cpu' or 'cuda'
    :param default_traversal_paths: Used in the encode method an defines traversal on the received `DocumentArray`
    :param default_batch_size: Defines the batch size for inference on the loaded PyTorch model.
    :param args:  Additional positional arguments
    :param kwargs: Additional keyword arguments
    """

    def __init__(
            self,
            model_path: str = 'pointnet/log/sem_seg/pointnet_sem_seg/checkpoints/best_model.pth',
            device: Optional[str] = None,
            default_traversal_path: Tuple = ('r',),
            default_batch_size: Optional[int] = 32,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.default_batch_size = default_batch_size

        self.default_traversal_path = default_traversal_path
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.eval()
        model.feat.global_feat = True
        self.model = model

    @requests
    def encode(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Encode 3D point cloud data into a ndarray of shape `(1024,)` as dimension, and fill the embedding of each
        Document.

        :param docs: DocumentArray containing 3D point cloud blobs
        :param parameters: dictionary to define the `traversal_paths` and the `batch_size`. For example,
               `parameters={'traversal_paths': ['r'], 'batch_size': 10}`.
        :param kwargs: Additional key value arguments.
        """
        if docs:
            docs_batch_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_path),
                batch_size=parameters.get('batch_size', self.default_batch_size),
                needs_attr='blob'
            )
            self._compute_embeddings(docs_batch_generator)

    def _compute_embeddings(self, docs_batch_generator: Iterable) -> None:
        with torch.no_grad():
            for document_batch in docs_batch_generator:
                blob_batch = np.stack([d.blob for d in document_batch])
                torch_data = torch.Tensor(blob_batch)
                torch_data = torch_data.float().cuda()
                torch_data = torch_data.transpose(2, 1)
                features, _, _ = self.model.feat(torch_data)

                for doc, embed in zip(document_batch, features):
                    doc.embedding = embed
