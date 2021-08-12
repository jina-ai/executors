from typing import Dict, Tuple, Optional, List

import numpy as np
from jina import Executor, DocumentArray, requests, Document
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina_commons import get_logger


class SimpleIndexer(Executor):
    """
    A simple indexer that stores all the Document data together,
    in a DocumentArrayMemmap object

    To be used as a unified indexer, combining both indexing and searching
    """

    def __init__(
        self,
        index_file_name: str,
        default_traversal_paths: Optional[List[str]] = None,
        default_top_k: int = 5,
        distance_metric: str = 'cosine',
        **kwargs,
    ):
        """
        Initializer function for the simple indexer
        :param index_file_name: The file name for the index file
        :param default_traversal_paths: The default traversal path that is used
            if no traversal path is given in the parameters of the request.
            This defaults to ['r'].
        :param default_top_k: default value for the top_k parameter
        :param distance_metric: The distance metric to be used for finding the
            most similar embeddings. Either 'euclidean' or 'cosine'.
        """
        super().__init__(**kwargs)
        self._docs = DocumentArrayMemmap(self.workspace + f'/{index_file_name}')
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.default_top_k = default_top_k
        self.metric_name = distance_metric
        if distance_metric == 'cosine':
            self.distance = _cosine
        elif distance_metric == 'euclidean':
            self.distance = _euclidean
        else:
            raise ValueError('This distance metric is not available!')
        self._flush = True
        self._docs_embeddings = None
        self.logger = get_logger(self)

    @requests(on='/index')
    def index(
        self,
        docs: Optional['DocumentArray'] = None,
        parameters: Optional[Dict] = {},
        **kwargs,
    ):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        :param parameters: the parameters dictionary
        """
        if not docs:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        self._docs.extend(flat_docs)
        self._flush = True

    @requests(on='/search')
    def search(
        self,
        docs: Optional['DocumentArray'] = None,
        parameters: Optional[Dict] = {},
        **kwargs,
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the parameters for the search"""
        if not docs:
            return
        if not self._docs:
            self.logger.warning(
                'no documents are indexed. searching empty docs. returning.'
            )
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        if not flat_docs:
            return
        top_k = int(parameters.get('top_k', self.default_top_k))
        flat_docs.match(
            self._docs,
            metric=lambda q_emb, d_emb, _: self.distance(q_emb, d_emb),
            limit=top_k,
            metric_name=self.metric_name
        )
        self._flush = False

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        if not docs:
            return
        for doc in docs:
            doc.embedding = self._docs[doc.id].embedding


def _ext_A(A):
    nA, dim = A.shape
    A_ext = np.ones((nA, dim * 3))
    A_ext[:, dim : 2 * dim] = A
    A_ext[:, 2 * dim :] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = np.ones((dim * 3, nB))
    B_ext[:dim] = (B ** 2).T
    B_ext[dim : 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A, B):
    sqdist = _ext_A(A).dot(_ext_B(B)).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A, B):
    return _ext_A(_norm(A)).dot(_ext_B(_norm(B))).clip(min=0) / 2
