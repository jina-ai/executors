__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Tuple, Dict, List

import numpy as np
from jina import Executor, requests, DocumentArray, Document

from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors


class NumpySearcher(Executor):
    def __init__(
            self,
            dump_path: str = None,
            default_top_k: int = 5,
            default_traversal_paths: List[str] = ['r'],
            metric: str = 'cosine',
            is_distance: bool = False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_traversal_paths = default_traversal_paths
        self.is_distance = is_distance
        self.metric = metric
        self.dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        self.logger = get_logger(self)
        self.default_top_k = default_top_k
        if self.dump_path is not None:
            self.logger.info(f'Importing data from {self.dump_path}')
            ids, vecs = import_vectors(self.dump_path, str(self.runtime_args.pea_id))
            self._ids = np.array(list(ids))
            self._vecs = np.array(list(vecs))
            self._ids_to_idx = {}
            self.logger.info(f'Imported {len(self._ids)} documents.')
        else:
            self.logger.warning(
                f'No dump_path provided for {self.__class__.__name__}. Use flow.rolling_update()...'
            )

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        if not hasattr(self, '_vecs') or not self._vecs.size:
            self.logger.warning('Searching an empty index')
            return

        top_k = int(parameters.get('top_k', self.default_top_k))

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        doc_embeddings = docs.traverse_flat(traversal_paths).get_attributes('embedding')

        if not docs:
            self.logger.info('No documents to search for')
            return

        if not doc_embeddings:
            self.logger.info('None of the docs have any embeddings')
            return

        doc_embeddings = np.stack(doc_embeddings)

        q_emb = _ext_A(_norm(doc_embeddings))
        d_emb = _ext_B(_norm(self._vecs))
        if self.metric == 'cosine':
            dists = _cosine(q_emb, d_emb)
        elif self.metric == 'euclidean':
            dists = _euclidean(q_emb, d_emb)
        else:
            self.logger.error(f'Metric {self.metric} not supported.')
        positions, dist = self._get_sorted_top_k(dists, top_k)
        for _q, _positions, _dists in zip(docs, positions, dist):
            for position, dist in zip(_positions, _dists):
                d = Document(id=self._ids[position], embedding=self._vecs[position])
                if self.is_distance:
                    d.scores[self.metric] = dist
                else:
                    if self.metric == 'cosine':
                        d.scores[self.metric] = 1 - dist
                    elif self.metric == 'euclidean':
                        d.scores[self.metric] = 1 / (1 + dist)
                _q.matches.append(d)

    @staticmethod
    def _get_sorted_top_k(
            dist: 'np.array', top_k: int
    ) -> Tuple['np.ndarray', 'np.ndarray']:
        if top_k >= dist.shape[1]:
            idx = dist.argsort(axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx, axis=1)
        else:
            idx_ps = dist.argpartition(kth=top_k, axis=1)[:, :top_k]
            dist = np.take_along_axis(dist, idx_ps, axis=1)
            idx_fs = dist.argsort(axis=1)
            idx = np.take_along_axis(idx_ps, idx_fs, axis=1)
            dist = np.take_along_axis(dist, idx_fs, axis=1)

        return idx, dist


def _get_ones(x, y):
    return np.ones((x, y))


def _ext_A(A):
    nA, dim = A.shape
    A_ext = _get_ones(nA, dim * 3)
    A_ext[:, dim: 2 * dim] = A
    A_ext[:, 2 * dim:] = A ** 2
    return A_ext


def _ext_B(B):
    nB, dim = B.shape
    B_ext = _get_ones(dim * 3, nB)
    B_ext[:dim] = (B ** 2).T
    B_ext[dim: 2 * dim] = -2.0 * B.T
    del B
    return B_ext


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2
