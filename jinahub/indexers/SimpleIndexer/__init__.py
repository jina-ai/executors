from collections import defaultdict
from typing import Dict, Tuple, Optional, List

from pathlib import Path
import numpy as np
from jina import Executor, DocumentArray, requests, Document
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina.excepts import ValidationError, NotSupportedError


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
        aggregate: bool = False,
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
        self.aggregate = aggregate
        self._docs = DocumentArrayMemmap(self.workspace + f'/{index_file_name}')

        if self.aggregate:
            top_level_docs_dir_path = Path(self.workspace) / 'top_level'
            top_level_docs_dir_path.mkdir(exist_ok=True)
            self._top_level_docs = DocumentArrayMemmap(str(top_level_docs_dir_path) + f'/{index_file_name}')

        self.default_traversal_paths = default_traversal_paths or ['r']
        self.default_top_k = default_top_k
        self.distance_metric = distance_metric
        if distance_metric == 'cosine':
            self.distance = _cosine
        elif distance_metric == 'euclidean':
            self.distance = _euclidean
        else:
            raise ValueError('This distance metric is not available!')
        self._flush = True
        self._docs_embeddings = None

    @property
    def index_embeddings(self):
        if self._flush:
            self._docs_embeddings = np.stack(self._docs.get_attributes('embedding'))
            self._flush = False
        return self._docs_embeddings

    @requests(on='/index')
    def index(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        """All Documents to the DocumentArray
        :param docs: the docs to add
        :param parameters: the parameters dictionary
        """
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)

        if self.aggregate and traversal_paths != self.default_traversal_paths:
            raise ValidationError(
                    'if aggregation is enabled, the traversal path in search flow '
                    'must match the one used in index flow. Hence, to enforce this rule, '
                    'please provide default_traversal_paths as an argument when initializing '
                    'and adding this indexer to the flow and ensure the traversal_paths '
                    'provided in parameters is the same as default_traversal_paths. '
                    f'traversal_paths: {traversal_paths} != '
                    f'default_traversal_paths: {self.default_traversal_paths}')

        if self.aggregate:
            self._top_level_docs.extend(docs)

        flat_docs = docs.traverse_flat(traversal_paths)
        self._docs.extend(flat_docs)
        self._flush = True

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: the parameters for the search"""
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)

        if self.aggregate and traversal_paths != self.default_traversal_paths:
            raise ValidationError(
                    'if aggregation is enabled, the traversal path in search flow '
                    'must match the one used in index flow. Hence, to enforce this rule, '
                    'please provide default_traversal_paths as an argument when initializing '
                    'and adding this indexer to the flow and ensure the traversal_paths '
                    'provided in parameters is the same as default_traversal_paths. '
                    f'traversal_paths: {traversal_paths} != '
                    f'default_traversal_paths: {self.default_traversal_paths}')

        top_k = parameters.get('top_k', self.default_top_k)
        flat_docs = docs.traverse_flat(traversal_paths)
        a = np.stack(flat_docs.get_attributes('embedding'))
        b = self.index_embeddings
        q_emb = _ext_A(_norm(a))
        d_emb = _ext_B(_norm(b))
        dists = self.distance(q_emb, d_emb)
        idx, dist = self._get_sorted_top_k(dists, int(top_k))
        for _q, _ids, _dists in zip(flat_docs, idx, dist):
            for _id, _dist in zip(_ids, _dists):
                d = Document(self._docs[int(_id)], copy=True)
                d.scores[self.distance_metric] = 1 - _dist if not self.aggregate else _dist
                _q.matches.append(d)

        if self.aggregate:
            self._aggregate(docs, flat_docs)

    def _aggregate(
        self,
        queries: 'DocumentArray',
        queries_descendants: 'DocumentArray',
    ):
        def reverse_traverse(doc, lvls):
            # TODO: modify this to support multilevels
            if lvls > 1:
                raise NotSupportedError(
                    'currently we only support reverse traverse on chunk with granularity=1')
            return doc.parent_id

        if len(self.default_traversal_paths) > 1 or 'm' in self.default_traversal_paths[0]:
            raise ValidationError('if traversal_paths is given, '
                'expects a single traversal path which traverses itself or its chunk descendants only')

        # if traversal path is 'r', no aggregation is needed, return
        if not self.default_traversal_paths or self.default_traversal_paths == ['r']: return

        ancestor_ids = defaultdict(list)

        levels = len(self.default_traversal_paths[0])
        print('here: ', levels)

        for descendant in queries_descendants:
            top_level_query_id = reverse_traverse(descendant, levels)
            for match in descendant.matches:
                ancestor_ids[(top_level_query_id, reverse_traverse(match, levels))].append(match.scores[self.distance_metric].value)

        for (query_id, index_id), scores in ancestor_ids.items():
            match = self._top_level_docs[index_id]
            match.scores[self.distance_metric] = 1 - np.mean(scores)
            queries[query_id].matches.append(match)

        for query in queries:
            query.matches.sort(key=lambda match: match.scores[self.distance_metric].value, reverse=True)

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

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
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


def _euclidean(A_ext, B_ext):
    sqdist = A_ext.dot(B_ext).clip(min=0)
    return np.sqrt(sqdist)


def _norm(A):
    return A / np.linalg.norm(A, ord=2, axis=1, keepdims=True)


def _cosine(A_norm_ext, B_norm_ext):
    return A_norm_ext.dot(B_norm_ext).clip(min=0) / 2
