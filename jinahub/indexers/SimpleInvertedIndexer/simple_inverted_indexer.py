import os
from typing import List, Optional, Dict
from collections import defaultdict
import pickle

import scipy
import numpy as np

from jina import Executor, DocumentArray, requests, Document
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina_commons import get_logger


class InvertedIndex:

    def __init__(self):
        self.inverted_index = defaultdict(set)
        self.document_frequencies = defaultdict(int)
        self.document_sparse_vectors = {}
        self.idfs = {}

    def cache_idfs(self):
        for term_idx in self.document_frequencies.keys():
            num = len(self.document_sparse_vectors.keys())
            den = 1 + self.document_frequencies[term_idx]
            self.idfs[term_idx] = np.log(num / den)

    def add(self, document_id, document_vector):
        def _add(term_id, document_id, value):
            self.inverted_index[term_id].add(document_id)
            self.document_frequencies[term_id] += value

        for term_id, term_value in zip(document_vector.indices, document_vector.data):
            _add(term_id, document_id, term_value)
        self.document_sparse_vectors[document_id] = document_vector

    def get_candidates(self, term_index):
        return self.inverted_index[term_index]

    def match(self, query, top_k, return_scores=False):
        candidates = set()

        for term_index in query.indices:
            candidates.update(self.get_candidates(term_index))

        scores = []
        candidates = list(candidates)
        for candidate in candidates:
            scores.append(self._relevance(query, candidate))

        results = sorted(zip(scores, candidates), reverse=True)
        if top_k:
            if return_scores:
                return results[: top_k]
            else:
                return [element for _, element in results[: top_k]]
        else:
            if return_scores:
                return results
            else:
                return [element for _, element in results]

    def _relevance(self, query_vec, candidate):
        candidate_vector = self.document_sparse_vectors[candidate]
        candidate_dense = np.array(candidate_vector.todense())[0]
        number_words = len(candidate_vector.indices)
        prod = 1
        for term_index in query_vec.indices:
            tf = self._tf(candidate_dense, term_index, number_words)
            idf = self._idf(term_index)
            prod = prod * tf * idf
        return prod

    def _tf(self, candidate_dense, term_idx, number_words):
        return candidate_dense[term_idx] / number_words

    def _idf(self, term_idx):
        return self.idfs.get(term_idx, 0)


class SimpleInvertedIndexer(Executor):
    """
    A simple inverted indexer that stores Document Lists in buckets given their sparse embedding input
    """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Delete entries from the index by id

        :param docs: the documents to delete
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        delete_docs_ids = docs.traverse_flat(traversal_paths).get_attributes('id')
        for idx in delete_docs_ids:
            if idx in self._docs:
                del self._docs[idx]

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Update doc with the same id

        :param docs: the documents to update
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        for doc in flat_docs:
            if doc.id is not None:
                self._docs[doc.id] = doc
            else:
                self._docs.append(doc)

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        if not docs:
            return
        for doc in docs:
            if doc.id in self._docs:
                doc.embedding = self._docs[doc.id].embedding
            else:
                self.logger.warning(f'Document {doc.id} not found in index')
