import os
import operator
from typing import Optional, Dict, Tuple, List
from collections import defaultdict
import pickle
from scipy.sparse import csr_matrix

import numpy as np

from jina import Executor, DocumentArray, requests, Document
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina_commons import get_logger


class InvertedIndex:

    def __init__(self, relevance_score: str = 'bm25', **kwargs):
        self.inverted_index = defaultdict(set)
        self.document_frequencies = defaultdict(int)
        self.document_sparse_vectors = {}
        self.idfs = {}
        self.relevance_score = relevance_score
        self.num_docs = 0
        self._lengths = 0
        self._k = kwargs.get('k', 1.2)
        self._b = kwargs.get('b', 0.75)
        assert self.relevance_score in {'bm25', 'tfidf'}

    def add(self, document_id: str, document_vector: csr_matrix):
        def _add(term_id, doc_id, value):
            self.inverted_index[term_id].add(doc_id)
            self.document_frequencies[term_id] += value

        for term_id, term_value in zip(document_vector.indices, document_vector.data):
            _add(term_id, document_id, term_value)
        self.document_sparse_vectors[document_id] = document_vector
        self._lengths += len(document_vector.indices)
        self.num_docs += 1

    def _retrieve_match_candidates(self, query: csr_matrix):
        candidates = set()

        for term_index in query.indices:
            candidates.update(self.inverted_index[term_index])
        return candidates

    def match(self, query: csr_matrix, top_k: Optional[int], return_scores=False):
        candidates = self._retrieve_match_candidates(query)

        candidates = list(candidates)
        scores = [self._relevance(query, candidate) for candidate in candidates]

        if top_k:
            scores_arr = np.array(scores)
            top_indices = np.argpartition(-1 * scores_arr, top_k)
            top_candidates = operator.itemgetter(*top_indices.tolist())(candidates)
            top_scores = np.take_along_axis(scores_arr, top_indices, axis=0).tolist()
            if return_scores:
                return zip(top_scores, top_candidates)
            else:
                return top_candidates
        else:
            results = sorted(zip(scores, candidates), reverse=True)

            if return_scores:
                return results
            else:
                return [element for _, element in results]

    def _tfidf_score(self, query: csr_matrix, candidate: csr_matrix):
        score = 0
        candidate_dense = np.array(candidate.todense())[0]
        number_words = len(candidate.indices)
        for term_index in query.indices:
            tf = self._tf(candidate_dense, term_index, number_words)
            idf = self._idf(term_index)
            score += tf * idf
        return score

    def _bm25_score(self, query: csr_matrix, candidate: csr_matrix):
        score = 0
        candidate_dense = np.array(candidate.todense())[0]
        avg_length = self._avg_length
        query_length = len(query.indices)
        for term_index in query.indices:
            f = candidate_dense[term_index]
            idf = self._idf(term_index)
            num = f * (self._k + 1)
            den = (f + self._k * (1 - self._b + (self._b * (avg_length / query_length))))
            bm25 = idf * num / den
            score += bm25
        return score

    def _relevance(self, query: csr_matrix, candidate_id: str):
        candidate = self.document_sparse_vectors[candidate_id]

        if self.relevance_score == 'tfidf':
            return self._tfidf_score(query, candidate)
        else:
            return self._bm25_score(query, candidate)

    def _tf(self, candidate_dense: 'np.ndarray', term_idx: int, number_words: int):
        return candidate_dense[term_idx] / number_words

    def _idf(self, term_idx: int):
        num = self.num_docs
        den = 1 + self.document_frequencies[term_idx]
        return np.log(num / den)

    @property
    def _avg_length(self):
        return self._lengths / len(self.document_sparse_vectors.keys())


class SimpleInvertedIndexer(Executor):
    """
    A simple inverted indexer that stores Document Lists in buckets given their sparse embedding input
    """

    @staticmethod
    def load_from_file(base_path: str):
        inverted_index_path = os.path.join(base_path, 'inverted_index.pickle')
        with open(inverted_index_path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def store_to_file(base_path: str, inverted_idx: 'InvertedIndex'):
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        inverted_index_path = os.path.join(base_path, 'inverted_index.pickle')
        with open(inverted_index_path, 'wb') as f:
            pickle.dump(inverted_idx, f)

    def __init__(
            self,
            inverted_index_file_name: str,
            pretrained_count_vectorizer_path: str,
            relevance_score: str = 'bm25',
            default_traversal_paths: Tuple[str] = ('r',),
            default_top_k: int = 2,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        with open(pretrained_count_vectorizer_path, 'rb') as fp:
            self.vectorizer = pickle.load(fp)
        self.inverted_index_file_name = inverted_index_file_name
        self.default_traversal_paths = default_traversal_paths
        self.default_top_k = default_top_k
        self.inverted_index = InvertedIndex(relevance_score=relevance_score, **kwargs)
        self.inverted_index_full_path = os.path.join(self.workspace, inverted_index_file_name)
        self._docs = DocumentArrayMemmap(self.workspace + f'/index_file_name')

        if os.path.exists(self.inverted_index_full_path):
            self.inverted_index = self.load_from_file(self.inverted_index_full_path)
        self.logger = get_logger(self)

    @requests(on='/index')
    def index(
            self,
            docs: Optional['DocumentArray'] = None,
            parameters: Optional[Dict] = {},
            **kwargs,
    ):
        """Add documents to the inverted index
        :param docs: the docs to add
        :param parameters: the parameters dictionary
        """
        if not docs:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        flat_docs = docs.traverse_flat(traversal_paths)
        texts = flat_docs.get_attributes('text')
        embeddings = self.vectorizer.transform(texts).toarray()
        for doc, dense_embedding in zip(flat_docs, embeddings):
            sparse_embedding = csr_matrix(dense_embedding)
            self.inverted_index.add(doc.id, sparse_embedding)
            self._docs.append(doc)

    @requests(on='/search')
    def search(
            self,
            docs: Optional['DocumentArray'] = None,
            parameters: Optional[Dict] = {},
            **kwargs,
    ):
        """Retrieve results from the inverted index

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
        texts = flat_docs.get_attributes('text')
        embeddings = self.vectorizer.transform(texts).toarray()
        top_k = int(parameters.get('top_k', self.default_top_k))
        for doc, embedding in zip(flat_docs, embeddings):
            sparse_embedding = csr_matrix(embedding)
            scores_matches = self.inverted_index.match(sparse_embedding, top_k, return_scores=True)
            for score, match_id in scores_matches:
                doc.matches.append(self._docs[match_id])
                doc.matches[-1].scores[self.inverted_index.relevance_score] = score

    @requests(on='/dump')
    def dump(
        self,
        **kwargs,
    ):
        self._dump()

    def _dump(self):
        self.store_to_file(self.inverted_index_full_path, self.inverted_index)

    def close(self):
        self._dump()
        super().close()
