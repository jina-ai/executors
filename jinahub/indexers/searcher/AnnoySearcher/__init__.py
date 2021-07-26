__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, List, Union, Dict

import numpy as np
from annoy import AnnoyIndex
from jina import Executor, requests, DocumentArray, Document

from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors


class AnnoySearcher(Executor):
    """Annoy powered vector indexer

    For more information about the Annoy supported parameters, please consult:
        - https://github.com/spotify/annoy

    .. note::
        Annoy package dependency is only required at the query time.
    """

    def __init__(
            self,
            default_top_k: int = 10,
            metric: str = 'euclidean',
            num_trees: int = 10,
            dump_path: Optional[str] = None,
            default_traversal_paths: List[str] = ['r'],
            is_distance: bool = False,
            **kwargs,
    ):
        """
        Initialize an AnnoyIndexer

        :param default_top_k: get tok k vectors
        :param metric: Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"
        :param num_trees: builds a forest of n_trees trees. More trees gives higher precision when querying.
        :param dump_path: the path to load ids and vecs
        :param traverse_path: traverse path on docs, e.g. ['r'], ['c']
        :param is_distance: Boolean flag that describes if distance metric need to be reinterpreted as similarities.
        :param args:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.default_top_k = default_top_k
        self.metric = metric
        self.num_trees = num_trees
        self.default_traversal_paths = default_traversal_paths
        self.is_distance = is_distance
        self.logger = get_logger(self)
        dump_path = dump_path or kwargs.get('runtime_args', {}).get('dump_path', None)
        if dump_path is not None:
            self.logger.info('Start building "AnnoyIndexer" from dump data')
            ids, vecs = import_vectors(dump_path, str(self.metas.pea_id))
            self._ids = np.array(list(ids))
            self._vecs = np.array(list(vecs))
            num_dim = self._vecs.shape[1]
            self._indexer = AnnoyIndex(num_dim, self.metric)
            self._doc_id_to_offset = {}
            self._load_index(self._ids, self._vecs)
        else:
            self.logger.warning(
                'No data loaded in "AnnoyIndexer". Use .rolling_update() to re-initialize it...'
            )

    def _load_index(self, ids, vecs):
        for idx, v in enumerate(vecs):
            self._indexer.add_item(idx, v.astype(np.float32))
            self._doc_id_to_offset[ids[idx]] = idx
        self._indexer.build(self.num_trees)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if not hasattr(self, '_indexer'):
            self.logger.warning('Querying against an empty index')
            return

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._indexer.get_nns_by_vector(
                doc.embedding, self.default_top_k, include_distances=True
            )
            for idx, dist in zip(indices, dists):
                match = Document(id=self._ids[idx], embedding=self._vecs[idx])
                if self.is_distance:
                    if self.metric == 'dot':
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = dist
                else:
                    if self.metric == 'dot':
                        match.scores[self.metric] = dist
                    elif self.metric == 'angular' or self.metric == 'hamming':
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = 1 / (1 + dist)
                doc.matches.append(match)

    @requests(on='/fill_embedding')
    def fill_embedding(self, query_da: DocumentArray, **kwargs):
        for doc in query_da:
            doc.embedding = np.array(
                self._indexer.get_item_vector(int(self._doc_id_to_offset[str(doc.id)]))
            )
