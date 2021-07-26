__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, List, Dict

import hnswlib
import numpy as np
from jina import Executor, requests, DocumentArray, Document
from jina_commons import get_logger
from jina_commons.indexers.dump import import_vectors


class HnswlibSearcher(Executor):
    """Hnswlib powered vector indexer

    For more information about the Hnswlib supported parameters, please consult:
        - https://github.com/nmslib/hnswlib

    .. note::
        Hnswlib package dependency is only required at the query time.
    """

    def __init__(
            self,
            default_top_k: int = 10,
            metric: str = 'cosine',
            dump_path: Optional[str] = None,
            default_traversal_paths: Optional[List[str]] = None,
            is_distance: bool = False,
            ef_construction: int = 400,
            ef_query: int = 50,
            max_connection: int = 64,
            *args,
            **kwargs,
    ):
        """
        Initialize an HnswlibSearcher

        :param default_top_k: get tok k vectors
        :param distance: distance can be 'l2', 'ip', or 'cosine'
        :param dump_path: the path to load ids and vecs
        :param traverse_path: traverse path on docs, e.g. ['r'], ['c']
        :param reverse_score: True if add reversed distance as the `similarity` for match score, else return `distance` as score for match score.
        :param ef_construction: defines a construction time/accuracy trade-off
        :param ef_query:  sets the query time accuracy/speed trade-off
        :param max_connection: defines tha maximum number of outgoing connections in the graph
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.default_top_k = default_top_k
        self.metric = metric
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.is_distance = is_distance
        self.ef_construction = ef_construction
        self.ef_query = ef_query
        self.max_connection = max_connection
        self.logger = get_logger(self)
        dump_path = dump_path or kwargs.get('runtime_args', {}).get('dump_path', None)
        if dump_path is not None:
            self.logger.info('Start building "HnswlibSearcher" from dump data')
            ids, vecs = import_vectors(dump_path, str(self.metas.pea_id))
            self._ids = np.array(list(ids))
            self._vecs = np.array(list(vecs))
            num_dim = self._vecs.shape[1]
            self._indexer = hnswlib.Index(space=self.metric, dim=num_dim)
            self._indexer.init_index(max_elements=len(self._vecs), ef_construction=self.ef_construction,
                                     M=self.max_connection)

            self._doc_id_to_offset = {}
            self._load_index(self._ids, self._vecs)
        else:
            self.logger.warning(
                'No data loaded in "HnswlibSearcher". Use .rolling_update() to re-initialize it...'
            )

    def _load_index(self, ids, vecs):
        for idx, v in enumerate(vecs):
            self._indexer.add_items(v.astype(np.float32), idx)
            self._doc_id_to_offset[ids[idx]] = idx
        self._indexer.set_ef(self.ef_query)

    @requests(on='/search')
    def search(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        if docs is None:
            return
        if not hasattr(self, '_indexer'):
            self.logger.warning('Querying against an empty index')
            return
        top_k = parameters.get('top_k', self.default_top_k)
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        for doc in docs.traverse_flat(traversal_paths):
            indices, dists = self._indexer.knn_query(doc.embedding, k=top_k)
            for idx, dist in zip(indices[0], dists[0]):
                match = Document(id=self._ids[idx], embedding=self._vecs[idx])
                if self.is_distance:
                    match.scores[self.metric] = dist
                else:
                    if self.metric == 'cosine' or self.metric == 'ip':
                        match.scores[self.metric] = 1 - dist
                    else:
                        match.scores[self.metric] = 1 / (1 + dist)

                doc.matches.append(match)

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        if docs is None:
            return
        for doc in docs:
            doc.embedding = np.array(
                self._indexer.get_items([int(self._doc_id_to_offset[str(doc.id)])])[0]
            )
