__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, Iterable, Optional

import hnswlib
import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class HnswlibSearcher(Executor):
    """Hnswlib powered vector indexer

    For more information about the Hnswlib supported parameters, please consult:
        - https://github.com/nmslib/hnswlib

    .. note::
        Hnswlib package dependency is only required at the query time.
    """

    def __init__(
        self,
        top_k: int = 10,
        metric: str = 'cosine',
        dim: int = 0,
        max_elements: int = 1000,
        ef_construction: int = 400,
        ef_query: int = 50,
        max_connection: int = 64,
        dump_path: Optional[str] = None,
        traversal_paths: Iterable[str] = ('r',),
        *args,
        **kwargs,
    ):
        """
        Initialize an HnswlibSearcher

        :param top_k: Number of results to get for each query document
        :param distance: Distance type, can be 'l2', 'ip', or 'cosine'
        :param dim: The dimensionality of vectors to index
        :param max_elements: Maximum number of elements (vectors) to index
        :param ef_construction: Defines a construction time/accuracy trade-off
        :param ef_query: Sets the query time accuracy/speed trade-off
        :param max_connection: defines tha maximum number of outgoing connections in the
            graph (the "M" parameter)
        :param dump_path: The path from where to load ids and vectors
        :param traversal_paths: The default traverseal path on docs, e.g. ['r'], ['c']
        """
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.metric = metric
        self.traversal_paths = traversal_paths
        self.ef_construction = ef_construction
        self.ef_query = ef_query
        self.max_connection = max_connection

        self.logger = JinaLogger(self.__class__.__name__)
        self._index = hnswlib.Index(space=self.metric, dim=dim)

        dump_path = dump_path or kwargs.get('runtime_args', {}).get('dump_path', None)
        if dump_path is not None:
            self.logger.info('Start building "HnswlibSearcher" from dump data')

            # Load index itself + saved ids

        else:
            self._indexer.init_index(
                max_elements=max_elements,
                ef_construction=self.ef_construction,
                M=self.max_connection,
            )

            self.logger.info(
                'No data loaded in "HnswlibSearcher", initializing empty index.'
            )

    @requests(on='/search')
    def search(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """

        :param docs: `Document` with `.embedding` the same shape as the `Documents`
            it has stored.
        :param parameters: dictionary to define the `traversal_paths` and the `top_k`.

        :return: Attaches matches to the Documents sent as inputs, with the id of the
            match, and its embedding.
        """
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
                match.scores[self.metric] = dist

                doc.matches.append(match)

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: Optional[DocumentArray], **kwargs):
        if docs is None:
            return
        for doc in docs:
            doc_idx = self._doc_id_to_offset.get(doc.id)
            if doc_idx is not None:
                doc.embedding = np.array(self._indexer.get_items([int(doc_idx)])[0])
            else:
                self.logger.warning(f'Document {doc.id} not found in index')

    @requests(on='/index')
    def index(self, docs: Optional[DocumentArray], **kwargs):
        pass

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], **kwargs):
        pass

    @requests(on='/delete')
    def delete(self, docs: Optional[DocumentArray], **kwargs):
        pass

    @requests(on='/save')
    def save(self, target_path: Optional[str] = None, **kwargs):
        pass
