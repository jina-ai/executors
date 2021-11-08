__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
import os
from typing import Dict, Iterable, Optional

import hnswlib
from bidict import bidict
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class HnswlibSearcher(Executor):
    """Hnswlib powered vector indexer.

    This indexer uses the HNSW algorithm to index and search for vectors. It does not
    require training, and can be built up incrementally.
    """

    def __init__(
        self,
        limit: int = 10,
        metric: str = 'cosine',
        dim: int = 0,
        max_elements: int = 1_000_000,
        ef_construction: int = 400,
        ef_query: int = 50,
        max_connection: int = 64,
        dump_path: Optional[str] = None,
        traversal_paths: Iterable[str] = ('r',),
        is_distance: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param limit: Number of results to get for each query document in search
        :param metric: Distance metric type, can be 'euclidean', 'inner_product', or 'cosine'
        :param dim: The dimensionality of vectors to index
        :param max_elements: Maximum number of elements (vectors) to index
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_query: The query time accuracy/speed trade-off
        :param max_connection: The maximum number of outgoing connections in the
            graph (the "M" parameter)
        :param dump_path: The path to the directory from where to load, and where to
            save the index state
        :param traversal_paths: The default traverseal path on docs (used for indexing,
            search and update), e.g. ['r'], ['c']
        :param is_distance: Boolean flag that describes if distance metric need to be reinterpreted as similarities.
        """
        super().__init__(*args, **kwargs)
        self.limit = limit
        self.metric = metric
        self.dim = dim
        self.max_elements = max_elements
        self.traversal_paths = traversal_paths
        self.ef_construction = ef_construction
        self.ef_query = ef_query
        self.max_connection = max_connection
        self.dump_path = dump_path
        self.is_distance = is_distance

        self.logger = JinaLogger(self.__class__.__name__)
        self._index = hnswlib.Index(space=self.metric_type, dim=self.dim)

        dump_path = self.dump_path or kwargs.get('runtime_args', {}).get(
            'dump_path', None
        )
        if dump_path is not None:
            self.logger.info('Starting to build HnswlibSearcher from dump data')

            self._index.load_index(
                f'{self.dump_path}/index.bin', max_elements=self.max_elements
            )
            with open(f'{self.dump_path}/ids.json', 'r') as f:
                self._ids_to_inds = bidict(json.load(f))

        else:
            self.logger.info('No `dump_path` provided, initializing empty index.')
            self._init_empty_index()

        self._index.set_ef(self.ef_query)

    def _init_empty_index(self):
        self._index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.max_connection,
        )
        self._ids_to_inds = bidict()

    @requests(on='/search')
    def search(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Attach matches to the Documents in `docs`, each match containing only the
        `id` of the matched document and the `score`.

        :param docs: An array of `Documents` that should have the `embedding` property
            of the same dimension as vectors in the index
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. Supported keys are
            `traversal_paths`, `limit` and `ef_query`.
        """
        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        docs_search = docs.traverse_flat(traversal_paths)
        if len(docs_search) == 0:
            return

        ef_query = parameters.get('ef_query', self.ef_query)
        limit = int(parameters.get('limit', self.limit))

        self._index.set_ef(ef_query)

        if limit > len(self._ids_to_inds):
            self.logger.warning(
                f'The `limit` parameter is set to a value ({limit}) that is higher than'
                f' the number of documents in the index ({len(self._ids_to_inds)})'
            )
            limit = len(self._ids_to_inds)

        embeddings_search = docs_search.embeddings
        if embeddings_search.shape[1] != self.dim:
            raise ValueError(
                'Query documents have embeddings with dimension'
                f' {embeddings_search.shape[1]}, which does not match the dimension of'
                f' the index ({self.dim})'
            )

        indices, dists = self._index.knn_query(docs_search.embeddings, k=limit)

        for i, (indices_i, dists_i) in enumerate(zip(indices, dists)):
            for idx, dist in zip(indices_i, dists_i):
                match = Document(id=self._ids_to_inds.inverse[idx])
                if self.is_distance:
                    match.scores[self.metric] = dist
                elif self.metric in ["inner_product", "cosine"]:
                    match.scores[self.metric] = 1 - dist
                elif self.metric == 'euclidean':
                    match.scores[self.metric] = 1 / (1 + dist)
                else:
                    match.scores[self.metric] = dist
                docs_search[i].matches.append(match)

    @requests(on='/index')
    def index(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Index the Documents' embeddings. If the document is already in index, it
        will be updated.

        :param docs: Documents whose `embedding` to index.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`.
        """
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        if docs is None:
            return

        docs_to_update = docs.traverse_flat(traversal_paths)
        if len(docs_to_update) == 0:
            return

        embeddings = docs_to_update.embeddings
        if embeddings.shape[-1] != self.dim:
            raise ValueError(
                f'Attempted to index vectors with dimension'
                f' {embeddings.shape[-1]}, but dimension of index is {self.dim}'
            )

        ids = docs_to_update.get_attributes('id')
        index_size = self._index.element_count
        doc_inds = []
        for doc in docs_to_update:
            if doc.id not in self._ids_to_inds:
                doc_inds.append(index_size)
                index_size += 1
            else:
                self.logger.info(
                    f'Document with id {doc.id} already in index, updating.'
                )
                doc_inds.append(self._ids_to_inds[doc.id])

        self._index.add_items(embeddings, ids=doc_inds)
        self._ids_to_inds.update({_id: ind for _id, ind in zip(ids, doc_inds)})

    @requests(on='/update')
    def update(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Update the Documents' embeddings. If a Document is not already present in
        the index, it will get ignored, and a warning will be raised.

        :param docs: Documents whose `embedding` to update.
        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `traversal_paths`.
        """
        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        if docs is None:
            return

        docs_to_update = docs.traverse_flat(traversal_paths)
        if len(docs_to_update) == 0:
            return

        doc_inds, docs_filtered = [], []
        for doc in docs_to_update:
            if doc.id not in self._ids_to_inds:
                self.logger.warning(
                    f'Attempting to update document with id {doc.id} which is not'
                    ' indexed, skipping. To add documents to index, use the /index'
                    ' endpoint'
                )
            else:
                docs_filtered.append(doc)
                doc_inds.append(self._ids_to_inds[doc.id])
        docs_filtered = DocumentArray(docs_filtered)

        embeddings = docs_filtered.embeddings
        if embeddings.shape[-1] != self.dim:
            raise ValueError(
                f'Attempted to update vectors with dimension'
                f' {embeddings.shape[-1]}, but dimension of index is {self.dim}'
            )

        self._index.add_items(embeddings, ids=doc_inds)

    @requests(on='/delete')
    def delete(self, parameters: Dict, **kwargs):
        """Delete entries from the index by id

        :param parameters: parameters to the request. Should contain the list of ids
            of entries (Documents) to delete under the `ids` key
        """
        deleted_ids = parameters.get('ids', [])

        for _id in set(deleted_ids).intersection(self._ids_to_inds.keys()):
            ind = self._ids_to_inds[_id]
            self._index.mark_deleted(ind)
            del self._ids_to_inds[_id]

    @requests(on='/dump')
    def dump(self, parameters: Dict = {}, **kwargs):
        """Save the index and document ids.

        The index and ids will be saved separately for each shard.

        :param parameters: Dictionary with optional parameters that can be used to
            override the parameters set at initialization. The only supported key is
            `dump_path`.
        """

        dump_path = parameters.get('dump_path', self.dump_path)
        dump_path = os.path.join(dump_path, str(self.runtime_args.pea_id))
        os.mkdir(dump_path)
        if dump_path is None:
            raise ValueError(
                'The `dump_path` must be provided to save the indexer state.'
            )

        self._index.save_index(f'{dump_path}/index.bin')
        with open(f'{dump_path}/ids.json', 'w') as f:
            json.dump(dict(self._ids_to_inds), f)

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index of all entries."""
        self._index = hnswlib.Index(space=self.metric_type, dim=self.dim)
        self._init_empty_index()
        self._index.set_ef(self.ef_query)

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(
            tags={
                'count_deleted': self._index.element_count - len(self._ids_to_inds),
                'count_indexed': self._index.element_count,
                'count_active': len(self._ids_to_inds),
            }
        )
        return DocumentArray([status])

    @property
    def metric_type(self):
        if self.metric == 'euclidean':
            metric_type = 'l2'
        elif self.metric == 'cosine':
            metric_type = 'cosine'
        elif self.metric == 'inner_product':
            metric_type = 'ip'

        if self.metric not in ['euclidean', 'cosine', 'inner_product']:
            self.logger.warning(
                f'Invalid distance metric {self.metric} for HNSW index construction! '
                'Default to euclidean distance'
            )
            metric_type = 'l2'

        return metric_type
