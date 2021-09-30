__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import json
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
        top_k: int = 10,
        metric: str = 'cosine',
        dim: int = 0,
        max_elements: int = 1_000_000,
        ef_construction: int = 400,
        ef_query: int = 50,
        max_connection: int = 64,
        dump_path: Optional[str] = None,
        traversal_paths: Iterable[str] = ('r',),
        *args,
        **kwargs,
    ):
        """
        :param top_k: Number of results to get for each query document in search
        :param distance: Distance type, can be 'l2', 'ip', or 'cosine'
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
        """
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.metric = metric
        self.dim = dim
        self.max_elements = max_elements
        self.traversal_paths = traversal_paths
        self.ef_construction = ef_construction
        self.ef_query = ef_query
        self.max_connection = max_connection
        self.dump_path = dump_path

        self.logger = JinaLogger(self.__class__.__name__)
        self._index = hnswlib.Index(space=self.metric, dim=self.dim)

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
            `traversal_paths`, `top_k` and `ef_query`.
        """
        if docs is None:
            return

        traversal_paths = parameters.get('traversal_paths', self.traversal_paths)
        docs_search = docs.traverse_flat(traversal_paths)
        if len(docs_search) == 0:
            return

        ef_query = parameters.get('ef_query', self.ef_query)
        top_k = parameters.get('top_k', self.top_k)

        self._index.set_ef(ef_query)

        if top_k > len(self._ids_to_inds):
            self.logger.warning(
                f'The `top_k` parameter is set to a value ({top_k}) that is higher than'
                f' the number of documents in the index ({len(self._ids_to_inds)})'
            )
            top_k = len(self._ids_to_inds)

        embeddings_search = docs_search.embeddings
        if embeddings_search.shape[1] != self.dim:
            raise ValueError(
                'Query documents have embeddings with dimension'
                f' {embeddings_search.shape[1]}, which does not match the dimension of'
                f' the index ({self.dim})'
            )

        indices, dists = self._index.knn_query(docs_search.embeddings, k=top_k)

        for i, (indices_i, dists_i) in enumerate(zip(indices, dists)):
            for idx, dist in zip(indices_i, dists_i):
                match = Document(id=self._ids_to_inds.inverse[idx])
                match.scores[self.metric] = dist

                docs_search[i].matches.append(match)

    @requests(on='/index')
    def index(
        self, docs: Optional[DocumentArray] = None, parameters: Dict = {}, **kwargs
    ):
        """Index the Documents' embeddings. The documents should not be already
        present in the index - for that case, use the update endpoint.

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
        doc_inds = list(range(index_size, index_size + len(ids)))

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
        if dump_path is None:
            raise ValueError(
                'The `dump_path` must be provided to save the indexer state.'
            )

        self._index.save_index(f'{dump_path}/index.bin')
        with open(f'{dump_path}/ids.json', 'w') as f:
            self._ids = json.dump(dict(self._ids_to_inds), f)

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index of all entries."""
        self._index = hnswlib.Index(space=self.metric, dim=self.dim)
        self._init_empty_index()
        self._index.set_ef(self.ef_query)
