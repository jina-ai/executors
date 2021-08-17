from typing import Dict, Optional, List

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
            most similar embeddings. The distance metrics supported are the ones supported by `DocumentArray` match method.
        """
        super().__init__(**kwargs)
        self._docs = DocumentArrayMemmap(self.workspace + f'/{index_file_name}')
        self.default_traversal_paths = default_traversal_paths or ['r']
        self.default_top_k = default_top_k
        self._distance = distance_metric
        self._use_scipy = True
        if distance_metric in ['cosine', 'euclidean', 'sqeuclidean']:
            self._use_scipy = False

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
            metric=self._distance,
            use_scipy=self._use_scipy,
            limit=top_k,
        )
        self._flush = False

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
        flat_docs = docs.traverse_flat(traversal_paths)
        delete_docs_ids = set(flat_docs.get_attributes('id'))
        for idx in reversed(range(len(self._docs))):
            if self._docs[idx].id in delete_docs_ids:
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

        update_docs_ids = set(flat_docs.get_attributes('id'))
        for idx in reversed(range(len(self._docs))):
            if self._docs[idx].id in update_docs_ids:
                update_doc = flat_docs.get_attributes_with_docs(self._docs[idx].id)[1][0]
                del self._docs[idx]
                self._docs.append(update_doc)

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """retrieve embedding of Documents by id

        :param docs: DocumentArray to search with
        """
        if not docs:
            return
        for doc in docs:
            doc.embedding = self._docs[doc.id].embedding
