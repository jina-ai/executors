__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable, Dict, Tuple

import redis
from redis import Redis
from jina.logging.logger import JinaLogger
from jina import Executor, requests, DocumentArray, Document
from jina_commons.batching import get_docs_batch_generator


class RedisStorage(Executor):
    """
    :class:`RedisStorage` redis-based Storage Indexer.

    Initialize the RedisStorage.

    :param hostname: hostname of the redis server
    :param port: the redis port
    :param db: the database number
    :param default_traversal_paths: default traversal paths
    :param default_batch_size: default batch size
    :param args: other arguments
    :param kwargs: other keyword arguments
    """
    def __init__(self,
                 hostname: str = '127.0.0.1',
                 # default port on linux
                 port: int = 6379,
                 db: int = 0,
                 default_traversal_paths: Tuple = ('r',),
                 default_batch_size: int = 32,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self.hostname = hostname
        self.port = port
        self.db = db
        self.connection_pool = redis.ConnectionPool(host=self.hostname, port=self.port, db=self.db)
        self.logger = JinaLogger(self.__class__.__name__)

    def get_query_handler(self) -> 'Redis':
        """Get the redis client handler.
        """
        import redis
        try:
            r = redis.Redis(connection_pool=self.connection_pool)
            r.ping()
            return r
        except redis.exceptions.ConnectionError as r_con_error:
            self.logger.error('Redis connection error: ', r_con_error)
            raise

    def _query_batch(self, docs: Iterable[Document]):
        with self.get_query_handler() as redis_handler:
            results = redis_handler.mget([doc.id for doc in docs])
            for doc, result in zip(docs, results):
                if not result:
                    continue
                data = bytes(result)
                retrieved_doc = Document(data)
                retrieved_doc.pop('embedding')
                doc.MergeFrom(retrieved_doc)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Searches documents in the redis server by document ID

        :param docs: document array
        :param parameters: parameters to the request
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._query_batch(document_batch)

    def _upsert_batch(self, docs: Iterable[Document]):
        with self.get_query_handler().pipeline() as redis_handler:
            redis_handler.mset({doc.id: doc.SerializeToString() for doc in docs})
            redis_handler.execute()

    @requests(on=['/upsert'])
    def upsert(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Upserts documents in the redis server where the key is the document ID. If a document with the same ID
        already exists, an update operation is performed instead.

        :param docs: document array
        :param parameters: parameters to the request
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._upsert_batch(document_batch)

    def _get_existing_ids(self, docs: Iterable[Document]):
        with self.get_query_handler() as redis_handler:
            response = redis_handler.mget([doc.id for doc in docs])
            existing_ids = set([doc_a.id for doc_a, doc_b in zip(docs, response) if doc_b])
            return set(existing_ids)

    def _add_batch(self, docs: Iterable[Document]):
        existing = self._get_existing_ids(docs)
        if existing:
            self.logger.warning(f'The following IDs already exist: {", ".join(existing)}')
        docs_to_add = {doc.id: doc.SerializeToString() for doc in docs if doc.id not in existing}
        if docs_to_add:
            with self.get_query_handler().pipeline() as redis_handler:
                redis_handler.mset(docs_to_add)
                redis_handler.execute()

    @requests(on=['/index'])
    def add(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Indexes documents in the redis server where the key is the document ID. If a document with the same ID
        already exists, a DuplicateIDError is raised

        :param docs: document array
        :param parameters: parameters to the request
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._add_batch(document_batch)

    def _update_batch(self, docs: Iterable[Document]):
        existing = self._get_existing_ids(docs)
        non_existing = set([doc.id for doc in docs]) - existing
        if non_existing:
            self.logger.warning(f'The following IDs do not exist: {", ".join(non_existing)}')
        docs_to_update = {doc.id: doc.SerializeToString() for doc in docs if doc.id in existing}
        if docs_to_update:
            with self.get_query_handler().pipeline() as redis_handler:
                redis_handler.mset(docs_to_update)
                redis_handler.execute()

    @requests(on=['/update'])
    def update(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Updates documents in the redis server where the key is the document ID. If no document with the same ID
        exists, a NoSuchIDError is raised

        :param docs: document array
        :param parameters: parameters to the request
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._update_batch(document_batch)

    def _delete_batch(self, docs: Iterable[Document]):
        existing = self._get_existing_ids(docs)
        non_existing = set([doc.id for doc in docs]) - existing
        if non_existing:
            self.logger.warning(f'The following IDs do not exist: {", ".join(non_existing)}')
        if existing:
            with self.get_query_handler().pipeline() as redis_handler:
                redis_handler.delete(*existing)
                redis_handler.execute()

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Deletes documents in the redis server by ID. If no document with the same ID exists, nothing happens.

        :param docs: document array
        :param parameters: parameters to the request
        """
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._delete_batch(document_batch)
