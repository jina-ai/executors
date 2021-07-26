__copyright__ = "Copyright (c) 2020 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Iterable, Dict, Tuple

import redis
from redis import Redis
from jina.logging.logger import JinaLogger
from jina import Executor, requests, DocumentArray, Document
from jina_commons.batching import get_docs_batch_generator

class RedisStorage():

    def __init__(self,
                 hostname: str = '0.0.0.0',
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
        """Get the database handler.
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
                # TODO: check why is this performed for postgres storage
                # retrieved_doc.pop('embedding')
                doc.MergeFrom(retrieved_doc)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
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

    @requests(on=['/index', '/update'])
    def upsert(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._upsert_batch(document_batch)

    def _delete_batch(self, docs: Iterable[Document]):
        with self.get_query_handler().pipeline() as redis_handler:
            redis_handler.delete(*[doc.id for doc in docs])
            redis_handler.execute()

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict, **kwargs):
        if docs:
            document_batches_generator = get_docs_batch_generator(
                docs,
                traversal_path=parameters.get('traversal_paths', self.default_traversal_paths),
                batch_size=parameters.get('batch_size', self.default_batch_size)
            )
            for document_batch in document_batches_generator:
                self._delete_batch(document_batch)
