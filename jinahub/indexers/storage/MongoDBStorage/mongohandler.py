__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional

from pymongo import MongoClient
from jina.logging.logger import JinaLogger
from jina import Document, DocumentArray


class MongoHandler:
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = 'jina_index_db',
        collection: str = 'jina_index_collection',
    ):
        self._logger = JinaLogger('mongo_handler')
        self._database_name = database
        self._collection_name = collection
        self._collection = None
        if username and password:
            self._connection = MongoClient(
                f'mongodb://{username}:{password}@{host}:{port}'
            )
        else:
            self._connection = MongoClient(f'mongodb://{host}:{port}')
        self._logger.info(f'Connected to mongodb instance at {host}:{port}')

    @property
    def collection(self):
        """Get the collection, if the collection is new, create index based on ID field."""
        if not self._collection:
            self._collection = self._connection[self._database_name][
                self._collection_name
            ]
            self._collection.create_index(
                'id', unique=True
            )  # create index on doc.id field if index not exist.
            return self._collection
        return self._collection

    def add(self, docs: DocumentArray, **kwargs):
        """Insert document from docs into mongodb instance."""
        dict_docs = []
        for doc in docs:
            item = doc.dict()
            if doc.embedding is not None:
                item['embedding'] = list(doc.embedding.flatten())
            dict_docs.append(item)
        self.collection.insert_many(
            documents=dict_docs,
            ordered=True,  # all document inserts will be attempted.
        )

    def update(self, docs: DocumentArray, **kwargs):
        """Update item from docs based on doc id."""
        for doc in docs:
            item = doc.dict()
            item['embedding'] = []
            if doc.embedding is not None:
                item['embedding'] = list(doc.embedding.flatten())
            self.collection.replace_one(
                filter={'id': {'$eq': doc.id}},
                replacement=item,
                upsert=True,
            )

    def delete(self, docs: DocumentArray, **kwargs):
        """Delete item from docs based on doc id."""
        doc_ids = [doc.id for doc in docs]
        self.collection.delete_many(filter={'id': {'$in': doc_ids}})

    def search(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            result = self.collection.find_one(
                filter={'id': doc.id}, projection={'_id': False}
            )
            if result:
                result.pop('embedding')
                retrieved_doc = Document(result)
                doc.update(retrieved_doc)

    def get_size(self) -> int:
        """Get the size of collection"""
        return self.collection.count()

    def close(self):
        """Close connection."""
        if self._connection:
            self._connection.close()
