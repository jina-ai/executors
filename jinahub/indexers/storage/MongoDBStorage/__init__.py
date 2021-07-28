__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"
from typing import Tuple, Generator, Dict, List, Optional

import numpy as np
from jina import Executor, requests, DocumentArray, Document
from jina_commons.indexers.dump import export_dump_streaming

from .mongohandler import MongoHandler


def doc_without_embedding(d: Document) -> str:
    new_doc = Document(d, copy=True, hash_content=False)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


class MongoDBStorage(Executor):
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 27017,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: str = 'jina_index',
        collection: str = 'jina_index',
        default_traversal_paths: List[str] = ['r'],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._handler = MongoHandler(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
            collection=collection,
        )
        self._traversal_paths = default_traversal_paths

    @requests(on='/index')
    def add(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """Add Documents to MongoDB

        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get('traversal_paths', self._traversal_paths)
        self._handler.add(docs.traverse_flat(traversal_paths))

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """Updated document from the database.

        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get('traversal_paths', self._traversal_paths)
        self._handler.update(docs.traverse_flat(traversal_paths))

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """Delete document from the database.

        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        traversal_paths = parameters.get('traversal_paths', self._traversal_paths)
        self._handler.delete(docs.traverse_flat(traversal_paths))

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """Get the Documents by the ids of the docs in the DocArray

        :param docs: the DocumentArray to search with (they only need to have the `.id` set)
        :param parameters: the parameters to this request
        """
        traversal_paths = parameters.get('traversal_paths', self._traversal_paths)
        self._handler.search(docs.traverse_flat(traversal_paths))

    @requests(on='/dump')
    def dump(self, parameters: Dict = {}, **kwargs):
        """Dump the index

        :param parameters: a dictionary containing the parameters for the dump
        """

        path = parameters.get('dump_path')
        if path is None:
            raise ValueError(f'No "dump_path" provided for {self}')

        shards = parameters.get('shards', None)
        if shards is None:
            raise ValueError(f'No "shards" provided for {self}.')

        export_dump_streaming(
            path, shards=int(shards), size=self.size, data=self._get_generator()
        )

    @property
    def size(self) -> int:
        """Obtain the size of the table

        .. # noqa: DAR201
        """
        return self._handler.get_size()

    def close(self) -> None:
        """
        Close the connections in the connection pool
        """
        self._handler.close()
        super().close()

    def _get_generator(self) -> Generator[Tuple[str, np.array, bytes], None, None]:
        # always order the dump by id as integer
        records = self._handler.collection.find({}, projection={'_id': False})
        for record in records:
            vec = np.array(record['embedding'])
            doc = Document(record, hash_content=False)
            metas = doc_without_embedding(doc)
            yield doc.id, vec, metas
