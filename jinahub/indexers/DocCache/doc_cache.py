import hashlib
import os
import pickle
from typing import Tuple, Optional

from jina import Executor, DocumentArray, requests, Document

from jina_commons import get_logger


class _CacheHandler:
    """A handler for loading and serializing the in-memory cache of the DocCache.

    :param path: Path to the file from which to build the actual paths.
    :param logger: Instance of logger.
    """

    def __init__(self, path, logger):
        self.path = path
        self.id_to_hash_fn = path + '.ids'
        self.hash_to_id_fn = path + '.cache'
        try:
            self.id_to_hash = pickle.load(open(self.id_to_hash_fn, 'rb'))
            self.hash_to_id = pickle.load(open(self.hash_to_id_fn, 'rb'))
        except FileNotFoundError as e:
            logger.warning(
                f'File path did not exist : {path}.ids or {path}.cache: {e!r}. Creating new CacheHandler...'
            )
            self.id_to_hash = dict()
            self.hash_to_id = dict()

    def close(self):
        """Flushes the in-memory cache to pickle files."""
        pickle.dump(self.id_to_hash, open(self.id_to_hash_fn, 'wb'))
        pickle.dump(self.hash_to_id, open(self.hash_to_id_fn, 'wb'))


class DocCache(Executor):
    """An indexer that caches combinations of fields
    and filters out documents that have been previously cached

    :param fields: Fields of the Document used for generating unique hashing code, which is further used to detect the
        duplicates.
    """

    def __init__(
        self,
        fields: Optional[Tuple[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if fields is None:
            fields = ('content_hash', ),
        self.fields = fields
        self.logger = get_logger(self)
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        self.cache_handler = _CacheHandler(
            os.path.join(self.workspace, 'cache'), self.logger
        )

    @requests(on='/index')
    def index_or_remove_from_request(self, docs: DocumentArray, **kwargs):
        """Index Documents in the cache, by hashing self.fields

        If the document was already previously cached,
        it is removed from the docs, so no further Executor will receive it

        :param docs: the documents to cache
        """

        indices_to_remove = []
        for i, d in enumerate(docs):
            doc_hash = DocCache.hash_doc(d, self.fields)
            exists = doc_hash in self.cache_handler.hash_to_id.keys()

            self.cache_handler.id_to_hash[d.id] = doc_hash

            if exists:
                indices_to_remove.append(i)
            else:
                # we keep all the mappings from ids to hash
                # but only the FIRST from hash to id
                self.cache_handler.hash_to_id[doc_hash] = d.id

        indices_to_remove = sorted(indices_to_remove, reverse=True)
        for i in indices_to_remove:
            del docs[i]
        self.logger.info(
            f'Finished index op. Cached doc ids: {len(self.cache_handler.id_to_hash)}; cached hashes: {len(self.cache_handler.hash_to_id)}'
        )

    def close(self) -> None:
        self.cache_handler.close()

    @staticmethod
    def hash_doc(doc: Document, fields: Tuple[str]) -> bytes:
        """Calculate hash by which we cache.

        :param doc: the Document
        :param fields: the list of fields
        :return: the hash value of the fields
        """
        values = doc.get_attributes(*fields)
        if not isinstance(values, list):
            values = [values]
        data = ''
        for field, value in zip(fields, values):
            data += f'{field}:{value};'
        digest = hashlib.sha256(bytes(data.encode('utf8'))).digest()
        return digest

    @property
    def ids_count(self):
        """Return the size

        NOTE: we only count nr of entries from id angle
        """
        return len(self.cache_handler.id_to_hash)

    @property
    def hashes_count(self):
        """Return the nr of distinct hashes"""
        return len(self.cache_handler.hash_to_id)

    @requests(on='/update')
    def update(self, docs: DocumentArray, **kwargs):
        """Update the documents in the cache with the new content, by id"""
        for i, d in enumerate(docs):
            exists = d.id in self.cache_handler.id_to_hash.keys()

            if exists:
                new_doc_hash = DocCache.hash_doc(d, self.fields)
                old_cache_value = self.cache_handler.id_to_hash[d.id]

                self.cache_handler.id_to_hash[d.id] = new_doc_hash

                try:
                    del self.cache_handler.hash_to_id[old_cache_value]
                except KeyError:
                    # could have been deleted by a previous Document having the same hash
                    pass

                self.cache_handler.hash_to_id[new_doc_hash] = d.id
        self.logger.info(
            f'Finished update op. Cached doc ids: {len(self.cache_handler.id_to_hash)}; cached hashes: {len(self.cache_handler.hash_to_id)}'
        )

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, **kwargs):
        for i, d in enumerate(docs):
            exists = d.id in self.cache_handler.id_to_hash.keys()

            if exists:
                old_cache_value = self.cache_handler.id_to_hash[d.id]
                try:
                    del self.cache_handler.hash_to_id[old_cache_value]
                except KeyError as e:
                    # no guarantee
                    pass
                del self.cache_handler.id_to_hash[d.id]
        self.logger.info(
            f'Finished delete op. Cached doc ids: {len(self.cache_handler.id_to_hash)}; cached hashes: {len(self.cache_handler.hash_to_id)}'
        )
