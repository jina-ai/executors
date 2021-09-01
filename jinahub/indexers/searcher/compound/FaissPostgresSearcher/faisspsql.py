__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
import functools
from typing import Dict, Optional

from jina import DocumentArray, Executor, requests
from jina_commons import get_logger

try:
    from jinahub.indexers.searcher.FaissSearcher import FaissSearcher
except ImportError:  # noqa: E722
    from jina_executors.indexers.searcher.FaissSearcher.faiss_searcher import (
        FaissSearcher,
    )

try:
    from jinahub.indexers.storage.PostgreSQLStorage import PostgreSQLStorage
except ImportError:  # noqa: E722
    from jina_executors.indexers.storage.PostgreSQLStorage import PostgreSQLStorage


class FaissPostgresSearcher(Executor):
    """A Compound Indexer made up of a FaissSearcher (for vectors) and a Postgres
    Indexer

    :param dump_path: a path to a dump folder containing
    the dump data obtained by calling jina_commons.dump_docs
    :param use_dump_func: whether to use the dump
     function of PostgreSQLStorage, when dump_path is not provided
    """

    def __init__(
        self,
        dump_path: Optional[str] = None,
        use_dump_func: bool = False,
        total_shards: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = get_logger(self)

        self.total_shards = total_shards
        if self.total_shards is None:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )

        # when constructed from rolling update the dump_path is passed via a
        # runtime_arg
        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        # TODO allow some parametrization
        # in large dbs, you might not want snapshot
        # and just `only_delta=True`
        use_dump_func = use_dump_func or kwargs.get('runtime_args').get('use_dump_func')

        self._kv_indexer = None
        self._vec_indexer = None
        self._init_kwargs = kwargs

        if dump_path is None and use_dump_func is None:
            name = getattr(self.metas, 'name', self.__class__.__name__)
            self.logger.warning(
                f'No "dump_path" or "use_dump_func" provided '
                f'for {name}. Use .rolling_update() to re-initialize...'
            )

        if use_dump_func:
            self._kv_indexer = PostgreSQLStorage(**kwargs)
            dump_func = self._kv_indexer.get_snapshot
            self._vec_indexer = FaissSearcher(dump_func=dump_func, **kwargs)
        else:
            self._kv_indexer = PostgreSQLStorage(**kwargs)
            self._vec_indexer = FaissSearcher(dump_path=dump_path, **kwargs)

    @requests(on='/sync')
    def sync(self, parameters: Dict, **kwargs):
        """
        Sync the data from the PSQLStorage into the FaissSearcher
        """
        use_delta = parameters.get('use_delta', False)
        only_delta = parameters.get('only_delta', False)

        if only_delta:
            self.logger.warning(
                'Syncing via delta method. This cannot guarantee consistency'
            )
            deltas = self._kv_indexer._get_delta(
                shard_id=self.runtime_args.pea_id,
                total_shards=self.total_shards,
                # TODO the comparison can be avoided to increase performacne
                timestamp=0,
            )
            # deltas will be like DOC_ID, OPERATION, DATA
            self._vec_indexer._add_delta(deltas)

        else:
            self.logger.info('Syncing via snapshot...')
            if self.total_shards:
                dump_func = functools.partial(
                    self._kv_indexer.get_snapshot, total_shards=self.total_shards
                )
                timestamp = self._kv_indexer.last_snapshot_timestamp
                self._vec_indexer = FaissSearcher(
                    dump_func=dump_func, **self._init_kwargs
                )

                if use_delta:
                    self.logger.info(f'Now adding delta from timestamp {timestamp}')

                    deltas = self._kv_indexer._get_delta(
                        shard_id=self.runtime_args.pea_id,
                        total_shards=self.total_shards,
                        timestamp=timestamp,
                    )
                    # deltas will be like DOC_ID, OPERATION, DATA
                    self._vec_indexer._add_delta(deltas)
            else:
                self.logger.warning(
                    'total_shards is None, rolling update '
                    'via PSQL import will not be possible.'
                )

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Dict = None, **kwargs):
        """
        Search the vec embeddings in Faiss and then lookup the metadata in PSQL
        """
        if self._kv_indexer and self._vec_indexer:
            self._vec_indexer.search(docs, parameters)
            kv_parameters = copy.deepcopy(parameters)
            kv_parameters['traversal_paths'] = [
                path + 'm' for path in kv_parameters.get('traversal_paths', ['r'])
            ]
            self._kv_indexer.search(docs, kv_parameters)
        else:
            self.logger.warning('Indexers have not been initialized. Empty results')
            return

    @requests(on='/cleanup')
    def cleanup(self, **kwargs):
        """
        Completely removes rows in PSQL that have been marked for soft-deletion
        """
        self._kv_indexer.cleanup()

    @requests(on='/snapshot')
    def snapshot(self, **kwargs):
        """
        Create a snapshot of the current database table
        """
        self._kv_indexer.snapshot()

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Index new documents

        NOTE: PSQL has a uniqueness constraint on ID
        """
        self._kv_indexer.add(docs, parameters, **kwargs)

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Update documents in PSQL, based on id
        """
        self._kv_indexer.update(docs, parameters, **kwargs)

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """
        Delete docs from PSQL, based on id
        """
        self._kv_indexer.delete(docs, parameters, **kwargs)
