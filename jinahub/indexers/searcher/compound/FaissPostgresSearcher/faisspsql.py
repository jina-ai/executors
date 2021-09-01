__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
import datetime
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
    :param startup_sync: whether to use the dump
     function of PostgreSQLStorage, when dump_path is not provided
    :param startup_sync_args: the arguments to be passed to the self.sync call on
    startup (if startup_sync)
    """

    def __init__(
        self,
        dump_path: Optional[str] = None,
        startup_sync: bool = False,
        startup_sync_args: Optional[None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = get_logger(self)

        self.total_shards = self.runtime_args.parallel
        if self.total_shards is None:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )

        # when constructed from rolling update
        # args are passed via runtime_args
        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')
        startup_sync = startup_sync or kwargs.get('runtime_args').get('startup_sync')

        self._kv_indexer = None
        self._vec_indexer = None
        self._init_kwargs = kwargs

        (
            self._kv_indexer,
            self._vec_indexer,
            startup_sync,
            startup_sync_args,
        ) = self._init_executors(dump_path, kwargs, startup_sync, startup_sync_args)
        if startup_sync:
            self.sync(parameters=startup_sync_args)

    def _init_executors(self, dump_path, kwargs, startup_sync, startup_sync_args):
        kv_indexer = PostgreSQLStorage(**kwargs)

        if startup_sync_args is None and startup_sync:
            startup_sync_args = {}
        # if the user passes args for syncing, they must've meant to sync as well
        if startup_sync_args and (startup_sync is False or startup_sync is None):
            startup_sync = True

        if dump_path is None and startup_sync is None:
            name = getattr(self.metas, 'name', self.__class__.__name__)
            self.logger.warning(
                f'No "dump_path" or "use_dump_func" provided '
                f'for {name}. Use .rolling_update() to re-initialize...'
            )
        if startup_sync:
            vec_indexer = FaissSearcher(**kwargs)
            startup_sync_args['startup'] = True
        else:
            vec_indexer = FaissSearcher(dump_path=dump_path, **kwargs)
        return kv_indexer, vec_indexer, startup_sync, startup_sync_args

    @requests(on='/sync')
    def sync(self, parameters: Optional[Dict], **kwargs):
        """
        Sync the data from the PSQLStorage into the FaissSearcher
        """
        use_delta = parameters.get('use_delta', False)
        only_delta = parameters.get('only_delta', False)

        if only_delta:
            self._sync_only_delta(parameters, **kwargs)

        else:
            self._sync_snapshot(use_delta)

    def _sync_snapshot(self, use_delta):
        self.logger.info('Syncing via snapshot...')
        if self.total_shards:
            dump_func = functools.partial(
                self._kv_indexer.get_snapshot, total_shards=self.total_shards
            )
            timestamp = self._kv_indexer.last_snapshot_timestamp
            self._vec_indexer = FaissSearcher(dump_func=dump_func, **self._init_kwargs)

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

    def _sync_only_delta(self, parameters, **kwargs):
        timestamp = parameters.get('timestamp', None)
        startup = parameters.get('startup', False)
        if timestamp is None:
            if startup:
                timestamp = datetime.datetime.min
            else:
                self.logger.error(
                    f'No timestamp provided in parameters: '
                    f'{parameters}. Cannot do sync delta'
                )
                return

        if startup:
            # this was startup, so treat the method as a dump_func
            dump_func = functools.partial(
                self._kv_indexer._get_delta,
                total_shards=self.total_shards,
                timestamp=datetime.datetime.min,
            )
            self._vec_indexer = FaissSearcher(dump_func=dump_func, **self._init_kwargs)
        else:
            self.logger.warning(
                'Syncing via delta method. This cannot guarantee consistency'
            )
            deltas = self._kv_indexer._get_delta(
                shard_id=self.runtime_args.pea_id,
                total_shards=self.total_shards,
                timestamp=timestamp,
            )
            # deltas will be like DOC_ID, OPERATION, DATA
            self._vec_indexer._add_delta(deltas)

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
