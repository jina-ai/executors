__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import copy
import datetime
import functools
from typing import Dict, Optional

import numpy as np
from jina import DocumentArray, Executor, requests
from jina_commons import get_logger

from jinahub.indexers.searcher.FaissSearcher import FaissSearcher
from jinahub.indexers.storage.PostgreSQLStorage import PostgreSQLStorage


class FaissPostgresSearcher(Executor):
    """A Compound Indexer made up of a FaissSearcher (for vectors) and a
    PostgreSQLStorage
    """

    def __init__(
        self,
        dump_path: Optional[str] = None,
        startup_sync_args: Optional[Dict] = None,
        total_shards: Optional[int] = None,
        **kwargs,
    ):
        """
        :param dump_path: a path to a dump folder containing
        the dump data obtained by calling jina_commons.dump_docs
        :param startup_sync_args: the arguments to be passed to the self.sync call on
        startup
        :param total_shards: the total nr of shards that this shard is part of.

            NOTE: This is REQUIRED in k8s, since there `runtime_args.parallel` is always 1
        """
        super().__init__(**kwargs)
        self.logger = get_logger(self)

        if total_shards is None:
            self.total_shards = getattr(self.runtime_args, 'parallel', None)
        else:
            self.total_shards = total_shards

        if self.total_shards is None:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )

        # when constructed from rolling update
        # args are passed via runtime_args
        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')

        self._kv_indexer = None
        self._vec_indexer = None
        self._init_kwargs = kwargs

        (
            self._kv_indexer,
            self._vec_indexer,
        ) = self._init_executors(dump_path, kwargs, startup_sync_args)
        if startup_sync_args:
            startup_sync_args['startup'] = True
            self.sync(parameters=startup_sync_args)

    def _init_executors(self, dump_path, kwargs, startup_sync_args):
        # float32 because that's what faiss expects
        kv_indexer = PostgreSQLStorage(dump_dtype=np.float32, **kwargs)
        vec_indexer = FaissSearcher(dump_path=dump_path, **kwargs)

        if dump_path is None and startup_sync_args is None:
            name = getattr(self.metas, 'name', self.__class__.__name__)
            self.logger.warning(
                f'No "dump_path" or "use_dump_func" provided '
                f'for {name}. Use .rolling_update() to re-initialize...'
            )
        return kv_indexer, vec_indexer

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

        :param docs: `Document` with `.embedding` the same shape as the
            `Documents` stored in the `FaissSearcher`. The ids of the `Documents`
            stored in `FaissSearcher` need to exist in the `PostgreSQLStorage`.
            Otherwise you will not get back the original metadata.
        :param parameters: dictionary to define the ``traversal_paths``. This will
            override the default parameters set at init.

        :return: The `FaissSearcher` attaches matches to the `Documents` sent as inputs,
            with the id of the match, and its embedding. Then, the `PostgreSQLStorage`
            retrieves the full metadata (original text or image blob) and attaches
            those to the Document. You receive back the full Document.

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
    def index(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """Index new documents

        NOTE: PSQL has a uniqueness constraint on ID
        """
        self._kv_indexer.add(docs, parameters, **kwargs)

    @requests(on='/update')
    def update(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Update documents in PSQL, based on id
        """
        self._kv_indexer.update(docs, parameters, **kwargs)

    @requests(on='/delete')
    def delete(self, docs: Optional[DocumentArray], parameters: Dict, **kwargs):
        """
        Delete docs from PSQL, based on id.

        By default, it will be a soft delete, where the entry is left in the DB,
        but its data will be set to None
        """
        if 'soft_delete' not in parameters:
            parameters['soft_delete'] = True

        self._kv_indexer.delete(docs, parameters, **kwargs)
