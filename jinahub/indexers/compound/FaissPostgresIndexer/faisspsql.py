__copyright__ = 'Copyright (c) 2021 Jina AI Limited. All rights reserved.'
__license__ = 'Apache-2.0'

import copy
import datetime
from typing import Dict, Optional

import numpy as np
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

from .faiss_searcher import FaissSearcher
from .postgres_indexer import PostgreSQLStorage

FAISS_PREFETCH_SIZE = 256


class FaissPostgresIndexer(Executor):
    """A Compound Indexer made up of a FaissSearcher (for vectors) and a
    PostgreSQLStorage
    """

    def __init__(
        self,
        dump_path: Optional[str] = None,
        startup_sync_args: Optional[Dict] = None,
        total_shards: Optional[int] = None,
        max_num_training_points: int = 0,
        **kwargs,
    ):
        """
        :param dump_path: a path to a dump folder containing
        the dump data obtained by calling jina_commons.dump_docs
        :param startup_sync_args: the arguments to be passed to the self.sync call on
        startup
        :param total_shards: the total nr of shards that this shard is part of.

            NOTE: This is REQUIRED in k8s, since there `runtime_args.replicas` is always 1
        """
        super().__init__(**kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        if total_shards is None:
            self.total_shards = getattr(self.runtime_args, 'replicas', None)
        else:
            self.total_shards = total_shards

        if self.total_shards is None:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )
        else:
            # shards is passed as str from Flow.add in yaml
            self.total_shards = int(self.total_shards)

        self.max_num_training_points = max_num_training_points

        # when constructed from rolling update
        # args are passed via runtime_args
        dump_path = dump_path or kwargs.get('runtime_args').get('dump_path')

        self._kv_indexer = None
        self._vec_indexer = None
        self._init_kwargs = kwargs

        self._init_executors(dump_path, kwargs, startup_sync_args)
        if startup_sync_args:
            startup_sync_args['init_faiss'] = True
            self.sync(parameters=startup_sync_args)

    def _init_executors(self, dump_path, kwargs, startup_sync_args):
        # float32 because that's what faiss expects
        self._kv_indexer = PostgreSQLStorage(dump_dtype=np.float32, **kwargs)

        try:
            import tempfile

            trained_index_file = kwargs.pop('trained_index_file', None)
            temp_trained_file = tempfile.NamedTemporaryFile(
                suffix='.bin', prefix='trained_faiss_'
            )

            # check the model from PSQL
            if (not trained_index_file) and (not self._kv_indexer.dry_run):
                (
                    trained_model,
                    trained_model_checksum,
                ) = self._kv_indexer.get_trained_model()

                index_key = kwargs.get('index_key', None)

                if (trained_model is not None) and (
                    not index_key or index_key == trained_model_checksum
                ):
                    self.logger.info('Load the trained index model from PSQL.')
                    temp_trained_file.write(trained_model)
                    temp_trained_file.flush()

                    trained_index_file = temp_trained_file.name

            self._vec_indexer = FaissSearcher(
                dump_path=dump_path,
                trained_index_file=trained_index_file,
                prefetch_size=FAISS_PREFETCH_SIZE,
                **kwargs,
            )
        finally:
            temp_trained_file.close()

        if dump_path is None and startup_sync_args is None:
            name = getattr(self.metas, 'name', self.__class__.__name__)
            self.logger.warning(
                f'No "dump_path" provided '
                f'for {name}. Use .rolling_update() to re-initialize...'
            )

    @requests(on='/train')
    def train(self, parameters: Optional[Dict] = {}, **kwargs):
        """
        Train the index by fetching training data from PSQLStorage
        :param parameters:
        :param kwargs:
        :return:
        """

        force_retrain = parameters.get('force', False)

        if self.runtime_args.shard_id != 0:
            return

        if self._vec_indexer.is_trained and not force_retrain:
            self.logger.warning(
                'The index has already been trained. '
                'Please use the `force=True` in parameters to retrain the index'
            )
            return

        max_num_training_points = int(
            parameters.get('max_num_training_points', self.max_num_training_points)
        )

        self.logger.info('Taking indexed data as training points...')
        train_docs = DocumentArray()
        for doc in self._kv_indexer.get_document_iterator(
            limit=max_num_training_points, check_embedding=True, return_embedding=True
        ):
            train_docs.append(doc)

        if len(train_docs) == 0:
            self.logger.warning('The training failed as there is no data for training!')
            return

        self._vec_indexer.train(train_docs, parameters={'index_data': False})

        import tempfile

        temp = tempfile.NamedTemporaryFile(suffix='.bin', prefix='trained_faiss_')
        try:
            success = self._vec_indexer.save_trained_model(temp.name)
            if success:
                self.logger.info('Dumping the trained indexer into PSQL database...')
                temp.seek(0)

                model_data = temp.read()
                model_checksum = self._vec_indexer.index_key

                self._kv_indexer.save_trained_model(model_data, model_checksum)
        finally:
            temp.close()

    @requests(on='/sync')
    def sync(self, parameters: Optional[Dict], **kwargs):
        """
        Sync the data from the PSQLStorage into the FaissSearcher

        :param parameters: dictionary of parameters

            `only_delta`: whether to do delta- or snapshot-based import.
            Snapshot requires a snapshot to have been created in PSQL
            delta imports Documents one by one

            If Faiss has size 0, a new Faiss will be created. (if no index
            exists, delta resolving won't work)

            If `only_delta` is None, we try to resolve it:
            Is there a snapshot in PSQL? If so, use snapshot.
            If there isn't, use delta import

            `use_delta`: Whether to also import delta after a snapshot.
        """
        only_delta = parameters.get('only_delta', not self._use_snapshot())
        use_delta = parameters.get('use_delta', False)

        if only_delta:
            self._sync_only_delta(parameters, **kwargs)

        else:
            self._sync_snapshot(use_delta)

    def _sync_snapshot(self, use_delta):
        self.logger.info('Syncing via snapshot...')

        # reset the faiss indexer first
        self._vec_indexer.clear()

        if self.total_shards:
            updates = self._kv_indexer.get_snapshot(
                shard_id=self.runtime_args.shard_id,
                total_shards=self.total_shards,
                include_metas=False,
                filter_deleted=True,
            )
            timestamp = self._kv_indexer.last_snapshot_timestamp
            self._vec_indexer.load_from_iterator(
                updates, prefetch_size=FAISS_PREFETCH_SIZE
            )

            if use_delta:
                self.logger.info(f'Now adding delta from timestamp {timestamp}')

                deltas = self._kv_indexer.get_delta_updates(
                    shard_id=self.runtime_args.shard_id,
                    total_shards=self.total_shards,
                    timestamp=timestamp,
                    filter_deleted=False,
                )
                # deltas will be like DOC_ID, OPERATION, DATA
                self._vec_indexer.add_delta_updates(deltas)
        else:
            self.logger.warning(
                'total_shards is None, rolling update '
                'via PSQL import will not be possible.'
            )

    def _sync_only_delta(self, parameters, **kwargs):
        """
        `init_faiss` is determined by either being passed or
        by checking if the vec indexer (Faiss) has been initialized.
        If it has already been initialized, then init_faiss cannot be True.
        If it has NOT been initialized, then init_faiss has to be True

        `timestamp`. If train_faiss, then it becomes datetime.min.
        Else, we get it from self._vec_indexer.last_timestamp

        """
        timestamp = parameters.get('timestamp', None)
        init_faiss = parameters.get(
            'init_faiss', not self._vec_indexer_is_initialized()
        )
        if timestamp is None:
            if init_faiss:
                timestamp = datetime.datetime.min
            elif self._vec_indexer.last_timestamp:
                timestamp = self._vec_indexer.last_timestamp
            else:
                self.logger.error(
                    f'No timestamp provided in parameters: '
                    f'"{parameters}" and vec_indexer.last_timestamp'
                    f'was None. Cannot do sync with delta'
                )
                return

        delta_updates = self._kv_indexer.get_delta_updates(
            shard_id=self.runtime_args.shard_id,
            total_shards=self.total_shards,
            timestamp=timestamp,
            filter_deleted=True if init_faiss else False,
        )

        if init_faiss:
            self._vec_indexer.load_from_iterator(delta_updates, FAISS_PREFETCH_SIZE)
        else:
            self._vec_indexer.add_delta_updates(delta_updates)

    @requests(on='/search')
    def search(self, docs: 'DocumentArray', parameters: Optional[Dict] = {}, **kwargs):
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
            kv_parameters['traversal_paths'] = ','.join(
                [
                    path + 'm'
                    for path in kv_parameters.get('traversal_paths', '@r').split(',')
                ]
            )
            self._kv_indexer.search(docs, kv_parameters)
        else:
            self.logger.warning('Indexers have not been initialized. Empty results')
            return

    @requests(on='/prune')
    def prune(self, **kwargs):
        """
        Completely removes rows in PSQL that have been marked for soft-deletion
        """
        self._kv_indexer.prune()

    @requests(on='/snapshot')
    def snapshot(self, **kwargs):
        """
        Create a snapshot of the current database table
        """
        self._kv_indexer.snapshot()

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Index new documents

        NOTE: PSQL has a uniqueness constraint on ID
        """
        self._kv_indexer.add(docs, parameters, **kwargs)

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """
        Update documents in PSQL, based on id
        """
        self._kv_indexer.update(docs, parameters, **kwargs)

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """
        Delete docs from PSQL, based on id.

        By default, it will be a soft delete, where the entry is left in the DB,
        but its data will be set to None
        """
        if 'soft_delete' not in parameters:
            parameters['soft_delete'] = True

        self._kv_indexer.delete(docs, parameters, **kwargs)

    @requests(on='/clear')
    def clear(self, **kwargs):
        self._kv_indexer.clear()
        self._vec_indexer.clear()

    @requests(on='/status')
    def status(self, **kwargs):
        """Return the document containing status information about the indexer."""

        status = Document(
            tags={
                'total_docs': self._kv_indexer.size,
                'active_docs': self._vec_indexer.size,
            }
        )
        return DocumentArray([status])

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump the index

        :param parameters: a dictionary containing the parameters for the dump
        """
        self._kv_indexer.dump(parameters)

    def _use_snapshot(self):
        return self._kv_indexer.snapshot_size > 0

    def _vec_indexer_is_initialized(self):
        return self._vec_indexer and self._vec_indexer.size > 0
