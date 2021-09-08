__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Dict, List

from jina import Document, DocumentArray, Executor, requests
from jina_commons import get_logger
from jina_commons.indexers.dump import export_dump_streaming

from .postgreshandler import PostgreSQLHandler


def doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


class PostgreSQLStorage(Executor):
    """:class:`PostgreSQLStorage` PostgreSQL-based Storage Indexer.

    Initialize the PostgreSQLStorage.

    :param hostname: hostname of the machine
    :param port: the port
    :param username: the username to authenticate
    :param password: the password to authenticate
    :param database: the database name
    :param table: the table name to use
    :param default_return_embeddings: whether to return embeddings on search or not
    :param dry_run: If True, no database connection will be build.
    :param virtual_shards: the number of shards to distribute
     the data (used when rolling update on Searcher side)
    :param args: other arguments
    :param kwargs: other keyword arguments
    """

    def __init__(
        self,
        hostname: str = '127.0.0.1',
        port: int = 5432,
        username: str = 'postgres',
        password: str = '123456',
        database: str = 'postgres',
        table: str = 'default_table',
        max_connections=5,
        default_traversal_paths: List[str] = ['r'],
        default_return_embeddings: bool = True,
        dry_run: bool = False,
        virtual_shards: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.default_traversal_paths = default_traversal_paths
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.table = table
        self.logger = get_logger(self)
        self.virtual_shards = virtual_shards
        self.handler = PostgreSQLHandler(
            hostname=self.hostname,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
            table=self.table,
            max_connections=max_connections,
            dry_run=dry_run,
            virtual_shards=virtual_shards,
        )
        self.default_return_embeddings = default_return_embeddings

    @property
    def dump_dtype(self):
        return self.handler.dump_dtype

    @property
    def size(self):
        """Obtain the size of the table

        .. # noqa: DAR201
        """
        with self.handler as postgres_handler:
            return postgres_handler.get_size()

    @property
    def snapshot_size(self):
        """Obtain the size of the table

        .. # noqa: DAR201
        """
        with self.handler as postgres_handler:
            return postgres_handler.get_snapshot_size()

    @requests(on='/index')
    def add(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Add Documents to Postgres

        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        with self.handler as postgres_handler:
            postgres_handler.add(docs.traverse_flat(traversal_paths))

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Updated document from the database.

        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        with self.handler as postgres_handler:
            postgres_handler.update(docs.traverse_flat(traversal_paths))

    @requests(on='/delete')
    def cleanup(self, **kwargs):
        """
        Full deletion of the entries that
        have been marked for soft-deletion
        """
        with self.handler as postgres_handler:
            postgres_handler.cleanup()

    @requests(on='/delete')
    def delete(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Delete document from the database.

        NOTE: This is a soft-deletion, required by the snapshotting
        mechanism in the PSQLFaissCompound

        For a real delete, use the /cleanup endpoint

        :param docs: list of Documents
        :param parameters: parameters to the request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        soft_delete = parameters.get('soft_delete', False)
        with self.handler as postgres_handler:
            postgres_handler.delete(docs.traverse_flat(traversal_paths), soft_delete)

    @requests(on='/dump')
    def dump(self, parameters: Dict, **kwargs):
        """Dump the index

        :param parameters: a dictionary containing the parameters for the dump
        """
        path = parameters.get('dump_path')
        if path is None:
            self.logger.error(f'No "dump_path" provided for {self}')

        shards = int(parameters.get('shards'))
        if shards is None:
            self.logger.error(f'No "shards" provided for {self}')

        include_metas = parameters.get('include_metas', True)

        with self.handler as postgres_handler:
            export_dump_streaming(
                path,
                shards=shards,
                size=self.size,
                data=postgres_handler.get_generator(include_metas=include_metas),
            )

    def close(self) -> None:
        """
        Close the connections in the connection pool
        """
        # TODO perhaps store next_shard_to_use?
        self.handler.close()

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Dict, **kwargs):
        """Get the Documents by the ids of the docs in the DocArray

        :param docs: the DocumentArray to search
         with (they only need to have the `.id` set)
        :param parameters: the parameters to this request
        """
        if docs is None:
            return
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        with self.handler as postgres_handler:
            postgres_handler.search(
                docs.traverse_flat(traversal_paths),
                return_embeddings=parameters.get(
                    'return_embeddings', self.default_return_embeddings
                ),
            )

    @requests(on='/snapshot')
    def snapshot(self, **kwargs):
        """
        Create a snapshot duplicate of the current table
        """
        # TODO argument with table name, database location
        # maybe send to another PSQL instance to avoid perf hit?
        with self.handler as postgres_handler:
            postgres_handler.snapshot()

    def get_snapshot(self, shard_id: int, total_shards: int):
        """Get the data meant out of the snapshot, distributed
        to this shard id, out of X total shards, based on the virtual
        shards allocated.
        """
        shards_to_get = self._vshards_to_get(
            shard_id, total_shards, self.virtual_shards
        )

        with self.handler as postgres_handler:
            return postgres_handler.get_snapshot(shards_to_get)

    @staticmethod
    def _vshards_to_get(shard_id, total_shards, virtual_shards):
        if shard_id > total_shards - 1:
            raise ValueError(
                'shard_id should be 0-indexed out ' 'of range(total_shards)'
            )
        vshards = list(range(virtual_shards))
        vshard_part = (
            virtual_shards // total_shards
        )  # nr of virtual shards given to one shard
        vshard_remainder = virtual_shards % total_shards
        if shard_id == total_shards - 1:
            shards_to_get = vshards[
                shard_id
                * vshard_part : ((shard_id + 1) * vshard_part + vshard_remainder)
            ]
        else:
            shards_to_get = vshards[
                shard_id * vshard_part : (shard_id + 1) * vshard_part
            ]
        return [str(shard_id) for shard_id in shards_to_get]

    def _get_delta(self, shard_id, total_shards, timestamp):
        """
        Get the rows that have changed since the last timestamp, per shard
        """
        shards_to_get = self._vshards_to_get(
            shard_id, total_shards, self.virtual_shards
        )

        with self.handler as postgres_handler:
            return postgres_handler._get_delta(shards_to_get, timestamp)

    @property
    def last_snapshot_timestamp(self):
        """
        Get the timestamp of the snapshot
        """
        with self.handler as postgres_handler:
            return postgres_handler._get_snapshot_timestamp()
