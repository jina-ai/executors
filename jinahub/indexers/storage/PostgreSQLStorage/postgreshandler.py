__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import datetime
import hashlib
from typing import Generator, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from jina import Document, DocumentArray
from jina.logging.logger import JinaLogger
from psycopg2 import pool  # noqa: F401


def doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


SCHEMA_VERSION = 2
SCHEMA_VERSIONS_TABLE_NAME = 'schema_versions'


class PostgreSQLHandler:
    """
    Postgres Handler to connect to the database and
     can apply add, update, delete and query.

    :param hostname: hostname of the machine
    :param port: the port
    :param username: the username to authenticate
    :param password: the password to authenticate
    :param database: the database name
    :param collection: the collection name
    :param dry_run: If True, no database connection will be build
    :param virtual_shards: the number of shards to
    distribute the data (used when rolling update on Searcher side)
    :param args: other arguments
    :param kwargs: other keyword arguments
    """

    def __init__(
        self,
        hostname: str = '127.0.0.1',
        port: int = 5432,
        username: str = 'default_name',
        password: str = 'default_pwd',
        database: str = 'postgres',
        table: Optional[str] = 'default_table',
        max_connections: int = 5,
        dump_dtype: type = np.float64,
        dry_run: bool = False,
        virtual_shards: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger('psq_handler')
        self.table = table
        self.dump_dtype = dump_dtype
        self.virtual_shards = virtual_shards
        self.snapshot_table = 'snapshot'

        if not dry_run:
            self.postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                max_connections,
                user=username,
                password=password,
                database=database,
                host=hostname,
                port=port,
            )
            self._init_table()

    def __enter__(self):
        self.connection = self._get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self._close_connection(self.connection)

    def _init_table(self):
        """
        Use table if exists or create one if it doesn't.

        Create table if needed with id, vecs and metas.
        """
        with self:
            self._create_schema_version()

            if self._table_exists():
                self._assert_table_schema_version()
                self.logger.info('Using existing table')
            else:
                self._create_table()

    def _execute_sql_gracefully(self, statement, data=tuple()):
        try:
            cursor = self.connection.cursor()
            if data:
                cursor.execute(statement, data)
            else:
                cursor.execute(statement)
        except psycopg2.errors.UniqueViolation as error:
            self.logger.debug(f'Error while executing {statement}: {error}.')

        self.connection.commit()
        return cursor

    def _create_schema_version(self):
        self._execute_sql_gracefully(
            f'''CREATE TABLE IF NOT EXISTS {SCHEMA_VERSIONS_TABLE_NAME} (
                table_name varchar,
                schema_version integer
            );'''
        )

    def _create_table(self):
        self._execute_sql_gracefully(
            f'''CREATE TABLE IF NOT EXISTS {self.table} (
                doc_id VARCHAR PRIMARY KEY,
                embedding BYTEA,
                doc BYTEA,
                shard int,
                last_updated timestamp with time zone default current_timestamp
            );
            INSERT INTO {SCHEMA_VERSIONS_TABLE_NAME} VALUES (%s, %s);''',
            (self.table, SCHEMA_VERSION),
        )

    def _table_exists(self):
        return self._execute_sql_gracefully(
            'SELECT EXISTS'
            '('
            'SELECT * FROM information_schema.tables '
            'WHERE table_name=%s'
            ')',
            (self.table,),
        ).fetchall()[0][0]

    def _assert_table_schema_version(self):
        cursor = self.connection.cursor()
        cursor.execute(
            f'SELECT schema_version FROM '
            f'{SCHEMA_VERSIONS_TABLE_NAME} '
            f'WHERE table_name=%s;',
            (self.table,),
        )
        result = cursor.fetchone()
        if result:
            if result[0] != SCHEMA_VERSION:
                raise RuntimeError(
                    f'The schema versions of the database '
                    f'(version {result[0]}) and the Executor '
                    f'(version {SCHEMA_VERSION}) do not match. '
                    f'Please migrate your data to the latest '
                    f'version or use an Executor version with a '
                    f'matching schema version.'
                )
        else:
            raise RuntimeError(
                f'The schema versions of the database '
                f'(NO version number) and the Executor '
                f'(version {SCHEMA_VERSION}) do not match.'
                f'Please migrate your data to the latest version.'
            )

    def add(self, docs: DocumentArray, *args, **kwargs):
        """Insert the documents into the database.

        :param docs: list of Documents
        :param args: other arguments
        :param kwargs: other keyword arguments
        :param args: other arguments
        :param kwargs: other keyword arguments
        :return record: List of Document's id added
        """
        cursor = self.connection.cursor()
        try:
            psycopg2.extras.execute_batch(
                cursor,
                f'INSERT INTO {self.table} '
                f'(doc_id, embedding, doc, shard, last_updated) '
                f'VALUES (%s, %s, %s, %s, current_timestamp)',
                [
                    (
                        doc.id,
                        doc.embedding.astype(self.dump_dtype).tobytes()
                        if doc.embedding is not None
                        else None,
                        doc_without_embedding(doc),
                        self._get_next_shard(doc.id),
                    )
                    for doc in docs
                ],
            )
        except psycopg2.errors.UniqueViolation as e:
            self.logger.warning(
                f'Document already exists in PSQL database.'
                f' {e}. Skipping entire transaction...'
            )
            self.connection.rollback()
        self.connection.commit()

    def update(self, docs: DocumentArray, *args, **kwargs):
        """Updated documents from the database.

        :param docs: list of Documents
        :param args: other arguments
        :param kwargs: other keyword arguments
        :return record: List of Document's id after update
        """
        cursor = self.connection.cursor()
        psycopg2.extras.execute_batch(
            cursor,
            f'UPDATE {self.table}\
             SET embedding = %s,\
             doc = %s,\
             last_updated = current_timestamp \
            WHERE doc_id = %s',
            [
                (
                    doc.embedding.astype(self.dump_dtype).tobytes(),
                    doc_without_embedding(doc),
                    doc.id,
                )
                for doc in docs
            ],
        )
        self.connection.commit()

    def cleanup(self):
        """
        Full deletion of the entries that
        have been marked for soft-deletion
        """
        cursor = self.connection.cursor()
        psycopg2.extras.execute_batch(
            cursor,
            f'DELETE FROM {self.table} ' f'WHERE doc == NULL',
        )
        self.connection.commit()
        return

    def delete(self, docs: DocumentArray, soft_delete=False, *args, **kwargs):
        """Delete document from the database.

        NOTE: This can be a soft-deletion, required by the snapshotting
        mechanism in the PSQLFaissCompound

        For a real delete, use the /cleanup endpoint

        :param docs: list of Documents
        :param args: other arguments
        :param soft_delete:
        :param kwargs: other keyword arguments
        :return record: List of Document's id after deletion
        """
        cursor = self.connection.cursor()
        if soft_delete:
            self.logger.warning(
                'Performing soft-delete. Use /cleanup or a hard '
                'delete to delete the records'
            )
            psycopg2.extras.execute_batch(
                cursor,
                f'UPDATE {self.table} '
                f'SET embedding = NULL, '
                f'doc = NULL, '
                f'last_updated = current_timestamp '
                f'WHERE doc_id = %s;',
                [(doc.id,) for doc in docs],
            )
        else:
            psycopg2.extras.execute_batch(
                cursor,
                f'DELETE FROM {self.table} WHERE doc_id = %s;',
                [(doc.id,) for doc in docs],
            )
        self.connection.commit()
        return

    def close(self):
        self.postgreSQL_pool.closeall()

    def search(self, docs: DocumentArray, return_embeddings: bool = True, **kwargs):
        """Use the Postgres db as a key-value engine,
        returning the metadata of a document id"""
        if return_embeddings:
            embeddings_field = ', embedding '
        else:
            embeddings_field = ''
        cursor = self.connection.cursor()
        for doc in docs:
            # retrieve metadata
            cursor.execute(
                f'SELECT doc {embeddings_field} FROM {self.table} WHERE doc_id = %s;',
                (doc.id,),
            )
            result = cursor.fetchone()
            data = bytes(result[0])
            retrieved_doc = Document(data)
            if return_embeddings and result[1] is not None:
                embedding = np.frombuffer(result[1], dtype=self.dump_dtype)
                retrieved_doc.embedding = embedding
            doc.MergeFrom(retrieved_doc)

    def _close_connection(self, connection):
        # restore it to the pool
        self.postgreSQL_pool.putconn(connection)

    def _get_connection(self):
        # by default psycopg2 is not auto-committing
        # this means we can have rollbacks
        # and maintain ACID-ity
        connection = self.postgreSQL_pool.getconn()
        connection.autocommit = False
        return connection

    def get_size(self):
        cursor = self.connection.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM {self.table}')
        records = cursor.fetchall()
        return records[0][0]

    def _get_next_shard(self, doc_id: str):
        sha = hashlib.sha256()
        sha.update(bytes(doc_id, 'utf-8'))
        return int(sha.hexdigest(), 16) % self.virtual_shards

    def snapshot(self):
        """
        Saves the state of the data table in a new table

        Required to be done in two steps because
        1. create table like ... doesn't include data
        2. insert into .. (select ...) doesn't include primary key definitions
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                f'drop table if exists {self.snapshot_table}; '
                f'create table {self.snapshot_table} '
                f'(like {self.table} including all);'
            )
            self.connection.commit()
            cursor = self.connection.cursor()
            cursor.execute(
                f'insert into {self.snapshot_table} (select * from {self.table});'
            )
            self.connection.commit()
            self.logger.info('Successfully created snapshot')
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f'Error snapshotting: {error}')
            self.connection.rollback()

    def get_snapshot(self, shards_to_get: List[int]):
        """
        Get the data from the snapshot, for a specific range of virtual shards
        """
        shards_quoted = tuple(int(shard) for shard in shards_to_get)
        try:
            cursor = self.connection.cursor('snapshot')
            cursor.itersize = 10000
            cursor.execute(
                f'SELECT doc_id, embedding from {self.snapshot_table} '
                f'WHERE shard in %s '
                f'ORDER BY doc_id',
                (shards_quoted,),
            )
            for rec in cursor:
                vec = (
                    np.frombuffer(rec[1], dtype=self.dump_dtype)
                    if rec[1] is not None
                    else None
                )
                yield rec[0], vec
        except (Exception, psycopg2.Error) as error:
            self.logger.error(f'Error importing snapshot: {error}')
            self.connection.rollback()
        self.connection.commit()

    def get_generator(
        self, include_metas=True
    ) -> Generator[Tuple[str, bytes, Optional[bytes]], None, None]:
        connection = self._get_connection()
        cursor = connection.cursor('generator')  # server-side cursor
        cursor.itersize = 10000
        if include_metas:
            cursor.execute(
                f'SELECT doc_id, embedding, doc FROM {self.table} ORDER BY doc_id'
            )
            for rec in cursor:
                yield rec[0], np.frombuffer(rec[1]) if rec[
                    1
                ] is not None else None, rec[2]
        else:
            cursor.execute(
                f'SELECT doc_id, embedding FROM {self.table} ORDER BY doc_id'
            )
            for rec in cursor:
                yield rec[0], np.frombuffer(rec[1]) if rec[
                    1
                ] is not None else None, None
        self._close_connection(connection)

    def _get_snapshot_timestamp(self):
        """Get the timestamp of the snapshot"""
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f'SELECT MAX(last_updated) FROM {self.snapshot_table}')
            for rec in cursor:
                return rec[0]
        except Exception as e:
            self.logger.error(f'Could not obtain timestamp from snapshot: {e}')

    def _get_delta(
        self, shards_to_get, timestamp
    ) -> Generator[Tuple[str, bytes, datetime.datetime], None, None]:
        connection = self._get_connection()
        cursor = connection.cursor('generator')  # server-side cursor
        cursor.itersize = 10000
        shards_quoted = tuple(int(shard) for shard in shards_to_get)
        cursor.execute(
            f'SELECT doc_id, embedding, last_updated '
            f'from {self.table} '
            f'WHERE shard in %s '
            f'and last_updated > %s '
            f'ORDER BY doc_id',
            (shards_quoted, timestamp),
        )
        for rec in cursor:
            second_val = (
                np.frombuffer(rec[1], dtype=self.dump_dtype)
                if rec[1] is not None
                else None
            )
            yield rec[0], second_val, rec[2]
        self._close_connection(connection)

    def get_snapshot_size(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(f'SELECT COUNT(*) FROM {self.snapshot_table}')
            records = cursor.fetchall()
            return records[0][0]
        except Exception as e:
            self.logger.error(f'Could not get size of snapshot: {e}')
