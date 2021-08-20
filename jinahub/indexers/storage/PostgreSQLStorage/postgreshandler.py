__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Optional, Generator, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger
from psycopg2 import pool


def doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


SCHEMA_VERSION = 1
SCHEMA_VERSIONS_TABLE_NAME = 'schema_versions'


class PostgreSQLHandler:
    """
    Postgres Handler to connect to the database and can apply add, update, delete and query.

    :param hostname: hostname of the machine
    :param port: the port
    :param username: the username to authenticate
    :param password: the password to authenticate
    :param database: the database name
    :param collection: the collection name
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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger('psq_handler')
        self.table = table
        self.dump_dtype = dump_dtype

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
                doc BYTEA
            );
            INSERT INTO {SCHEMA_VERSIONS_TABLE_NAME} VALUES (%s, %s);''',
            (self.table, SCHEMA_VERSION),
        )

    def _table_exists(self):
        return self._execute_sql_gracefully(
            'SELECT EXISTS(SELECT * FROM information_schema.tables WHERE table_name=%s)',
            (self.table,),
        ).fetchall()[0][0]

    def _assert_table_schema_version(self):
        cursor = self.connection.cursor()
        cursor.execute(
            f'SELECT schema_version FROM {SCHEMA_VERSIONS_TABLE_NAME} WHERE table_name=%s;',
            (self.table,),
        )
        result = cursor.fetchone()
        if result:
            if result[0] != SCHEMA_VERSION:
                raise RuntimeError(
                    f'The schema versions of the database (version {result[0]}) and the Executor (version {SCHEMA_VERSION}) do not match. \
Please migrate your data to the latest version or use an Executor version with a matching schema version.'
                )
        else:
            raise RuntimeError(
                f'The schema versions of the database (NO version number) and the Executor (version {SCHEMA_VERSION}) do not match. \
Please migrate your data to the latest version.'
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
                f'INSERT INTO {self.table} (doc_id, embedding, doc) VALUES (%s, %s, %s)',
                [
                    (
                        doc.id,
                        doc.embedding.astype(self.dump_dtype).tobytes(),
                        doc_without_embedding(doc),
                    )
                    for doc in docs
                ],
            )
        except psycopg2.errors.UniqueViolation as e:
            self.logger.warning(
                f'Document already exists in PSQL database. {e}. Skipping entire transaction...'
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
            f'UPDATE {self.table} SET embedding = %s, doc = %s WHERE doc_id = %s',
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

    def delete(self, docs: DocumentArray, *args, **kwargs):
        """Delete document from the database.

        :param docs: list of Documents
        :param args: other arguments
        :param kwargs: other keyword arguments
        :return record: List of Document's id after deletion
        """
        cursor = self.connection.cursor()
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
        """Use the Postgres db as a key-value engine, returning the metadata of a document id"""
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
            if return_embeddings:
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

    def get_generator(
        self, include_metas=True
    ) -> Generator[Tuple[str, bytes, Optional[bytes]], None, None]:
        connection = self._get_connection()
        # always order the dump by id as integer
        cursor = connection.cursor('generator')  # server-side cursor
        cursor.itersize = 10000
        if include_metas:
            cursor.execute(
                f'SELECT doc_id, embedding, doc FROM {self.table} ORDER BY doc_id'
            )
            for rec in cursor:
                yield rec[0], rec[1], rec[2]
        else:
            cursor.execute(
                f'SELECT doc_id, embedding FROM {self.table} ORDER BY doc_id'
            )
            for rec in cursor:
                yield rec[0], rec[1], None
        self._close_connection(connection)
