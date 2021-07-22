__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import psycopg2
from psycopg2 import pool
import psycopg2.extras

from jina import DocumentArray, Document
from jina.logging.logger import JinaLogger
from typing import Optional


def doc_without_embedding(d: Document):
    new_doc = Document(d, copy=True)
    new_doc.ClearField('embedding')
    return new_doc.SerializeToString()


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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger('psq_handler')
        self.table = table

        try:
            self.postgreSQL_pool = psycopg2.pool.SimpleConnectionPool(
                1,
                max_connections,
                user=username,
                password=password,
                database=database,
                host=hostname,
                port=port,
            )
            self.use_table()
        except (Exception, psycopg2.Error) as error:
            self.logger.error('Error while connecting to PostgreSQL', error)

    def __enter__(self):
        self.connection = self._get_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            self._close_connection(self.connection)

    def use_table(self):
        """
        Use table if exists or create one if it doesn't.

        Create table if needed with id, vecs and metas.
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        cursor.execute(
            'select exists(select * from information_schema.tables where table_name=%s)',
            (self.table,),
        )
        if cursor.fetchone()[0]:
            self.logger.info('Using existing table')
        else:
            try:
                cursor.execute(
                    f'CREATE TABLE {self.table} ( \
                    ID VARCHAR PRIMARY KEY,  \
                    DOC BYTEA);'
                )
                self.logger.info('Successfully created table')
            except (Exception, psycopg2.Error) as error:
                self.logger.error('Error while creating table!')
        connection.commit()
        self._close_connection(connection)

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
                f'INSERT INTO {self.table} (ID, DOC) VALUES (%s, %s)',
                [
                    (
                        doc.id,
                        doc.SerializeToString(),
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
            f'UPDATE {self.table} SET DOC = %s WHERE ID = %s',
            [
                (
                    doc.SerializeToString(),
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
            f'DELETE FROM {self.table} where (ID) = (%s);',
            [(doc.id,) for doc in docs],
        )
        self.connection.commit()
        return

    def close(self):
        self.postgreSQL_pool.closeall()

    def search(self, docs: DocumentArray, **kwargs):
        """Use the Postgres db as a key-value engine, returning the metadata of a document id"""
        cursor = self.connection.cursor()
        for doc in docs:
            # retrieve metadata
            cursor.execute(f'SELECT DOC FROM {self.table} WHERE ID = %s;', (doc.id,))
            result = cursor.fetchone()
            data = bytes(result[0])
            retrieved_doc = Document(data)
            retrieved_doc.pop('embedding')
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
        cursor.execute(f'SELECT COUNT(*) from {self.table}')
        records = cursor.fetchall()
        return records[0][0]
