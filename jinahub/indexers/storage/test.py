import numpy as np

from PostgreSQLStorage import PostgreSQLStorage
from jina import DocumentArray, Document

storage = PostgreSQLStorage(hostname="localhost", port=5432, username="postgresadmin", password="1235813", database="postgresdb", table="searcher2")

docs = DocumentArray([
    Document(embedding=np.random.uniform(-1, 1, 128))
    for i in range(500)
])
storage.add(docs, {})