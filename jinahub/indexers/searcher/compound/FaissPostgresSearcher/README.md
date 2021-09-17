# FaissPostgresSearcher

**FaissPostgresSearcher** is a compound Searcher Executor for Jina, made up of [FaissSearcher](https://hub.jina.ai/executor/gilkzt3f) for performing similarity search on the embeddings, and of [PostgreSQLStorage](https://hub.jina.ai/executor/d45rawx6) for retrieving the metadata of the Documents. 


Additionally, you will need a running PostgreSQL database. This can be a local instance, a Docker image, or a virtual machine in the cloud. Make sure you have the credentials and connection parameters.

You can start one in a Docker container, like so: 

```bash
docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:13.2 
```

## Usage

Check [integration tests](https://github.com/jina-ai/executors/tree/main/tests/integration/psql_dump_reload) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: FaissPostgresSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/advanced/experimental/indexers/).

The folder needs to contain the data exported from your Indexer. Again, see [docs](https://docs.jina.ai/advanced/experimental/indexers/).

#### Direct import from PSQL

Alternatively, you can use the import from PSQL feature. Check the [integration test](https://github.com/jina-ai/executors/tree/main/tests/integration/psql_import).


### Inputs 


### Returns

The FaissSearcher attaches matches to the Documents sent as inputs, with the id of the match, and its embedding.
Then, the PostgresSearcher retrieves the full metadata (original text or image blob) and attaches those to the Document.
You receive back the full Document.



