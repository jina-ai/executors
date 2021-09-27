# FaissPostgresIndexer

**FaissPostgresIndexer** is a compound Executor for Jina, made up of [FaissSearcher](https://hub.jina.ai/executor/gilkzt3f) for performing similarity search on the embeddings, and of [PostgreSQLStorage](https://hub.jina.ai/executor/d45rawx6) for retrieving the metadata of the Documents.

## Usage

**Note** that you will need a running PostgreSQL database. This can be a local instance, a Docker image, or a virtual machine in the cloud. Make sure you have the credentials and connection parameters.

You can start one in a Docker container, like so: 

```bash
docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:13.2 
```

This is a combination of an Indexer and a Searcher. Thus, you can perform all the CRUD operations on it, while still being able to search.

Check the usage in [integration tests](../../../../../tests/integration/psql_import/test_import_psql.py).

### Loading data

Since this contains a "Searcher"-type Executor it can take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  ``
```yaml
jtype: FaissPostgresIndexer
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/advanced/experimental/indexers/).

The folder needs to contain the data exported from your Indexer. Again, see [docs](https://docs.jina.ai/advanced/experimental/indexers/).

Check [integration tests](https://github.com/jina-ai/executors/tree/main/tests/integration/psql_dump_reload) for an example on how to use it.

#### Direct import from PSQL

Alternatively, you can use the import from PSQL feature. 

This allows you to bypass the expensive writing to disk operation. It also removes the complexities that can arise out of doing this in a cloud environment, where you would need to mount the volume in one instance, and then migrate it, and then perform the import. 

Direct import from PSQL can be done in two ways:

1. Via *snapshot*. This guarantees consistency between the shards, but at the cost of speed and extra disk usage. This creates a duplicate of your data in another table, at a specific moment. This way all the shards import from the same view/version of your data.
2. Via *delta import*. This can **not** guarantee consistency, but is faster and with lower overhead. This simply imports the data in your PSQL in each shard, distributing it to each shard. If the data changes during this process, you will have **inconsistency** between the shards. This can manifest in missing results or inconsistency between the data in the Searcher vs data in PSQL.

##### Usage

**snapshotting**

```python
with get_storage_flow() as storage_flow:
    storage_flow.index(your_docs)
    # this creates the snapshot
    storage_flow.snapshot()

with get_query_flow() as query_flow:
    query_flow.post(on='/sync')
    query_flow.search(search_docs)
```

**delta import**

```python
with get_storage_flow() as storage_flow:
    storage_flow.index(your_docs)
    # no snapshotting    
    
with get_query_flow() as query_flow:
    query_flow.post(
        on='/sync',
        # we need some specific parameters here
        parameters={
            'only_delta': True, 'startup': True
        }
    )
    query_flow.search(search_docs)
```


For more advanced, full-fledged examples, check the [integration tests](https://github.com/jina-ai/executors/tree/main/tests/integration/psql_import).

<!-- version=v0.1 -->

