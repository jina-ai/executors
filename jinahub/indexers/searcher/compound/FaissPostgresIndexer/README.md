# FaissPostgresIndexer

**FaissPostgresIndexer** is a compound Executor for Jina, made up of [FaissSearcher](https://hub.jina.ai/executor/gilkzt3f) for performing similarity search on the embeddings, and of [PostgreSQLStorage](https://hub.jina.ai/executor/d45rawx6) for retrieving the metadata of the Documents.

**Note** that you will need a running PostgreSQL database.
This can be a local instance, a Docker image, or a virtual machine in the cloud.
Make sure you have the credentials and connection parameters.

You can start one in a Docker container, like so: 

```bash
docker run -e POSTGRES_PASSWORD=123456  -p 127.0.0.1:5432:5432/tcp postgres:13.2 
```

This is a combination of an Indexer and a Searcher.
Thus, you can perform all the [CRUD operations](https://docs.jina.ai/advanced/experimental/indexers/#crud-operations-and-the-executor-endpoints) on it, while still being able to [search](https://docs.jina.ai/advanced/experimental/indexers/#crud-operations-and-the-executor-endpoints).

Check the usage in [integration tests](../../../../../tests/integration/psql_import/test_import_psql.py).

## Loading data

### Via dump

Since this contains a "Searcher"-type Executor it can take as data source a `dump_path`.

This can be provided in different ways:

- in the YAML definition

```yaml
jtype: FaissPostgresIndexer
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/advanced/experimental/indexers/).

The folder needs to contain the data exported from your Indexer.
Again, see [docs](https://docs.jina.ai/advanced/experimental/indexers/).

Check [integration tests](https://github.com/jina-ai/executors/tree/main/tests/integration/psql_dump_reload) for an example on how to use it.

### Direct import from PSQL

Alternatively, you can use the import from PSQL feature.

This allows you to bypass the expensive writing to disk operation.
It also removes the complexities that can arise when doing this in a cloud environment.
There, you would need to mount the volume in one instance, dump, migrate it, then perform the import.

The recommended way to do it is via the **delta import** method.
This simply imports the data in your PSQL in each shard, distributing it to each shard.
If the data changes during this process, you will have **inconsistency** between the shards.
This can manifest in missing results or inconsistency between the data in the Searcher vs data in PSQL.
Currently, we can only guarantee eventual consistency via manual delta updates.

```python
with get_my_flow() as flow:
    flow.index(your_docs)
    flow.post(
        on='/sync',
        # we need some specific parameters here
        parameters={
            'only_delta': True, 'startup': True
        }
    )
    flow.search(search_docs)
```

While thi can **not** guarantee consistency, it is fast and has low overhead.

If you need consistency, you can use the **snapshot** method.
This guarantees consistency between the shards, but at the cost of speed, extra disk usage, and an extra API call.
This creates a duplicate of your data in another table, at a specific moment.
This way all the shards import from the same view/version of your data.

```python
with get_my_flow() as flow:
    flow.index(your_docs)
    # extra call required: this creates the snapshot
    flow.snapshot()

    flow.post(on='/sync')
    flow.search(search_docs)
```

#### Note on polling

In the above code we use the same Flow for both indexing and searching.
The assumed polling is `all`.
This will lead to warnings if you create more than one shard.
This happens because all PSQL connections, from each shard, will race to create the same data in the underlying database.
However, only one will succeed.
All the others will fail and emit warning.
To avoid this, you will need to split the Flow into two Flows, one for storage, and one for search, and use different polling methods.
This is shown in the integration tests as well.


For more advanced, full-fledged examples, check the [integration tests](https://github.com/jina-ai/executors/tree/main/tests/integration/psql_import).

<!-- version=v0.1 -->

