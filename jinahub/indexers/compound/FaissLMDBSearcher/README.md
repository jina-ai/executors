# FaissLMDBSearcher

**FaissLMDBSearcher** is a compound Searcher Executor for Jina, made up of [FaissSearcher](https://hub.jina.ai/executor/gilkzt3f) for performing similarity search on the embeddings, and of [FileSearcher](https://hub.jina.ai/executor/cmykq7s7) for retrieving the metadata of the Documents. 


## Usage 

Check [integration tests](https://github.com/jina-ai/executors/tree/main/tests/integration/lmdb_dump_reload) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: FaissLMDBSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/advanced/experimental/indexers/).

The folder needs to contain the data exported from your Indexer. 


## Reference
- [indexer docs](https://docs.jina.ai/advanced/experimental/indexers/).

<!-- version=v0.3 -->
