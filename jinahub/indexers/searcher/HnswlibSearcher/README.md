# HnswlibSearcher

**HnswlibSearcher** is a Hnswlib-powered vector Searcher.

Hnswlib is a fast approximate nearest neighbor search library and clustering of dense vectors.






## Usage

Check [tests](tests) for an example on how to use it.

### Loading data

Since this is a "Searcher"-type Executor, it does not _index_ new data. Rather they are write-once classes, which take as data source a `dump_path`. 

This can be provided in different ways:

- in the YAML definition
  
```yaml
jtype: HnswlibSearcher
with:
    dump_path: /tmp/your_dump_location
...
```

- from the `Flow.rolling_update` method. See [docs](https://docs.jina.ai/fundamentals/executor/indexers/).

The folder needs to contain the data exported from your Indexer. Again, see [docs](https://docs.jina.ai/fundamentals/executor/indexers/).


### Inputs 


### Returns

Attaches matches to the Documents sent as inputs, with the id of the match, and its embedding. 


## Reference


- [Hnswlib](https://github.com/nmslib/hnswlib)
